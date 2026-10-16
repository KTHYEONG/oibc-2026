"""구성 배정 저장·조회."""

from __future__ import annotations

import fcntl
import hashlib
import io
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

import pandas as pd
from pydantic import BaseModel, ConfigDict

from src.oibc.compose.constraints import CompositionError, validate_assignment
from src.oibc.config import REFERENCE_COMP_ID, ArtifactPaths, CompetitionSpec, ComposeConfig, repo_root

logger = logging.getLogger(__name__)

_COMP_ID = re.compile(r"[A-Za-z0-9_.-]+")


class CompositionRecord(BaseModel):
    """저장된 구성의 메타데이터."""

    model_config = ConfigDict(frozen=True)

    comp_id: str
    created_at: datetime
    method: Literal["spatial_anticluster"]
    config: dict[str, object]
    git_commit: str | None
    assignment_sha256: str
    group_capacity_kw: dict[str, float]
    n_per_group: dict[str, int]


def _resolve_spec(spec: CompetitionSpec | None) -> CompetitionSpec:
    return spec if spec is not None else CompetitionSpec()


def _resolve_paths(paths: ArtifactPaths | None) -> ArtifactPaths:
    return paths if paths is not None else ArtifactPaths()


def _check_comp_id(comp_id: str) -> None:
    if not comp_id or comp_id in {".", ".."} or _COMP_ID.fullmatch(comp_id) is None:
        raise CompositionError(f"composition: unsafe comp_id {comp_id!r}")
    if comp_id == REFERENCE_COMP_ID:
        raise CompositionError(f"composition: reserved comp_id {comp_id!r}")


def _canonical_csv_bytes(assignment: pd.DataFrame) -> bytes:
    if list(assignment.columns) != ["site_id", "group"]:
        raise CompositionError(f"assignment: columns must be ['site_id', 'group'] got={list(assignment.columns)}")
    work = pd.DataFrame(
        {
            "site_id": assignment["site_id"].astype("string"),
            "group": assignment["group"].astype("string"),
        }
    )
    if work.isna().any().any():
        raise CompositionError("assignment: null value")
    work = work.sort_values("site_id", kind="mergesort").reset_index(drop=True)
    buf = io.StringIO()
    work.to_csv(buf, index=False, lineterminator="\n")
    return buf.getvalue().encode("utf-8")


def assignment_sha256(assignment: pd.DataFrame) -> str:
    """site_id 오름차순 정규 CSV 직렬화의 sha256을 반환한다."""
    return hashlib.sha256(_canonical_csv_bytes(assignment)).hexdigest()


def _git_commit() -> str | None:
    exe = shutil.which("git")
    if exe is None:
        return None
    try:
        out = subprocess.run(  # noqa: S603
            [exe, "rev-parse", "HEAD"], cwd=repo_root(), capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    sha = out.stdout.strip()
    return sha if out.returncode == 0 and sha else None


@contextmanager
def _directory_lock(base: Path) -> Iterator[None]:
    fd = os.open(base, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def write_composition(
    assignment: pd.DataFrame,
    comp_id: str,
    cfg: ComposeConfig,
    sites: pd.DataFrame,
    *,
    paths: ArtifactPaths | None = None,
    spec: CompetitionSpec | None = None,
) -> Path:
    """검증된 배정을 <compositions>/<comp_id>.csv와 .json으로 원자적으로 쓰고 CSV 경로를 반환한다."""
    _check_comp_id(comp_id)
    cfg_spec = _resolve_spec(spec)
    caps = validate_assignment(assignment, sites, spec=cfg_spec)
    base = _resolve_paths(paths).compositions
    csv_path = base / f"{comp_id}.csv"
    json_path = base / f"{comp_id}.json"
    counts = assignment["group"].astype("string").value_counts()
    payload = _canonical_csv_bytes(assignment)
    digest = hashlib.sha256(payload).hexdigest()
    record = CompositionRecord(
        comp_id=comp_id,
        created_at=datetime.now(ZoneInfo("Asia/Seoul")),
        method="spatial_anticluster",
        config=cfg.model_dump(mode="json"),
        git_commit=_git_commit(),
        assignment_sha256=digest,
        group_capacity_kw={g: float(caps[g]) for g in cfg_spec.groups},
        n_per_group={g: int(counts.get(g, 0)) for g in cfg_spec.groups},
    )
    base.mkdir(parents=True, exist_ok=True)
    tmp_csv: Path | None = None
    tmp_json: Path | None = None
    with _directory_lock(base):
        if csv_path.exists() or csv_path.is_symlink() or json_path.exists() or json_path.is_symlink():
            raise CompositionError(f"composition: already exists comp_id={comp_id!r}")
        published = False
        try:
            with tempfile.NamedTemporaryFile(dir=base, delete=False, suffix=".csv") as f:
                tmp_csv = Path(f.name)
                f.write(payload)
            with tempfile.NamedTemporaryFile(dir=base, delete=False, suffix=".json", mode="w", encoding="utf-8") as f:
                tmp_json = Path(f.name)
                json.dump(record.model_dump(mode="json"), f, ensure_ascii=False, indent=2)
                f.write("\n")
            tmp_csv.replace(csv_path)
            published = True
            tmp_json.replace(json_path)
        except Exception:
            if published:
                csv_path.unlink(missing_ok=True)
            raise
        finally:
            for tmp in (tmp_csv, tmp_json):
                if tmp is not None:
                    tmp.unlink(missing_ok=True)
    logger.info("[VPP] composition written comp_id=%s sha256=%s", comp_id, digest)
    return csv_path


def load_composition(
    comp_id: str,
    sites: pd.DataFrame,
    *,
    paths: ArtifactPaths | None = None,
    spec: CompetitionSpec | None = None,
) -> tuple[pd.DataFrame, str]:
    """저장된 배정을 읽어 검증하고 (배정, assignment_sha256)를 반환한다."""
    _check_comp_id(comp_id)
    cfg_spec = _resolve_spec(spec)
    base = _resolve_paths(paths).compositions
    csv_path = base / f"{comp_id}.csv"
    json_path = base / f"{comp_id}.json"
    if not csv_path.is_file() or not json_path.is_file():
        raise CompositionError(f"composition: missing file comp_id={comp_id!r}")
    try:
        record = CompositionRecord.model_validate_json(json_path.read_text(encoding="utf-8"))
        payload = csv_path.read_bytes()
    except (OSError, ValueError) as e:
        raise CompositionError(f"composition: invalid record comp_id={comp_id!r} ({e})") from e
    if record.comp_id != comp_id:
        raise CompositionError(f"composition: record id mismatch comp_id={comp_id!r}")
    digest = hashlib.sha256(payload).hexdigest()
    if digest != record.assignment_sha256:
        raise CompositionError(f"composition: hash mismatch comp_id={comp_id!r}")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as e:
        raise CompositionError(f"composition: not utf-8 comp_id={comp_id!r}") from e
    try:
        assignment = pd.read_csv(io.StringIO(text), dtype="string", keep_default_na=False)
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as e:
        raise CompositionError(f"composition: invalid CSV comp_id={comp_id!r}") from e
    if assignment_sha256(assignment) != digest:
        raise CompositionError(f"composition: noncanonical CSV comp_id={comp_id!r}")
    validate_assignment(assignment, sites, spec=cfg_spec)
    ordered = assignment.sort_values("site_id", kind="mergesort").reset_index(drop=True)
    ordered = pd.DataFrame(
        {
            "site_id": ordered["site_id"].astype("string"),
            "group": ordered["group"].astype("string"),
        }
    )
    logger.info("[VPP] composition loaded comp_id=%s sha256=%s", comp_id, digest)
    return ordered, digest
