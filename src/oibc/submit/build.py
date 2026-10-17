"""배정과 예측으로 제출 폴더 생성."""

from __future__ import annotations

import hashlib
import logging
import math
import re
import shutil
import subprocess
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from src.oibc.config import ArtifactPaths, CompetitionSpec
from src.oibc.submit.validate import (
    SubmissionError,
    SubmissionReference,
    assert_valid,
    load_reference,
    validate_submission,
)

logger = logging.getLogger(__name__)

_SUB_ID_RE = re.compile(r"[A-Za-z0-9_.-]+")


class SubmissionManifest(BaseModel):
    """제출 1건의 재현 정보와 무결성 해시."""

    model_config = ConfigDict(frozen=True)

    sub_id: str
    created_at: datetime
    run_id: str | None
    composition_id: str | None
    git_commit: str | None
    composition_sha256: str
    forecast_sha256: str
    group_capacity_kw: dict[str, float]
    group_forecast_total_kwh: dict[str, float]
    local_metrics: dict[str, float]
    warnings: tuple[str, ...]
    leaderboard_score: float | None = None
    note: str = ""
    calibration: dict[str, float] | None = None

    @field_validator("sub_id")
    @classmethod
    def _check_sub_id(cls, v: str) -> str:
        if not v or _SUB_ID_RE.fullmatch(v) is None:
            raise ValueError("invalid sub_id")
        return v

    @field_validator("created_at")
    @classmethod
    def _check_created_at(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("tz-aware created_at required")
        return v

    @field_validator("local_metrics")
    @classmethod
    def _check_metrics(cls, v: dict[str, float]) -> dict[str, float]:
        for key, val in v.items():
            if not math.isfinite(val):
                raise ValueError(f"non-finite metric: {key}")
        return v

    @field_validator("calibration")
    @classmethod
    def _check_calibration(cls, v: dict[str, float] | None) -> dict[str, float] | None:
        if v is None:
            return v
        for key, val in v.items():
            if not math.isfinite(val):
                raise ValueError(f"non-finite calibration: {key}")
        return v


def _git_commit() -> str | None:
    try:
        res = subprocess.run(  # noqa: S603
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if res.returncode != 0:
        return None
    commit = res.stdout.strip()
    return commit or None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_assignment_keys(assignment: pd.DataFrame) -> None:
    if assignment.empty:
        raise SubmissionError("assignment: empty")
    if list(assignment.columns) != ["site_id", "group"]:
        raise SubmissionError("assignment: columns must be site_id,group")
    sids = [str(v) for v in assignment["site_id"].tolist()]
    if any(s == "" for s in sids):
        raise SubmissionError("assignment: empty site_id")
    if len(set(sids)) != len(sids):
        raise SubmissionError("assignment: duplicate site_id")


def _check_forecast_keys(forecast: pd.DataFrame) -> None:
    if forecast.empty:
        raise SubmissionError("forecast: empty")
    if list(forecast.columns) != ["group", "ts", "forecast_kwh"]:
        raise SubmissionError("forecast: columns must be group,ts,forecast_kwh")


def _normalize_forecast_values(values: pd.Series) -> list[float]:
    out: list[float] = []
    for v in values.tolist():
        try:
            f = float(v)
        except (TypeError, ValueError) as e:
            raise SubmissionError(f"forecast: bad value {v!r}") from e
        if math.isnan(f):
            raise SubmissionError("forecast: nan value")
        if not math.isfinite(f):
            raise SubmissionError("forecast: non-finite value")
        if f < 0:
            raise SubmissionError("forecast: negative value")
        out.append(0.0 if f == 0 else f)
    return out


def write_submission(
    assignment: pd.DataFrame,
    forecast: pd.DataFrame,
    sub_id: str,
    *,
    run_id: str | None = None,
    composition_id: str | None = None,
    local_metrics: Mapping[str, float] | None = None,
    note: str = "",
    calibration: Mapping[str, float] | None = None,
    out_root: Path | None = None,
    ref: SubmissionReference | None = None,
    spec: CompetitionSpec | None = None,
) -> Path:
    """배정과 집합 예측으로 제출 폴더를 만들고 검증한 뒤 경로를 반환한다. 위반 시 SubmissionError."""
    cfg = spec if spec is not None else CompetitionSpec()
    if not sub_id or _SUB_ID_RE.fullmatch(sub_id) is None:
        raise SubmissionError(f"invalid sub_id: {sub_id!r}")
    cal_record: dict[str, float] | None = None
    if calibration is not None:
        try:
            cal_record = {str(k): float(v) for k, v in dict(calibration).items()}
        except (TypeError, ValueError) as e:
            raise SubmissionError(f"calibration: bad value ({e})") from e
        if any(not math.isfinite(v) for v in cal_record.values()):
            raise SubmissionError("calibration: non-finite value")
    _check_assignment_keys(assignment)
    _check_forecast_keys(forecast)
    reference = ref if ref is not None else load_reference(spec=cfg)
    root = out_root if out_root is not None else ArtifactPaths().submissions
    target = root / sub_id
    if target.exists():
        raise SubmissionError(f"submission exists: {sub_id}")
    vals = _normalize_forecast_values(forecast["forecast_kwh"])
    # 오프셋이 섞인 object 열은 pd.to_datetime이 거부하고 utc=True는 naive를 UTC로 가정하므로 원소별로 파싱한다.
    try:
        stamps = [pd.Timestamp(v) for v in forecast["ts"].tolist()]
    except (TypeError, ValueError) as e:
        raise SubmissionError(f"forecast: bad ts ({e})") from e
    if any(t is pd.NaT or t.tzinfo is None for t in stamps):
        raise SubmissionError("forecast: tz-naive or missing ts")
    ts_kst = pd.DatetimeIndex([t.tz_convert("Asia/Seoul") for t in stamps])
    instant_to_raw = dict(zip(reference.score_index, reference.score_ts, strict=True))
    ref_set = set(reference.score_index)
    if any(t not in ref_set for t in ts_kst):
        raise SubmissionError("forecast: unknown ts")
    per_group: dict[str, set[pd.Timestamp]] = {g: set() for g in cfg.groups}
    for g, t in zip([str(v) for v in forecast["group"].tolist()], ts_kst, strict=True):
        if g not in per_group:
            raise SubmissionError(f"forecast: invalid group {g!r}")
        # 오프셋 표기만 다른 같은 순간도 중복이므로 KST 변환 후 판정한다.
        if t in per_group[g]:
            raise SubmissionError("forecast: duplicate pair")
        per_group[g].add(t)
    for g in cfg.groups:
        if per_group[g] != ref_set:
            raise SubmissionError(f"forecast: incomplete ts group={g}")
    comp_df = pd.DataFrame({"site_id": [str(v) for v in assignment["site_id"].tolist()], "group": [str(v) for v in assignment["group"].tolist()]})
    comp_df = comp_df.sort_values("site_id", kind="mergesort").reset_index(drop=True)
    ts_raw_out = [instant_to_raw[t] for t in ts_kst]
    fc_df = pd.DataFrame(
        {
            "group": [str(v) for v in forecast["group"].tolist()],
            "ts": ts_raw_out,
            "_ts": list(ts_kst),
            "forecast_kwh": vals,
        }
    )
    fc_df = fc_df.sort_values(["group", "_ts"], kind="mergesort").drop(columns="_ts").reset_index(drop=True)
    tmp = root / f".{sub_id}.tmp"
    if tmp.exists():
        raise SubmissionError(f"submission tmp exists: {tmp}")
    try:
        tmp.mkdir(parents=True, exist_ok=False)
        comp_path = tmp / "composition.csv"
        fc_path = tmp / "forecast.csv"
        comp_df.to_csv(comp_path, index=False, encoding="utf-8", lineterminator="\n")
        fc_df.to_csv(fc_path, index=False, encoding="utf-8", lineterminator="\n")
        report = validate_submission(comp_path, fc_path, reference, cfg)
        assert_valid(report)
        manifest = SubmissionManifest(
            sub_id=sub_id,
            created_at=datetime.now().astimezone(),
            run_id=run_id,
            composition_id=composition_id,
            git_commit=_git_commit(),
            composition_sha256=_sha256(comp_path),
            forecast_sha256=_sha256(fc_path),
            group_capacity_kw=dict(report.group_capacity_kw),
            group_forecast_total_kwh=dict(report.group_forecast_total_kwh),
            local_metrics=dict(local_metrics) if local_metrics is not None else {},
            warnings=tuple(report.warnings),
            leaderboard_score=None,
            note=note,
            calibration=cal_record,
        )
        (tmp / "manifest.json").write_text(
            manifest.model_dump_json(indent=2, ensure_ascii=False), encoding="utf-8"
        )
        if target.exists():
            raise SubmissionError(f"submission exists: {sub_id}")
        tmp.rename(target)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    logger.info(
        "[SUBMIT] write sub_id=%s rows=%d/%d sha=%s",
        sub_id,
        len(comp_df),
        len(fc_df),
        _sha256(target / "forecast.csv")[:12],
    )
    return target


def record_leaderboard_score(sub_dir: Path, score: float) -> SubmissionManifest:
    """manifest의 leaderboard_score를 한 번만 기록하고 갱신된 manifest를 반환한다."""
    if not math.isfinite(score):
        raise SubmissionError("score must be finite")
    manifest_path = sub_dir / "manifest.json"
    try:
        current = SubmissionManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as e:
        raise SubmissionError(f"cannot read manifest ({e})") from e
    if current.leaderboard_score is not None:
        raise SubmissionError("leaderboard_score already set")
    for name, expected in (
        ("composition.csv", current.composition_sha256),
        ("forecast.csv", current.forecast_sha256),
    ):
        try:
            actual = _sha256(sub_dir / name)
        except OSError as e:
            raise SubmissionError(f"cannot read {name} ({e})") from e
        if actual != expected:
            raise SubmissionError(f"{name} modified")
    updated = current.model_copy(update={"leaderboard_score": score})
    tmp_path = sub_dir / ".manifest.json.tmp"
    tmp_path.write_text(updated.model_dump_json(indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.rename(manifest_path)
    return updated
