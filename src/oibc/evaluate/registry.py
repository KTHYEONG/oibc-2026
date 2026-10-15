"""실험 기록 append-only JSONL 레지스트리."""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator, model_validator

from src.oibc.config import ArtifactPaths

logger = logging.getLogger(__name__)

_RUN_ID = re.compile(r"[A-Za-z0-9_.-]+")


class RegistryError(ValueError):
    """레지스트리 계약 위반."""


def _reject_nonfinite(value: object, where: str) -> None:
    # pydantic JSON 직렬화는 NaN·inf를 조용히 null로 바꾸므로 기록 전에 막는다.
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{where} must be finite")
    if isinstance(value, dict):
        for k, v in value.items():
            _reject_nonfinite(v, f"{where}.{k}")
    elif isinstance(value, list | tuple):
        for i, v in enumerate(value):
            _reject_nonfinite(v, f"{where}[{i}]")


class ExperimentRecord(BaseModel):
    """실험 1건의 재현 정보와 집계 지표."""

    model_config = ConfigDict(frozen=True)
    run_id: str
    created_at: datetime
    hypothesis: str
    config: dict[str, Any]
    git_commit: str | None
    seeds: tuple[int, ...]
    folds: tuple[str, ...]
    metrics: dict[str, dict[str, float]]
    comparison: dict[str, Any] | None = None
    baseline_run_id: str | None = None
    note: str = ""

    @field_validator("run_id")
    @classmethod
    def _check_run_id(cls, v: str) -> str:
        if not v or _RUN_ID.fullmatch(v) is None:
            raise ValueError("run_id must match [A-Za-z0-9_.-]+")
        return v

    @field_validator("created_at")
    @classmethod
    def _check_created_at(cls, v: datetime) -> datetime:
        if v.tzinfo is None or v.tzinfo.utcoffset(v) is None:
            raise ValueError("created_at must be tz-aware")
        return v

    @field_validator("hypothesis")
    @classmethod
    def _check_hypothesis(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("hypothesis must be non-empty")
        return v

    @model_validator(mode="after")
    def _check_metrics(self) -> ExperimentRecord:
        folds = set(self.folds)
        seeds = set(self.seeds)
        for key, values in self.metrics.items():
            parts = key.split("/")
            if len(parts) != 2 or not parts[0] or not parts[1]:
                raise ValueError(f"metrics key {key!r} must look like '<fold>/<seed>'")
            try:
                seed = int(parts[1])
            except ValueError:
                raise ValueError(f"metrics key {key!r} must look like '<fold>/<seed>'") from None
            if parts[0] not in folds or seed not in seeds:
                raise ValueError(f"metrics key {key!r} outside folds/seeds")
            for name, value in values.items():
                if not math.isfinite(value):
                    raise ValueError(f"metrics[{key}][{name}] must be finite")
        _reject_nonfinite(self.config, "config")
        _reject_nonfinite(self.comparison, "comparison")
        return self


def _registry_path(path: Path | None) -> Path:
    return path if path is not None else ArtifactPaths().registry


def _load(path: Path) -> list[ExperimentRecord]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    if text and not text.endswith("\n"):
        raise RegistryError(f"{path}: missing trailing newline")
    records: list[ExperimentRecord] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise RegistryError(f"{path}: line {lineno}: blank line")
        try:
            records.append(ExperimentRecord.model_validate_json(line))
        except ValidationError as e:
            raise RegistryError(f"{path}: line {lineno}: {e}") from e
    return records


def append_record(record: ExperimentRecord, path: Path | None = None) -> None:
    """레코드를 JSONL 한 줄로 추가한다. 같은 run_id가 이미 있으면 RegistryError."""
    target = _registry_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    records = _load(target) if target.exists() else []
    if any(r.run_id == record.run_id for r in records):
        raise RegistryError(f"{target}: duplicate run_id {record.run_id!r}")
    with open(target, "a", encoding="utf-8") as fh:
        fh.write(record.model_dump_json(ensure_ascii=False) + "\n")
    logger.info("[EVAL] registry append run_id=%s n_total=%d", record.run_id, len(records) + 1)


def read_records(path: Path | None = None) -> list[ExperimentRecord]:
    """레지스트리의 모든 레코드를 기록 순서대로 반환한다. 파일이 없으면 빈 목록."""
    return _load(_registry_path(path))


def count_comparisons(records: Sequence[ExperimentRecord], baseline_run_id: str | None = None) -> int:
    """비교 결과가 기록된 실험 수를 반환한다(다중 비교 횟수 추적용)."""
    return sum(
        1
        for r in records
        if r.comparison is not None and (baseline_run_id is None or r.baseline_run_id == baseline_run_id)
    )
