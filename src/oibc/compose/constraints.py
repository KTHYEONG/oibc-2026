"""배정 분할 계약 검사."""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd

from src.oibc.config import CompetitionSpec

logger = logging.getLogger(__name__)


class CompositionError(ValueError):
    """구성 계약 위반."""


def _resolve_spec(spec: CompetitionSpec | None) -> CompetitionSpec:
    return spec if spec is not None else CompetitionSpec()


def validate_assignment(
    assignment: pd.DataFrame, sites: pd.DataFrame, *, spec: CompetitionSpec | None = None
) -> dict[str, float]:
    """(site_id, group) 배정이 sites 전체의 2집합 분할이고 용량 하한을 지키는지 검사해 그룹별 용량(kW)을 반환한다."""
    cfg = _resolve_spec(spec)
    if list(assignment.columns) != ["site_id", "group"]:
        raise CompositionError(f"assignment: columns must be ['site_id', 'group'] got={list(assignment.columns)}")
    if not sites.columns.is_unique:
        raise CompositionError("sites: duplicate columns")
    for col in ("site_id", "capacity_kw"):
        if col not in sites.columns:
            raise CompositionError(f"sites: missing column {col!r}")
    if len(sites) != cfg.n_sites:
        raise CompositionError(f"sites: count={len(sites)} expected={cfg.n_sites}")
    site_ids = sites["site_id"].astype("string")
    if site_ids.isna().any():
        raise CompositionError("sites: null site_id")
    if site_ids.duplicated().any():
        raise CompositionError("sites: duplicate site_id")
    cap = pd.to_numeric(sites["capacity_kw"], errors="coerce").to_numpy(dtype="float64")
    if not np.all(np.isfinite(cap)) or bool((cap <= 0).any()):
        raise CompositionError("sites: capacity_kw must be finite and > 0")
    if assignment["site_id"].isna().any() or assignment["group"].isna().any():
        raise CompositionError("assignment: null value")
    if assignment["site_id"].astype("string").duplicated().any():
        raise CompositionError("assignment: duplicate site_id")
    want = set(site_ids.astype("string").tolist())
    got = set(assignment["site_id"].astype("string").tolist())
    if got != want:
        missing = sorted(want - got)[:5]
        extra = sorted(got - want)[:5]
        raise CompositionError(f"assignment: site set mismatch missing={missing} extra={extra}")
    allowed = set(cfg.groups)
    if (~assignment["group"].astype("string").isin(allowed)).any():
        bad = sorted(set(assignment["group"].astype("string").tolist()) - allowed)
        raise CompositionError(f"assignment: unknown group {bad}")
    counts = assignment["group"].astype("string").value_counts()
    for grp in cfg.groups:
        if counts.get(grp, 0) < 1:
            raise CompositionError(f"assignment: empty group {grp}")
    cap_of: dict[str, float] = dict(zip(site_ids.astype("string").tolist(), [float(v) for v in cap], strict=True))
    caps: dict[str, float] = {}
    for grp in cfg.groups:
        members = assignment.loc[assignment["group"].astype("string") == grp, "site_id"].astype("string").tolist()
        try:
            caps[grp] = math.fsum(cap_of[s] for s in members)
        except OverflowError as e:
            raise CompositionError(f"assignment: non-finite capacity group={grp}") from e
    for grp in cfg.groups:
        if caps[grp] < cfg.min_group_capacity_kw:
            raise CompositionError(f"assignment: under capacity group={grp} cap={caps[grp]}")
    logger.info("[VPP] assignment valid caps=%s n=%s", caps, {g: int(counts[g]) for g in cfg.groups})
    return caps
