"""기준 배정 생성."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.oibc.compose.constraints import CompositionError, validate_assignment
from src.oibc.config import CompetitionSpec


def _resolve_spec(spec: CompetitionSpec | None) -> CompetitionSpec:
    return spec if spec is not None else CompetitionSpec()


def _sorted_sites(sites: pd.DataFrame) -> pd.DataFrame:
    if sites.columns.has_duplicates:
        raise CompositionError("sites: duplicate columns")
    if "site_id" not in sites.columns or "capacity_kw" not in sites.columns:
        raise CompositionError("sites: missing site_id or capacity_kw")
    work = sites[["site_id", "capacity_kw"]].copy()
    work["site_id"] = work["site_id"].astype("string")
    if work["site_id"].isna().any():
        raise CompositionError("sites: null site_id")
    if work["site_id"].duplicated().any():
        raise CompositionError("sites: duplicate site_id")
    cap = pd.to_numeric(work["capacity_kw"], errors="coerce").to_numpy(dtype="float64")
    if not np.all(np.isfinite(cap)) or bool((cap <= 0).any()):
        raise CompositionError("sites: capacity_kw must be finite and > 0")
    work["capacity_kw"] = cap
    return work.sort_values("site_id", kind="mergesort").reset_index(drop=True)


def random_assignment(
    sites: pd.DataFrame,
    rng: np.random.Generator,
    *,
    spec: CompetitionSpec | None = None,
    max_tries: int = 1000,
) -> pd.DataFrame:
    """사이트별 독립 균등 무작위 배정 중 용량 하한을 만족하는 첫 표본을 반환한다. max_tries 초과 시 CompositionError."""
    cfg = _resolve_spec(spec)
    if not isinstance(max_tries, int) or isinstance(max_tries, bool) or max_tries < 1:
        raise CompositionError("max_tries: must be >= 1")
    ordered = _sorted_sites(sites)
    sids = ordered["site_id"].tolist()
    n = len(ordered)
    last_error: CompositionError | None = None
    for _ in range(int(max_tries)):
        picks = rng.integers(0, 2, size=n).tolist()
        groups = [cfg.groups[int(v)] for v in picks]
        out = pd.DataFrame(
            {
                "site_id": pd.Series(sids, dtype="string"),
                "group": pd.Series(groups, dtype="string"),
            }
        )
        out = out.sort_values("site_id", kind="mergesort").reset_index(drop=True)
        try:
            validate_assignment(out, sites, spec=cfg)
        except CompositionError as exc:
            last_error = exc
            continue
        return out
    raise CompositionError(
        f"random_assignment: infeasible after {int(max_tries)} tries ({last_error})"
    )


def capacity_balanced_assignment(
    sites: pd.DataFrame, rng: np.random.Generator, *, spec: CompetitionSpec | None = None
) -> pd.DataFrame:
    """무작위 순서로 사이트를 용량이 작은 그룹에 차례로 넣은 배정을 반환한다."""
    cfg = _resolve_spec(spec)
    ordered = _sorted_sites(sites)
    sids = ordered["site_id"].tolist()
    caps = ordered["capacity_kw"].to_numpy(dtype="float64").tolist()
    order = rng.permutation(len(ordered)).tolist()
    g0, g1 = cfg.groups
    totals: dict[str, float] = {g0: 0.0, g1: 0.0}
    assign: dict[str, str] = {}
    for idx in order:
        sid = str(sids[int(idx)])
        if totals[g0] < totals[g1]:
            pick = g0
        elif totals[g1] < totals[g0]:
            pick = g1
        else:
            pick = g0
        assign[sid] = pick
        totals[pick] += float(caps[int(idx)])
    out = pd.DataFrame(
        {
            "site_id": pd.Series(sorted(assign), dtype="string"),
            "group": pd.Series([assign[s] for s in sorted(assign)], dtype="string"),
        }
    )
    try:
        validate_assignment(out, sites, spec=cfg)
    except CompositionError as exc:
        raise CompositionError(f"capacity_balanced: {exc}") from exc
    return out
