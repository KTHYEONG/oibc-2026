"""지리 층 반클러스터링 배정."""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from src.oibc.compose.constraints import CompositionError, validate_assignment
from src.oibc.config import CompetitionSpec, ComposeConfig

logger = logging.getLogger(__name__)


def _resolve_spec(spec: CompetitionSpec | None) -> CompetitionSpec:
    return spec if spec is not None else CompetitionSpec()


def _site_frame(sites: pd.DataFrame, *, need_geo: bool) -> pd.DataFrame:
    need = ["site_id", "capacity_kw"] + (["lat", "lon"] if need_geo else [])
    if not sites.columns.is_unique:
        raise CompositionError("sites: duplicate columns")
    missing = [c for c in need if c not in sites.columns]
    if missing:
        raise CompositionError(f"sites: missing column {missing[0]!r}")
    if sites["site_id"].isna().any():
        raise CompositionError("sites: null site_id")
    if sites["site_id"].astype("string").duplicated().any():
        raise CompositionError("sites: duplicate site_id")
    cap = pd.to_numeric(sites["capacity_kw"], errors="coerce").to_numpy(dtype="float64")
    if not np.all(np.isfinite(cap)) or bool((cap <= 0).any()):
        raise CompositionError("sites: capacity_kw must be finite and > 0")
    try:
        math.fsum(cap.tolist())
    except OverflowError as e:
        raise CompositionError("sites: non-finite total capacity") from e
    out = pd.DataFrame(
        {
            "site_id": sites["site_id"].astype("string"),
            "capacity_kw": cap,
        },
        index=sites.index,
    )
    if need_geo:
        lat = pd.to_numeric(sites["lat"], errors="coerce").to_numpy(dtype="float64")
        lon = pd.to_numeric(sites["lon"], errors="coerce").to_numpy(dtype="float64")
        if not np.all(np.isfinite(lat)) or not np.all(np.isfinite(lon)):
            raise CompositionError("sites: lat/lon must be finite")
        out["lat"] = lat
        out["lon"] = lon
    return out


def spatial_strata(sites: pd.DataFrame, cfg: ComposeConfig) -> np.ndarray:
    """sites 행 순서와 같은 길이의 층 라벨(int64, 0..n_strata-1)을 반환한다."""
    work = _site_frame(sites, need_geo=True)
    if len(work) < cfg.n_strata:
        raise CompositionError(f"sites: count={len(work)} < n_strata={cfg.n_strata}")
    order = np.argsort(work["site_id"].to_numpy(dtype="object"))
    lat_s = work["lat"].to_numpy(dtype="float64")[order]
    lon_s = work["lon"].to_numpy(dtype="float64")[order]
    cap_s = work["capacity_kw"].to_numpy(dtype="float64")[order]
    factor = cfg.km_per_deg * math.cos(math.radians(cfg.ref_lat_deg))
    with np.errstate(over="ignore", invalid="ignore"):
        coords = np.column_stack((lon_s * factor, lat_s * cfg.km_per_deg))
    if not np.isfinite(coords).all():
        raise CompositionError("sites: non-finite projected coordinates")
    km = KMeans(n_clusters=cfg.n_strata, init="k-means++", n_init=cfg.n_init, random_state=cfg.seed)
    raw = np.asarray(km.fit_predict(coords, sample_weight=cap_s))
    if len(set(raw.tolist())) != cfg.n_strata:
        raise CompositionError("strata: empty stratum")
    ids_s = work["site_id"].to_numpy(dtype="object")[order].tolist()
    min_of: dict[int, str] = {}
    for label, sid in zip(raw.tolist(), ids_s, strict=True):
        prev = min_of.get(int(label))
        if prev is None or str(sid) < prev:
            min_of[int(label)] = str(sid)
    remap = {old: new for new, old in enumerate(sorted(min_of, key=lambda v: min_of[v]))}
    canon_s = np.array([remap[int(v)] for v in raw.tolist()], dtype="int64")
    labels = np.empty(len(work), dtype="int64")
    labels[order] = canon_s
    return labels


def balanced_split(sites: pd.DataFrame, strata: np.ndarray, *, spec: CompetitionSpec | None = None) -> pd.DataFrame:
    """층마다 용량을 두 그룹에 반씩 탐욕 배분하고 용량 하한을 보정한 (site_id, group) 배정을 반환한다."""
    cfg = _resolve_spec(spec)
    work = _site_frame(sites, need_geo=False)
    arr = np.asarray(strata)
    if arr.shape != (len(work),):
        raise CompositionError(f"strata: length={arr.shape} expected=({len(work)},)")
    if not np.issubdtype(arr.dtype, np.integer):
        raise CompositionError("strata: must be integer labels")
    g0, g1 = cfg.groups
    sids = work["site_id"].tolist()
    caps = work["capacity_kw"].to_numpy(dtype="float64")
    cap_of: dict[str, float] = {str(s): float(c) for s, c in zip(sids, caps.tolist(), strict=True)}
    labels = [int(v) for v in arr.tolist()]

    strata_ids = sorted(set(labels))
    strat_total: dict[int, float] = {}
    strat_min: dict[int, str] = {}
    for s in strata_ids:
        idx = [i for i, v in enumerate(labels) if v == s]
        strat_total[s] = math.fsum(float(caps[i]) for i in idx)
        strat_min[s] = min(str(sids[i]) for i in idx)
    ordered_strata = sorted(strata_ids, key=lambda s: (-strat_total[s], strat_min[s]))

    assign: dict[str, str] = {}
    total: dict[str, float] = {g0: 0.0, g1: 0.0}
    for s in ordered_strata:
        idx = [i for i, v in enumerate(labels) if v == s]
        idx.sort(key=lambda i: (-float(caps[i]), str(sids[i])))
        local: dict[str, float] = {g0: 0.0, g1: 0.0}
        for i in idx:
            if local[g0] < local[g1]:
                pick = g0
            elif local[g1] < local[g0]:
                pick = g1
            elif total[g0] < total[g1]:
                pick = g0
            elif total[g1] < total[g0]:
                pick = g1
            else:
                pick = g0
            assign[str(sids[i])] = pick
            local[pick] += float(caps[i])
            total[pick] += float(caps[i])

    moves = 0
    floor = cfg.min_group_capacity_kw
    for _ in range(len(work)):
        small = g0 if total[g0] < total[g1] else g1
        if total[small] >= floor:
            break
        donor = g1 if small == g0 else g0
        deficit = floor - total[small]
        donor_sids = sorted(assign, key=lambda s: str(s))
        best: str | None = None
        best_key: tuple[float, str] | None = None
        for sid in donor_sids:
            if assign[sid] != donor:
                continue
            cap = cap_of[sid]
            if total[donor] - cap < floor:
                continue
            key = (abs(cap - deficit), sid)
            if best_key is None or key < best_key:
                best_key = key
                best = sid
        if best is None:
            raise CompositionError("split: capacity floor infeasible")
        assign[best] = small
        moved = cap_of[best]
        total[small] += moved
        total[donor] -= moved
        moves += 1
    if any(v < floor for v in total.values()):
        raise CompositionError("split: capacity floor infeasible")
    logger.info("[VPP] split repair_moves=%d caps=%s", moves, total)
    rows = sorted(((sid, grp) for sid, grp in assign.items()), key=lambda r: r[0])
    return pd.DataFrame({"site_id": pd.Series([r[0] for r in rows], dtype="string"), "group": pd.Series([r[1] for r in rows], dtype="string")})


def build_composition(sites: pd.DataFrame, cfg: ComposeConfig, *, spec: CompetitionSpec | None = None) -> pd.DataFrame:
    """spatial_strata → balanced_split → validate_assignment를 거친 site_id 오름차순 배정을 반환한다."""
    cfg_spec = _resolve_spec(spec)
    labels = spatial_strata(sites, cfg)
    assignment = balanced_split(sites, labels, spec=cfg_spec)
    caps = validate_assignment(assignment, sites, spec=cfg_spec)
    counts = assignment["group"].astype("string").value_counts()
    logger.info(
        "[VPP] composition built n_strata=%d caps=%s n_per_group=%s",
        cfg.n_strata,
        caps,
        {g: int(counts.get(g, 0)) for g in cfg_spec.groups},
    )
    return assignment
