"""발전·실측 품질 플래그."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.oibc.config import QualityConfig
from src.oibc.solar import solar_position


def generation_flags(
    gen: pd.DataFrame, sites: pd.DataFrame, qcfg: QualityConfig
) -> pd.DataFrame:
    """generation에 품질 플래그 4종을 추가해 반환한다."""
    out = gen.copy()
    energy = out["energy"].to_numpy(dtype="float64")
    lookup = sites.set_index("site_id")
    missing = set(out["site_id"].unique()) - set(lookup.index.unique())
    if missing:
        raise ValueError(f"unknown site_id: {sorted(missing)[:5]}")
    matched = lookup.reindex(out["site_id"].to_numpy())
    cap = matched["capacity_kw"].to_numpy(dtype="float64")
    lat = matched["lat"].to_numpy(dtype="float64")
    lon = matched["lon"].to_numpy(dtype="float64")
    elev, _, _ = solar_position(out["ts"], lat, lon)
    out["dq_negative"] = pd.Series(energy < 0, index=out.index, dtype="bool")
    out["dq_over_cap"] = pd.Series(
        energy > cap * qcfg.over_cap_ratio, index=out.index, dtype="bool"
    )
    out["dq_night_gen"] = pd.Series(
        (elev < qcfg.night_elev_deg) & (energy > cap * qcfg.night_gen_ratio),
        index=out.index,
        dtype="bool",
    )
    day = elev > qcfg.zero_day_min_elev_deg
    viol = day & ~(energy <= cap * qcfg.zero_day_ratio)
    work = pd.DataFrame(
        {"site_id": out["site_id"].to_numpy(), "op_date": out["op_date"], "day": day, "viol": viol},
        index=out.index,
    )
    grouped = work.groupby(["site_id", "op_date"], sort=False, observed=True)
    n_day = grouped["day"].transform("sum")
    has_viol = grouped["viol"].transform("max")
    out["dq_zero_day"] = pd.Series(
        (np.asarray(n_day) > 0) & (np.asarray(has_viol) == 0),
        index=out.index,
        dtype="bool",
    )
    return out


def observed_sentinel(obs: pd.DataFrame, qcfg: QualityConfig) -> pd.Series:
    """관측 범위를 벗어난 행을 표시한다."""
    flag = pd.Series(np.zeros(len(obs), dtype="bool"), index=obs.index)
    for col, (low, high) in qcfg.observed_bounds.items():
        if col not in obs.columns:
            continue
        vals = obs[col]
        hit = pd.Series(np.zeros(len(obs), dtype="bool"), index=obs.index)
        if low is not None:
            hit |= vals.lt(low).fillna(False)
        if high is not None:
            hit |= vals.gt(high).fillna(False)
        flag |= hit.fillna(False).astype("bool")
    return flag.astype("bool")
