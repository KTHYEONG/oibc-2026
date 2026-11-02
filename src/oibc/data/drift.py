"""원천 예보 단절 탐지와 인과적 제출 이어붙이기. 위반 시 DriftError로 중단한다(fail-closed)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd

from src.oibc.config import DriftConfig, FeatureConfig
from src.oibc.timeaxis import TimeAxisError, op_date, require_op_dates

_KST = "Asia/Seoul"


class DriftError(ValueError):
    """단절 탐지·이어붙이기 입력 계약 위반."""


def used_raw_variables(features: FeatureConfig) -> dict[str, tuple[str, ...]]:
    """특성 설정이 소비하는 원천 예보 변수를 제공사별로 설정 순서대로 반환한다."""
    pools = (*features.wa_vars, *features.wb_vars)
    return {
        "wa": tuple(v for v in pools if v.startswith("wa_")),
        "wb": tuple(v for v in pools if v.startswith("wb_")),
    }


def detect_breaks(
    forecast: pd.DataFrame,
    provider: Literal["wa", "wb"],
    variables: Sequence[str],
    train_end: pd.Timestamp,
    cfg: DriftConfig,
) -> dict[str, pd.Timestamp | None]:
    """변수별 첫 단절 운영일(KST 자정)을 반환하고, 단절이 없으면 None. 열·키 위반 시 DriftError."""
    station = f"{provider}_station"
    required = [station, "round", "ts", *variables]
    missing = [c for c in required if c not in forecast.columns]
    if missing:
        raise DriftError(f"detect_breaks: missing columns {missing}")
    if not isinstance(train_end, pd.Timestamp) or train_end.tzinfo is None or str(train_end.tz) != _KST:
        raise DriftError("detect_breaks: train_end must be tz-aware Asia/Seoul")
    if forecast.duplicated(subset=[station, "round", "ts"]).any():
        raise DriftError("detect_breaks: duplicate (station, round, ts)")
    frame = forecast[[station, "round", "ts", *variables]].reset_index(drop=True)
    try:
        frame["_day"] = op_date(frame["ts"])
    except TimeAxisError as exc:
        raise DriftError(f"detect_breaks: {exc}") from exc
    base = frame[frame["ts"] <= train_end]
    test = frame[frame["ts"] > train_end]
    if base.empty:
        raise DriftError("detect_breaks: empty base period")
    out: dict[str, pd.Timestamp | None] = {}
    for v in variables:
        base_vals = base[v].to_numpy(dtype="float64")
        finite = base_vals[~np.isnan(base_vals)]
        if finite.size == 0:
            raise DriftError(f"detect_breaks: base has no values for {v}")
        lo, hi = float(finite.min()), float(finite.max())
        span = hi - lo
        if not span > 0.0:
            raise DriftError(f"detect_breaks: zero base span for {v}")
        margin = cfg.range_margin_frac * span
        fire_days = _range_fire_days(test, v, lo - margin, hi + margin, cfg.range_share)
        xsd_ref = _base_xsd_ref(base, station, v)
        fire_days |= _dispersion_fire_days(test, station, v, xsd_ref, cfg.dispersion_factor)
        out[v] = min(fire_days) if fire_days else None
    return out


def _range_fire_days(
    test: pd.DataFrame, v: str, low: float, high: float, share_threshold: float
) -> set[pd.Timestamp]:
    x = test[v].to_numpy(dtype="float64")
    valid = ~np.isnan(x)
    outside = valid & ((x < low) | (x > high))
    agg = pd.DataFrame({"day": test["_day"].to_numpy(), "valid": valid, "outside": outside}).groupby("day").sum()
    fired = agg[(agg["valid"] > 0) & (agg["outside"] > share_threshold * agg["valid"])]
    return set(pd.DatetimeIndex(fired.index))


def _base_xsd_ref(base: pd.DataFrame, station: str, v: str) -> float:
    """기준 구간 운영일별 관측소 간 SD의 최대값. 판정할 수 없으면 NaN."""
    sd = _station_day_sd(base, station, v)
    return float(sd.max()) if sd.notna().any() else float("nan")


def _station_day_sd(frame: pd.DataFrame, station: str, v: str) -> pd.Series:
    means = frame.groupby([station, "_day"])[v].mean().rename("m").reset_index()
    return means.groupby("_day")["m"].std(ddof=1)


def _dispersion_fire_days(
    test: pd.DataFrame, station: str, v: str, xsd_ref: float, factor: float
) -> set[pd.Timestamp]:
    if not np.isfinite(xsd_ref) or xsd_ref <= 0.0:
        return set()
    sd = _station_day_sd(test, station, v)
    fired = sd[sd > factor * xsd_ref]
    return set(pd.DatetimeIndex(fired.index))


def splice_forecasts(base: pd.DataFrame, fallback: pd.DataFrame, switch_op_date: pd.Timestamp) -> pd.DataFrame:
    """운영일이 switch 이전이면 base, 이후(포함)면 fallback 값을 쓴 (group, ts, forecast_kwh) 표를 반환한다."""
    for name, table in (("base", base), ("fallback", fallback)):
        missing = [c for c in ("group", "ts", "forecast_kwh") if c not in table.columns]
        if missing:
            raise DriftError(f"splice_forecasts: {name} missing columns {missing}")
        if table.duplicated(subset=["group", "ts"]).any():
            raise DriftError(f"splice_forecasts: {name} duplicate (group, ts)")
    if (
        not isinstance(switch_op_date, pd.Timestamp)
        or switch_op_date.tzinfo is None
        or str(switch_op_date.tz) != _KST
    ):
        raise DriftError("splice_forecasts: switch_op_date must be tz-aware Asia/Seoul")
    try:
        require_op_dates(pd.Series([switch_op_date]))
    except TimeAxisError as exc:
        raise DriftError(f"splice_forecasts: {exc}") from exc
    base_keys = pd.MultiIndex.from_frame(base[["group", "ts"]])
    fallback_keys = pd.MultiIndex.from_frame(fallback[["group", "ts"]])
    if len(base_keys.symmetric_difference(fallback_keys)) > 0:
        raise DriftError("splice_forecasts: (group, ts) keys differ")
    merged = base[["group", "ts", "forecast_kwh"]].merge(
        fallback[["group", "ts", "forecast_kwh"]],
        on=["group", "ts"],
        how="inner",
        suffixes=("_base", "_fallback"),
    ).reset_index(drop=True)
    try:
        use_fallback = (op_date(merged["ts"]) >= switch_op_date).to_numpy()
    except TimeAxisError as exc:
        raise DriftError(f"splice_forecasts: {exc}") from exc
    values = np.where(
        use_fallback,
        merged["forecast_kwh_fallback"].to_numpy(dtype="float64"),
        merged["forecast_kwh_base"].to_numpy(dtype="float64"),
    )
    if not np.isfinite(values).all() or (values < 0.0).any():
        raise DriftError("splice_forecasts: spliced forecast must be finite and non-negative")
    out = pd.DataFrame({"group": merged["group"], "ts": merged["ts"], "forecast_kwh": values})
    return out.sort_values(["group", "ts"], kind="mergesort").reset_index(drop=True)
