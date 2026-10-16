"""발행 내 예보 변환."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.oibc.config import CompetitionSpec, FeatureConfig
from src.oibc.data import gate
from src.oibc.timeaxis import op_date, require_hourly_kst

# Meinel 청천 모델 상수.
_MEINEL_SCALE = 1.1
_MEINEL_SOLAR = 1.353
_MEINEL_TAU = 0.7
_MEINEL_TAU_EXP = 0.678
_MEINEL_E0_AMP = 0.033


class FeatureError(ValueError):
    """특성 계약 위반."""


def _check_forecast_keys(forecast: pd.DataFrame, station_col: str, spec: CompetitionSpec) -> list[int]:
    if station_col not in forecast.columns:
        raise FeatureError(f"missing station column {station_col!r}")
    for col in ("round", "ts", "op_date", "issued_at", "fc_source"):
        if col not in forecast.columns:
            raise FeatureError(f"missing column {col!r}")
    if forecast[[station_col, "round", "ts"]].isna().any().any():
        raise FeatureError("forecast: missing key")
    if forecast.duplicated([station_col, "round", "ts"]).any():
        raise FeatureError("forecast: duplicate key")
    rounds = sorted(spec.rounds)
    if not forecast["round"].isin(rounds).all():
        raise FeatureError("forecast: invalid round")
    require_hourly_kst(forecast["ts"], "forecast.ts")
    if not forecast["op_date"].eq(op_date(forecast["ts"])).all():
        raise FeatureError("forecast: inconsistent op_date")
    if (forecast["issued_at"].isna() & forecast["fc_source"].ne("missing")).any():
        raise FeatureError("forecast: missing issuance for available forecast")
    gate.assert_causal(forecast["ts"], forecast["issued_at"], spec)
    return rounds


def station_round_wide(
    forecast: pd.DataFrame, station_col: str, variables: Sequence[str], spec: CompetitionSpec
) -> pd.DataFrame:
    """(station, ts) 키로 라운드별 변수를 `{var}_r{round}` 열로 펼친다."""
    rounds = _check_forecast_keys(forecast, station_col, spec)
    for name in variables:
        if name not in forecast.columns:
            raise FeatureError(f"missing variable {name!r}")
    counts = forecast.groupby([station_col, "ts"], sort=False).size()
    if (counts != len(rounds)).any():
        raise FeatureError("forecast: missing round for some (station, ts)")
    base = forecast[[station_col, "ts"]].drop_duplicates().sort_values([station_col, "ts"]).reset_index(drop=True)
    out = base.copy()
    for var in variables:
        pivot = forecast.pivot(index=[station_col, "ts"], columns="round", values=var)
        for r in rounds:
            col = f"{var}_r{r}"
            vals = pivot[r].reindex(pd.MultiIndex.from_frame(base)).to_numpy(dtype="float64")
            out[col] = vals.astype("float32")
    return out


def neighbor_features(
    forecast: pd.DataFrame, station_col: str, variables: Sequence[str], offsets: Sequence[int], spec: CompetitionSpec
) -> pd.DataFrame:
    """(station, ts) 키로 `{var}_r{round}_n{off:+d}` 이웃 시각 예보를 반환한다."""
    _check_forecast_keys(forecast, station_col, spec)
    for name in variables:
        if name not in forecast.columns:
            raise FeatureError(f"missing variable {name!r}")
    rounds = sorted(spec.rounds)
    gated = gate.forecast_neighbors(forecast, station_col, list(variables), list(offsets), spec)
    base = forecast[[station_col, "ts"]].drop_duplicates().sort_values([station_col, "ts"]).reset_index(drop=True)
    base_op = op_date(base["ts"])
    out = base.copy()
    for off in offsets:
        donor_op = op_date(base["ts"] + pd.Timedelta(hours=off))
        cross = donor_op.to_numpy() != base_op.to_numpy()
        for r in rounds:
            sub = gated[gated["round"] == r].set_index([station_col, "ts"])
            for var in variables:
                col = f"{var}_r{r}_n{off:+d}"
                src = f"{var}_lag{off:+d}"
                if sub.empty or src not in sub.columns:
                    out[col] = np.nan
                    out[col] = out[col].astype("float32")
                    continue
                vals = np.array(sub[src].reindex(pd.MultiIndex.from_frame(base)), dtype="float64", copy=True)
                vals[cross] = np.nan
                out[col] = vals.astype("float32")
    return out


def round_revision_features(wide: pd.DataFrame, variables: Sequence[str], spec: CompetitionSpec) -> pd.DataFrame:
    """라운드 평균 `{var}_rmean`, 수정폭 `{var}_rdiff`(마지막 라운드 - 첫 라운드)를 반환한다."""
    rounds = sorted(spec.rounds)
    first, last = rounds[0], rounds[-1]
    for var in variables:
        for r in rounds:
            if f"{var}_r{r}" not in wide.columns:
                raise FeatureError(f"missing wide column {var}_r{r}")
    keys = [c for c in wide.columns if not (any(c == f"{v}_r{r}" for v in variables for r in rounds))]
    if not keys:
        raise FeatureError("wide: no key columns")
    out = wide[keys].copy()
    for var in variables:
        cols = [f"{var}_r{r}" for r in rounds]
        mat = wide[cols].to_numpy(dtype="float64")
        with np.errstate(all="ignore"):
            mean = np.mean(mat, axis=1)
            diff = wide[f"{var}_r{last}"].to_numpy(dtype="float64") - wide[f"{var}_r{first}"].to_numpy(dtype="float64")
        out[f"{var}_rmean"] = mean.astype("float32")
        out[f"{var}_rdiff"] = diff.astype("float32")
    return out


def day_features(
    forecast: pd.DataFrame, station_col: str, cfg: FeatureConfig, spec: CompetitionSpec
) -> pd.DataFrame:
    """`day_round` 발행분에서 (station, ts) 키로 일중 파생 열을 반환한다."""
    _check_forecast_keys(forecast, station_col, spec)
    need = (*cfg.day_mean_vars, *cfg.day_max_vars, *cfg.day_sum_vars)
    for name in need:
        if name not in forecast.columns:
            raise FeatureError(f"missing variable {name!r}")
    filt = forecast[forecast["round"] == cfg.day_round].copy()
    if filt.empty:
        raise FeatureError("forecast: no rows for day_round")
    filt = filt.assign(__op=pd.DatetimeIndex(filt["op_date"]))
    filt = filt.sort_values([station_col, "ts"]).reset_index(drop=True)
    base = filt[[station_col, "ts"]].copy()
    bundles = filt[[station_col, "__op"]].drop_duplicates()
    hours_grid = pd.DataFrame({"__hour": np.arange(1, 25)})
    full = bundles.merge(hours_grid, how="cross")
    full["ts"] = full["__op"] + pd.to_timedelta(full.pop("__hour"), unit="h")
    filt = full.merge(filt.drop(columns="__op"), on=[station_col, "ts"], how="left", validate="one_to_one")
    filt = filt.sort_values([station_col, "ts"]).reset_index(drop=True)
    grouped = filt.groupby([station_col, "__op"], sort=False)
    out = filt[[station_col, "ts"]].copy()
    shift_vars = tuple(dict.fromkeys((*cfg.day_mean_vars, *cfg.day_max_vars)))
    for name in shift_vars:
        for off in cfg.day_shift_offsets:
            out[f"{name}_d{off:+d}"] = grouped[name].shift(-off).to_numpy(dtype="float64").astype("float32")
        rolled = grouped[name].transform(lambda s: s.rolling(cfg.day_roll_window, center=True, min_periods=1).mean())
        out[f"{name}_roll{cfg.day_roll_window}"] = rolled.to_numpy(dtype="float64").astype("float32")
    hours = pd.DatetimeIndex(filt["ts"]).hour.to_numpy()
    lo, hi = cfg.day_hours
    daytime = (hours >= lo) & (hours <= hi)
    for name in cfg.day_mean_vars:
        masked = filt[name].where(daytime)
        means = masked.groupby([filt[station_col], filt["__op"]]).transform("mean")
        out[f"{name}_daymean"] = means.to_numpy(dtype="float64").astype("float32")
    for name in cfg.day_max_vars:
        out[f"{name}_daymax"] = grouped[name].transform("max").to_numpy(dtype="float64").astype("float32")
    for name in cfg.day_sum_vars:
        out[f"{name}_daysum"] = grouped[name].transform("sum", min_count=1).to_numpy(dtype="float64").astype("float32")
    return base.merge(out, on=[station_col, "ts"], how="left", validate="one_to_one")


def fleet_features(forecast: pd.DataFrame, variables: Sequence[str], round_: int) -> pd.DataFrame:
    """ts 키로 같은 라운드의 전 지점 평균 `{var}_fleet`을 반환한다."""
    if "ts" not in forecast.columns or "round" not in forecast.columns:
        raise FeatureError("forecast: missing key")
    stations = [c for c in forecast.columns if c.endswith("_station")]
    if len(stations) != 1:
        raise FeatureError("forecast: exactly one station column required")
    _check_forecast_keys(forecast, stations[0], CompetitionSpec())
    for name in variables:
        if name not in forecast.columns:
            raise FeatureError(f"missing variable {name!r}")
    filt = forecast[forecast["round"] == round_].copy()
    if filt.empty:
        raise FeatureError("forecast: no rows for round")
    gate.assert_causal(filt["ts"], filt["issued_at"], CompetitionSpec())
    grouped = filt.groupby("ts", sort=True)
    out = pd.DataFrame({"ts": pd.DatetimeIndex(sorted(filt["ts"].unique()))})
    for name in variables:
        means = grouped[name].mean()
        vals = means.reindex(pd.DatetimeIndex(out["ts"])).to_numpy(dtype="float64")
        out[f"{name}_fleet"] = vals.astype("float32")
    out["ts"] = pd.DatetimeIndex(out["ts"]).as_unit("us")
    return out


def clear_sky_ghi(cosz: NDArray[np.generic], doy: NDArray[np.generic]) -> NDArray[np.float32]:
    """Meinel 청천 수평면 일사량(kW/m², cosz≤0이면 0)을 반환한다."""
    cz = np.asarray(cosz, dtype="float64")
    dd = np.asarray(doy, dtype="float64")
    am = 1.0 / np.maximum(cz, 0.02)
    e0 = 1.0 + _MEINEL_E0_AMP * np.cos(2.0 * np.pi * dd / 365.0)
    ghi = _MEINEL_SCALE * e0 * _MEINEL_SOLAR * np.power(_MEINEL_TAU, np.power(am, _MEINEL_TAU_EXP)) * cz
    result: NDArray[np.float32] = np.maximum(ghi, 0.0).astype("float32")
    return result
