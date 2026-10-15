"""마감 시각 기준 인과 게이트."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from src.oibc.config import CompetitionSpec
from src.oibc.timeaxis import cutoff, op_date, require_hourly_kst


class LookAheadError(RuntimeError):
    """마감 이후 정보 사용."""


def labels(table: pd.DataFrame, *, until: pd.Timestamp) -> pd.DataFrame:
    """ts가 until 이하인 실적 행만 반환한다."""
    if not isinstance(until, pd.Timestamp) or until.tzinfo is None:
        raise TypeError("until: tz-aware Timestamp required")
    require_hourly_kst(table["ts"], "ts")
    return table.loc[table["ts"] <= until].copy()


def assert_causal(
    target_ts: pd.Series,
    source_issued_at: pd.Series,
    spec: CompetitionSpec,
) -> None:
    """발행 시각이 대상 운영일 마감 이전인지 검증한다."""
    if len(target_ts) != len(source_issued_at):
        raise ValueError("length mismatch")
    target_ts = target_ts.reset_index(drop=True)
    source_issued_at = source_issued_at.reset_index(drop=True)
    limit = cutoff(op_date(target_ts), spec)
    known = source_issued_at.notna()
    if bool(known.any()) and str(pd.DatetimeIndex(source_issued_at[known]).tz) != spec.timezone:
        raise TypeError("source_issued_at: tz-aware Asia/Seoul required")
    bad = pd.Series(False, index=target_ts.index)
    bad.loc[known] = source_issued_at[known] > limit[known]
    if bool(bad.any()):
        first = target_ts[bad].iloc[0]
        raise LookAheadError(f"causal violation: {int(bad.sum())} rows, first ts={first}")


def _validate_forecast(forecast: pd.DataFrame, station_col: str, spec: CompetitionSpec) -> None:
    keys = [station_col, "round", "ts"]
    if forecast[keys].isna().any().any() or forecast.duplicated(keys).any():
        raise ValueError("forecast: missing or duplicate key")
    if not forecast["round"].isin(spec.rounds).all():
        raise ValueError("forecast: invalid round")
    require_hourly_kst(forecast["ts"], "ts")
    known = forecast["issued_at"].notna()
    if bool(known.any()):
        require_hourly_kst(forecast.loc[known, "issued_at"], "issued_at")


def forecast_neighbors(
    forecast: pd.DataFrame,
    station_col: str,
    columns: Sequence[str],
    offsets: Sequence[int],
    spec: CompetitionSpec,
) -> pd.DataFrame:
    """이웃 시각 예보를 마감 게이트를 거쳐 붙인다."""
    _validate_forecast(forecast, station_col, spec)
    out = forecast.copy()
    base_ts = pd.DatetimeIndex(forecast["ts"])
    limit = cutoff(op_date(forecast["ts"]), spec).reset_index(drop=True)
    right_base = pd.DataFrame(
        {
            "__st": forecast[station_col].to_numpy(),
            "__rd": forecast["round"].to_numpy(),
            "__nts": base_ts,
            "__iss": pd.DatetimeIndex(forecast["issued_at"]),
        }
    )
    for name in columns:
        right_base[f"__v_{name}"] = forecast[name].to_numpy()
    for offset in offsets:
        need = pd.DataFrame(
            {
                "__st": forecast[station_col].to_numpy(),
                "__rd": forecast["round"].to_numpy(),
                "__nts": base_ts + pd.Timedelta(hours=offset),
            }
        )
        merged = need.merge(right_base, on=["__st", "__rd", "__nts"], how="left", validate="one_to_one")
        blocked = (merged["__iss"].isna() | (merged["__iss"] > limit)).to_numpy()
        used = np.zeros(len(merged), dtype="bool")
        for name in columns:
            vals = merged[f"__v_{name}"].to_numpy(dtype="float64", na_value=np.nan, copy=True)
            vals[blocked] = np.nan
            col = f"{name}_lag{offset:+d}"
            out[col] = pd.Series(vals, index=out.index)
            used |= ~np.isnan(vals)
        if bool(used.any()):
            assert_causal(
                forecast["ts"].iloc[np.flatnonzero(used)].reset_index(drop=True),
                merged["__iss"][used].reset_index(drop=True),
                spec,
            )
    return out
