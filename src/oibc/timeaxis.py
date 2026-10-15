"""우측닫힘 시간축 연산."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.oibc.config import CompetitionSpec

_KST = "Asia/Seoul"
_OUT_UNIT = "datetime64[us, Asia/Seoul]"


class TimeAxisError(ValueError):
    """시간축 입력 위반."""


def _require_kst(ts: pd.Series, name: str) -> pd.DatetimeIndex:
    try:
        dti = pd.DatetimeIndex(ts)
    except Exception as e:
        raise TimeAxisError(f"{name}: not datetime") from e
    if dti.tz is None:
        raise TimeAxisError(f"{name}: tz-naive")
    if str(dti.tz) != _KST:
        raise TimeAxisError(f"{name}: must be Asia/Seoul")
    if dti.hasnans:
        raise TimeAxisError(f"{name}: missing timestamp")
    return dti


def require_hourly_kst(ts: pd.Series, name: str) -> None:
    """ts가 tz-aware Asia/Seoul 정각인지 검증한다."""
    dti = _require_kst(ts, name)
    if (dti != dti.floor("h")).any():
        raise TimeAxisError(f"{name}: must be hourly on the hour")


def _require_op_dates(op_dates: pd.Series) -> pd.DatetimeIndex:
    dti = _require_kst(op_dates, "op_dates")
    if (dti != dti.floor("D")).any():
        raise TimeAxisError("op_dates: must be midnight")
    return dti


def op_date(ts: pd.Series) -> pd.Series:
    """우측닫힘 시각을 운영일로 변환한다."""
    dti = _require_kst(ts, "ts")
    require_hourly_kst(ts, "ts")
    out = (dti - pd.Timedelta(hours=1)).floor("D").tz_convert(_KST)
    return pd.Series(out.as_unit("us"), index=ts.index, dtype=_OUT_UNIT)


def cutoff(op_dates: pd.Series, spec: CompetitionSpec) -> pd.Series:
    """운영일에서 예측 마감 시각을 계산한다."""
    dti = _require_op_dates(op_dates)
    out = (dti - pd.Timedelta(days=1)) + pd.to_timedelta(spec.cutoff_hour, unit="h")
    return pd.Series(out.as_unit("us"), index=op_dates.index, dtype=_OUT_UNIT)


def issued_at(
    op_dates: pd.Series,
    rounds: pd.Series,
    spec: CompetitionSpec | None = None,
) -> pd.Series:
    """운영일과 라운드에서 예보 발행 시각을 계산한다."""
    cfg = spec or CompetitionSpec()
    dti = _require_op_dates(op_dates)
    if not rounds.index.equals(op_dates.index):
        raise TimeAxisError("rounds: index must match op_dates")
    r = rounds
    allowed = set(cfg.rounds)
    if (~r.isin(allowed)).any():
        raise TimeAxisError(f"rounds: must be in {sorted(allowed)}")
    out = (dti - pd.Timedelta(days=1)) + pd.to_timedelta(r.to_numpy(), unit="h")
    return pd.Series(out.as_unit("us"), index=op_dates.index, dtype=_OUT_UNIT)


def k_weight(ts: pd.Series, spec: CompetitionSpec) -> pd.Series:
    """시각을 정산 가중치로 변환한다."""
    dti = _require_kst(ts, "ts")
    require_hourly_kst(ts, "ts")
    peak = set(spec.k_peak_hours)
    flag = dti.hour.isin(peak)
    vals = np.where(flag, spec.k_peak, spec.k_base)
    return pd.Series(vals, index=ts.index, dtype="int8")
