"""집합 정산 계산."""

# skill_score 정규화는 가정이며 리더보드 공식 미확인.
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

from src.oibc.config import CompetitionSpec
from src.oibc.data.schemas import _DT
from src.oibc.timeaxis import TimeAxisError, k_weight, op_date, require_hourly_kst

logger = logging.getLogger(__name__)

Breakdown = Literal["group", "hour", "op_date", "month", "quarter"]


class SettlementError(ValueError):
    """정산 입력 위반."""


@dataclass(frozen=True)
class SettlementResult:
    """정산 결과: 총계와 (group, ts) 단위 내역."""

    revenue: float
    penalty: float
    settlement: float
    hourly: pd.DataFrame


def _spec_or_default(spec: CompetitionSpec | None) -> CompetitionSpec:
    return spec if spec is not None else CompetitionSpec()


def _require_columns(df: pd.DataFrame, cols: list[str], label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SettlementError(f"{label}: missing column {missing[0]!r}")


def _check_hourly(ts: pd.Series, name: str) -> None:
    try:
        require_hourly_kst(ts, name)
    except TimeAxisError as e:
        raise SettlementError(str(e)) from e


def _finite_nonneg(values: pd.Series, name: str) -> np.ndarray[Any, Any]:
    arr = cast("np.ndarray[Any, Any]", pd.to_numeric(values, errors="coerce").to_numpy(dtype="float64"))
    if not np.all(np.isfinite(arr)) or bool((arr < 0).any()):
        raise SettlementError(f"{name}: must be finite and >= 0")
    return arr


def _finite(values: pd.Series, name: str) -> np.ndarray[Any, Any]:
    arr = cast("np.ndarray[Any, Any]", pd.to_numeric(values, errors="coerce").to_numpy(dtype="float64"))
    if not np.all(np.isfinite(arr)):
        raise SettlementError(f"{name}: must be finite")
    return arr


def aggregate_groups(
    site_frame: pd.DataFrame, assignment: pd.DataFrame, *, spec: CompetitionSpec | None = None
) -> pd.DataFrame:
    """발전소 예측·실적을 배정에 따라 (group, ts) 합계로 묶는다. 계약 위반 시 SettlementError."""
    cfg = _spec_or_default(spec)
    _require_columns(site_frame, ["site_id", "ts", "pred_kwh", "actual_kwh"], "site_frame")
    _require_columns(assignment, ["site_id", "group"], "assignment")
    if site_frame["site_id"].isna().any() or assignment["site_id"].isna().any():
        raise SettlementError("site_id: null value")
    if assignment.duplicated(subset=["site_id"]).any():
        raise SettlementError("assignment: duplicate site_id")
    allowed = set(cfg.groups)
    groups = assignment["group"]
    if (~groups.isin(allowed)).any():
        raise SettlementError("assignment: unknown group")
    if site_frame.duplicated(subset=["site_id", "ts"]).any():
        raise SettlementError("site_frame: duplicate key (site_id, ts)")
    _check_hourly(site_frame["ts"], "ts")
    missing = set(site_frame["site_id"].astype("string").tolist()) - set(
        assignment["site_id"].astype("string").tolist()
    )
    if missing:
        raise SettlementError("site_frame: unassigned site_id")
    _finite_nonneg(site_frame["pred_kwh"], "pred_kwh")
    _finite(site_frame["actual_kwh"], "actual_kwh")
    mapping: dict[str, str] = dict(
        zip(
            assignment["site_id"].astype("string").tolist(),
            assignment["group"].astype("string").tolist(),
            strict=True,
        )
    )
    work = pd.DataFrame(
        {
            "site_id": site_frame["site_id"].astype("string"),
            "ts": pd.Series(
                pd.DatetimeIndex(site_frame["ts"]).as_unit("us"), index=site_frame.index, dtype=_DT
            ),
            "pred_kwh": site_frame["pred_kwh"].to_numpy(dtype="float64", na_value=np.nan),
            "actual_kwh": site_frame["actual_kwh"].to_numpy(dtype="float64", na_value=np.nan),
        }
    )
    work["group"] = work["site_id"].map(mapping).astype("string")
    grouped = (
        work.groupby(["group", "ts"], observed=True, sort=False)
        .agg(forecast_kwh=("pred_kwh", "sum"), actual_kwh=("actual_kwh", "sum"), n_sites=("site_id", "size"))
        .reset_index()
    )
    out = pd.DataFrame(
        {
            "group": grouped["group"].astype("string"),
            "ts": grouped["ts"].astype(_DT),
            "forecast_kwh": grouped["forecast_kwh"].to_numpy(dtype="float64"),
            "actual_kwh": grouped["actual_kwh"].to_numpy(dtype="float64"),
            "n_sites": grouped["n_sites"].to_numpy(dtype="int32"),
        }
    )
    return out.sort_values(["group", "ts"], kind="mergesort").reset_index(drop=True)


def settle(group_frame: pd.DataFrame, smp: pd.DataFrame, *, spec: CompetitionSpec | None = None) -> SettlementResult:
    """집합 예측·실적과 SMP로 S, 매전수익, W를 계산한다. 계약 위반 시 SettlementError."""
    cfg = _spec_or_default(spec)
    _require_columns(group_frame, ["group", "ts", "forecast_kwh", "actual_kwh"], "group_frame")
    _require_columns(smp, ["ts", "smp", "k"], "smp")
    if len(group_frame) == 0:
        raise SettlementError("group_frame: empty")
    if group_frame.duplicated(subset=["group", "ts"]).any():
        raise SettlementError("group_frame: duplicate key (group, ts)")
    if (~group_frame["group"].astype("string").isin(set(cfg.groups))).any():
        raise SettlementError("group_frame: unknown group")
    _check_hourly(group_frame["ts"], "ts")
    _check_hourly(smp["ts"], "smp.ts")
    if smp.duplicated(subset=["ts"]).any():
        raise SettlementError("smp: duplicate ts")
    forecast = _finite_nonneg(group_frame["forecast_kwh"], "forecast_kwh")
    actual = _finite(group_frame["actual_kwh"], "actual_kwh")
    smp_vals = pd.to_numeric(smp["smp"], errors="coerce").to_numpy(dtype="float64")
    if not np.all(np.isfinite(smp_vals)) or bool((smp_vals < 0).any()):
        raise SettlementError("smp.smp: must be finite and >= 0")
    expected_k = k_weight(smp["ts"], cfg).to_numpy(dtype="float64")
    got_k = pd.to_numeric(smp["k"], errors="coerce").to_numpy(dtype="float64")
    if not np.array_equal(got_k, expected_k):
        raise SettlementError("smp.k: mismatch with k_weight")
    pos = pd.DatetimeIndex(smp["ts"]).as_unit("us").get_indexer(pd.DatetimeIndex(group_frame["ts"]).as_unit("us"))
    if bool((pos < 0).any()):
        raise SettlementError("group_frame: ts missing in smp")
    price = smp_vals[pos]
    kvals = got_k[pos]
    abs_err = np.abs(forecast - actual)
    penalty_arr = cfg.penalty_rate * price * kvals * abs_err
    revenue_arr = price * actual
    revenue = float(revenue_arr.sum())
    penalty = float(penalty_arr.sum())
    settlement = revenue - penalty
    opd = op_date(group_frame["ts"])
    hourly = pd.DataFrame(
        {
            "group": group_frame["group"].astype("string").reset_index(drop=True),
            "ts": pd.Series(pd.DatetimeIndex(group_frame["ts"]), dtype=_DT).reset_index(drop=True),
            "op_date": opd.astype(_DT).reset_index(drop=True),
            "smp": pd.Series(price, dtype="float64"),
            "k": pd.Series(kvals.astype("int8"), dtype="int8"),
            "forecast_kwh": pd.Series(forecast, dtype="float64"),
            "actual_kwh": pd.Series(actual, dtype="float64"),
            "abs_err": pd.Series(abs_err, dtype="float64"),
            "penalty": pd.Series(penalty_arr, dtype="float64"),
            "revenue": pd.Series(revenue_arr, dtype="float64"),
        }
    )
    hourly = hourly.sort_values(["group", "ts"], kind="mergesort").reset_index(drop=True)
    logger.info("[EVAL] settle rows=%d groups=%d revenue=%r W=%r S=%r", len(hourly), int(hourly["group"].nunique()), revenue, penalty, settlement)
    return SettlementResult(revenue=revenue, penalty=penalty, settlement=settlement, hourly=hourly)


def breakdown(result: SettlementResult, by: Breakdown) -> pd.DataFrame:
    """정산 내역을 축별로 합산해 penalty, revenue, abs_err 합계를 반환한다."""
    hourly = result.hourly
    if by == "group":
        key = hourly["group"].astype("string")
        out = pd.DataFrame({"group": key, "penalty": hourly["penalty"], "revenue": hourly["revenue"], "abs_err": hourly["abs_err"]})
        return out.groupby("group", observed=True, sort=True)[["penalty", "revenue", "abs_err"]].sum().reset_index()
    if by == "hour":
        key = pd.DatetimeIndex(hourly["ts"]).tz_convert("Asia/Seoul").hour
        out = pd.DataFrame({"hour": key, "penalty": hourly["penalty"], "revenue": hourly["revenue"], "abs_err": hourly["abs_err"]})
        return out.groupby("hour", sort=True)[["penalty", "revenue", "abs_err"]].sum().reset_index()
    if by == "op_date":
        key = pd.Series(pd.DatetimeIndex(hourly["op_date"]), dtype=_DT)
        out = pd.DataFrame({"op_date": key, "penalty": hourly["penalty"], "revenue": hourly["revenue"], "abs_err": hourly["abs_err"]})
        return out.groupby("op_date", sort=True)[["penalty", "revenue", "abs_err"]].sum().reset_index()
    if by == "month":
        opd = pd.DatetimeIndex(hourly["op_date"]).tz_convert("Asia/Seoul")
        key = pd.Series(opd.strftime("%Y-%m"), dtype="string")
        out = pd.DataFrame({"month": key, "penalty": hourly["penalty"], "revenue": hourly["revenue"], "abs_err": hourly["abs_err"]})
        return out.groupby("month", observed=True, sort=True)[["penalty", "revenue", "abs_err"]].sum().reset_index()
    if by == "quarter":
        opd = pd.DatetimeIndex(hourly["op_date"]).tz_convert("Asia/Seoul")
        key = pd.Series([f"{y}Q{q}" for y, q in zip(opd.year.tolist(), opd.quarter.tolist(), strict=True)], dtype="string")
        out = pd.DataFrame({"quarter": key, "penalty": hourly["penalty"], "revenue": hourly["revenue"], "abs_err": hourly["abs_err"]})
        return out.groupby("quarter", observed=True, sort=True)[["penalty", "revenue", "abs_err"]].sum().reset_index()
    raise SettlementError(f"by: unknown axis {by!r}")


def daily_totals(result: SettlementResult) -> pd.DataFrame:
    """운영일별 penalty·revenue 합계(index=op_date)를 반환한다."""
    hourly = result.hourly
    key = pd.Series(pd.DatetimeIndex(hourly["op_date"]), dtype=_DT)
    grouped = (
        pd.DataFrame({"op_date": key, "penalty": hourly["penalty"].to_numpy(dtype="float64"), "revenue": hourly["revenue"].to_numpy(dtype="float64")})
        .groupby("op_date", sort=True)[["penalty", "revenue"]]
        .sum()
    )
    grouped.index = pd.DatetimeIndex(grouped.index, dtype=_DT)
    grouped = grouped.sort_index()
    grouped["penalty"] = grouped["penalty"].astype("float64")
    grouped["revenue"] = grouped["revenue"].astype("float64")
    return grouped


def skill_score(penalty: float, reference_penalty: float) -> float:
    """기준 대비 점수 100*(1 - W/W_ref)를 반환한다. W_ref가 0 이하이거나 비유한 값이면 SettlementError."""
    ref = float(reference_penalty)
    if not math.isfinite(ref) or ref <= 0:
        raise SettlementError("reference_penalty: must be finite and > 0")
    return 100.0 * (1.0 - float(penalty) / ref)
