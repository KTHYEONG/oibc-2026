"""집합 합계 운량 선형 사후 보정."""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.oibc.config import CalibrationConfig, CompetitionSpec
from src.oibc.models.gbm import ModelError

logger = logging.getLogger(__name__)


class CalibrationError(ModelError):
    """사후 보정 입력 계약 위반."""


@dataclass(frozen=True)
class CloudCalibration:
    """운량 선형 보정 계수 s=max(a+b·cloud, 0)와 적합 진단."""

    a: float
    b: float
    at_boundary: bool
    n_fit_hours: int


def _as_dti(values: pd.Series, label: str) -> pd.DatetimeIndex:
    try:
        return pd.DatetimeIndex(values)
    except (TypeError, ValueError) as exc:
        raise CalibrationError(f"{label}: must be datetime") from exc


def _lookup_pos(query: pd.DatetimeIndex, target: pd.DatetimeIndex) -> np.ndarray:
    q = query.as_unit("us")
    t = target.as_unit("us")
    pos: np.ndarray = t.get_indexer(q)
    return pos


def _cloud_values(frame: pd.DataFrame, var: str) -> np.ndarray:
    try:
        arr = np.asarray(frame[var].to_numpy(dtype="float64"), dtype="float64")
    except (ValueError, TypeError) as exc:
        raise CalibrationError(f"{var}: must be numeric") from exc
    return arr


def _lookup_cloud(dti: pd.DatetimeIndex, index: pd.Series) -> np.ndarray:
    target = _as_dti(index.index.to_series(), "index")
    try:
        c_all = np.asarray(index.to_numpy(dtype="float64"), dtype="float64")
    except (ValueError, TypeError) as exc:
        raise CalibrationError("index must be numeric") from exc
    pos = _lookup_pos(dti, target)
    c = np.full(len(dti), np.nan)
    hit = pos >= 0
    c[hit] = c_all[pos[hit]]
    return c


def cloud_index(frame: pd.DataFrame, cfg: CalibrationConfig) -> pd.Series:
    """frame의 ts별 운량 지수(0~1, 모두 NaN이면 NaN)를 오름차순 ts index Series로 반환한다. 위반 시 CalibrationError."""
    missing = [c for c in ("ts", *cfg.cloud_vars) if c not in frame.columns]
    if missing:
        raise CalibrationError(f"missing columns {missing[:5]}")
    if frame.columns.has_duplicates:
        raise CalibrationError("duplicate columns")
    dti = _as_dti(frame["ts"], "ts")
    if dti.isna().any():
        raise CalibrationError("ts must not contain missing values")
    unit = dti.unit
    codes, uniques = pd.factorize(dti.as_unit("us"), sort=True)
    key = pd.Series(codes)
    cols: list[np.ndarray] = []
    for var in cfg.cloud_vars:
        x = _cloud_values(frame, var)
        finite = x[~np.isnan(x)]
        if finite.size > 0 and (bool((finite < 0).any()) or bool((finite > 100).any())):
            raise CalibrationError(f"{var}: must be within [0, 100]")
        nunique = pd.Series(x).groupby(key, sort=True).nunique(dropna=False).to_numpy()
        if bool((nunique != 1).any()):
            raise CalibrationError(f"{var}: must be unique within ts")
        first = pd.Series(x).groupby(key, sort=True).first().to_numpy(dtype="float64")
        cols.append(first)
    mat = np.column_stack(cols)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(mat, axis=1) / 100.0
    values = np.asarray(mean, dtype="float64")
    idx = pd.DatetimeIndex(uniques).as_unit(unit)
    return pd.Series(values, index=idx, dtype="float64")


def scale_factors(ts: pd.Series, index: pd.Series, calibration: CloudCalibration) -> np.ndarray:
    """ts 각 원소의 보정 계수(float64, ≥0)를 반환한다. 지수가 없거나 NaN인 ts는 1.0."""
    dti = _as_dti(ts, "ts")
    if dti.isna().any():
        raise CalibrationError("ts must not contain missing values")
    for name in ("a", "b"):
        if not np.isfinite(float(getattr(calibration, name))):
            raise CalibrationError(f"calibration.{name} must be finite")
    c = _lookup_cloud(dti, index)
    out = np.full(len(dti), 1.0, dtype="float64")
    valid = ~np.isnan(c)
    with np.errstate(over="ignore", invalid="ignore"):
        out[valid] = np.maximum(calibration.a + calibration.b * c[valid], 0.0)
    return out


def _hourly_arrays(hourly: pd.DataFrame) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    missing = [c for c in ("ts", "forecast_kwh", "actual_kwh", "smp", "k") if c not in hourly.columns]
    if missing:
        raise CalibrationError(f"missing columns {missing[:5]}")
    if len(hourly) == 0:
        raise CalibrationError("hourly must not be empty")
    dti = _as_dti(hourly["ts"], "ts")
    if dti.isna().any():
        raise CalibrationError("ts must not contain missing values")
    try:
        f = np.asarray(hourly["forecast_kwh"].to_numpy(dtype="float64"), dtype="float64")
        a = np.asarray(hourly["actual_kwh"].to_numpy(dtype="float64"), dtype="float64")
        s = np.asarray(hourly["smp"].to_numpy(dtype="float64"), dtype="float64")
        k = np.asarray(hourly["k"].to_numpy(dtype="float64"), dtype="float64")
    except (ValueError, TypeError) as exc:
        raise CalibrationError("hourly columns must be numeric") from exc
    if not np.all(np.isfinite(f)) or not np.all(np.isfinite(a)):
        raise CalibrationError("forecast and actual must be finite")
    if not np.all(np.isfinite(s)) or not np.all(np.isfinite(k)):
        raise CalibrationError("smp and k must be finite")
    if bool((f < 0).any()):
        raise CalibrationError("forecast_kwh must be >= 0")
    return dti, f, a, s, k


def fit_cloud_calibration(
    hourly: pd.DataFrame, index: pd.Series, cfg: CalibrationConfig, spec: CompetitionSpec
) -> CloudCalibration:
    """정산 hourly 프레임에서 가중 오차합을 최소화하는 격자 (a,b)를 반환한다. 위반 시 CalibrationError."""
    dti, f, a_vals, smp, k = _hourly_arrays(hourly)
    n_a = round((cfg.a_max - cfg.a_min) / cfg.step) + 1
    n_b = round((cfg.b_max - cfg.b_min) / cfg.step) + 1
    a_grid = np.array([round(cfg.a_min + i * cfg.step, 10) for i in range(n_a)], dtype="float64")
    b_grid = np.array([round(cfg.b_min + j * cfg.step, 10) for j in range(n_b)], dtype="float64")
    c = _lookup_cloud(dti, index)
    w = spec.penalty_rate * smp * k
    if not np.all(np.isfinite(w)):
        raise CalibrationError("weights must be finite")
    fixed = (f > 0) & np.isnan(c)
    best_val = np.inf
    best_i = 0
    best_j = 0
    for i in range(n_a):
        ai = float(a_grid[i])
        with np.errstate(over="ignore", invalid="ignore"):
            smat = np.maximum(ai + np.outer(b_grid, np.nan_to_num(c, nan=0.0)), 0.0)
            smat[:, fixed] = 1.0
            jvec = (np.abs(smat * f[None, :] - a_vals[None, :]) * w[None, :]).sum(axis=1)
        for j in range(n_b):
            v = float(jvec[j])
            if v < best_val:
                best_val = v
                best_i = i
                best_j = j
    at_boundary = best_i in (0, n_a - 1) or best_j in (0, n_b - 1)
    if at_boundary:
        logger.warning("[EVAL] calibration at grid boundary a=%r b=%r", float(a_grid[best_i]), float(b_grid[best_j]))
    return CloudCalibration(
        a=float(a_grid[best_i]), b=float(b_grid[best_j]), at_boundary=at_boundary, n_fit_hours=int((f > 0).sum())
    )


def apply_calibration(preds: np.ndarray, ts: pd.Series, index: pd.Series, calibration: CloudCalibration) -> np.ndarray:
    """1-D 또는 (n_seeds, n_rows) 예측에 행별 s(ts)를 곱한 새 배열을 반환한다. 모양·값 위반 시 CalibrationError."""
    try:
        arr = np.asarray(preds, dtype="float64")
    except (ValueError, TypeError) as exc:
        raise CalibrationError("preds must be numeric") from exc
    if arr.ndim == 1:
        if arr.shape[0] != len(ts):
            raise CalibrationError("preds length must match ts")
    elif arr.ndim == 2:
        if arr.shape[1] != len(ts):
            raise CalibrationError("preds columns must match ts")
    else:
        raise CalibrationError("preds must be 1-D or 2-D")
    if arr.size == 0:
        raise CalibrationError("preds must not be empty")
    if not np.all(np.isfinite(arr)):
        raise CalibrationError("preds must be finite")
    if bool((arr < 0).any()):
        raise CalibrationError("preds must be >= 0")
    s = scale_factors(ts, index, calibration)
    with np.errstate(over="ignore", invalid="ignore"):
        out = np.asarray(arr * s if arr.ndim == 1 else arr * s[None, :], dtype="float64")
    if not np.all(np.isfinite(out)) or bool((out < 0).any()):
        raise CalibrationError("calibrated predictions must be finite and >= 0")
    return out
