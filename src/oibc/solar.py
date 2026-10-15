"""태양 기하 계산."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.oibc.timeaxis import require_hourly_kst


def solar_position(
    ts: pd.Series, lat: NDArray[np.floating], lon: NDArray[np.floating]
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """구간 중앙 기준 태양 고도·방위·cos천정각을 반환한다."""
    n = len(ts)
    lat_a = np.asarray(lat)
    lon_a = np.asarray(lon)
    if lat_a.shape != (n,) or lon_a.shape != (n,):
        raise ValueError("length mismatch")
    if not (np.isfinite(lat_a).all() and np.isfinite(lon_a).all()):
        raise ValueError("coordinates must be finite")
    if ((np.abs(lat_a) > 90).any() or (np.abs(lon_a) > 180).any()):
        raise ValueError("coordinates outside geographic bounds")
    require_hourly_kst(ts, "ts")
    dti = pd.DatetimeIndex(ts)
    mid = dti - pd.Timedelta(minutes=30)
    doy = mid.dayofyear.to_numpy(dtype="float64")
    hour = (
        mid.hour.to_numpy(dtype="float64") + mid.minute.to_numpy(dtype="float64") / 60.0
    )
    g = 2.0 * np.pi / 365.0 * (doy - 1.0 + (hour - 12.0) / 24.0)
    eqtime = 229.18 * (
        0.000075
        + 0.001868 * np.cos(g)
        - 0.032077 * np.sin(g)
        - 0.014615 * np.cos(2.0 * g)
        - 0.040849 * np.sin(2.0 * g)
    )
    decl = (
        0.006918
        - 0.399912 * np.cos(g)
        + 0.070257 * np.sin(g)
        - 0.006758 * np.cos(2.0 * g)
        + 0.000907 * np.sin(2.0 * g)
        - 0.002697 * np.cos(3.0 * g)
        + 0.00148 * np.sin(3.0 * g)
    )
    tst = hour * 60.0 + eqtime + 4.0 * (lon_a.astype("float64") - 135.0)
    ha = np.radians(tst / 4.0 - 180.0)
    la = np.radians(lat_a.astype("float64"))
    cosz = np.sin(la) * np.sin(decl) + np.cos(la) * np.cos(decl) * np.cos(ha)
    cosz = np.clip(cosz, -1.0, 1.0)
    elev = 90.0 - np.degrees(np.arccos(cosz))
    az = np.degrees(np.arctan2(-np.sin(ha), np.tan(decl) * np.cos(la) - np.sin(la) * np.cos(ha))) % 360.0
    return (
        elev.astype("float32"),
        az.astype("float32"),
        cosz.astype("float32"),
    )
