import numpy as np
import pandas as pd
import pytest

from src.oibc.solar import solar_position

KST = "Asia/Seoul"


def _ts(vals: list[str]) -> pd.Series:
    return pd.Series(pd.to_datetime(vals).tz_localize(KST))


def test_summer_noon_high() -> None:
    ts = _ts(["2024-06-21 13:00"])
    elev, _, _ = solar_position(ts, np.array([36.0]), np.array([128.0]))
    assert elev.dtype == np.float32
    assert float(elev[0]) > 70.0


def test_midnight_below_horizon() -> None:
    ts = _ts(["2024-06-22 00:00"])
    elev, _, _ = solar_position(ts, np.array([36.0]), np.array([128.0]))
    assert float(elev[0]) < 0.0


def test_symmetry_around_noon() -> None:
    ts = _ts(["2024-06-21 10:00", "2024-06-21 16:00"])
    elev, _, _ = solar_position(ts, np.array([36.0, 36.0]), np.array([128.0, 128.0]))
    assert abs(float(elev[0]) - float(elev[1])) < 10.0


def test_midnight_uses_previous_day_midpoint() -> None:
    ts0 = _ts(["2024-01-01 00:00"])
    e0, a0, c0 = solar_position(ts0, np.array([36.0]), np.array([128.0]))
    mid = pd.DatetimeIndex(ts0) - pd.Timedelta(minutes=30)
    doy = float(mid.dayofyear.to_numpy()[0])
    hour = 23.5
    g = 2 * np.pi / 365.0 * (doy - 1 + (hour - 12) / 24.0)
    eqtime = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g) - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    decl = 0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g) + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g)
    tst = hour * 60 + eqtime + 4 * (128.0 - 135.0)
    ha = np.radians(tst / 4.0 - 180.0)
    la = np.radians(36.0)
    cosz = np.clip(np.sin(la) * np.sin(decl) + np.cos(la) * np.cos(decl) * np.cos(ha), -1, 1)
    elev_ref = 90.0 - np.degrees(np.arccos(cosz))
    assert abs(float(e0[0]) - elev_ref) < 1e-3


def test_length_mismatch_rejected() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        solar_position(_ts(["2024-06-21 13:00"]), np.array([36.0, 37.0]), np.array([128.0]))


@pytest.mark.parametrize(("lat", "lon"), [(np.nan, 128), (36, np.inf), (91, 128), (36, 181)])
def test_invalid_coordinates_rejected(lat: float, lon: float) -> None:
    with pytest.raises(ValueError, match="coordinates"):
        solar_position(_ts(["2024-06-21 13:00"]), np.array([lat]), np.array([lon]))


def test_coordinate_matrix_rejected() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        solar_position(_ts(["2024-06-21 13:00"]), np.array([[36.0]]), np.array([128.0]))


def test_solar_rejects_naive_time() -> None:
    from src.oibc.timeaxis import TimeAxisError

    with pytest.raises(TimeAxisError, match="tz-naive"):
        solar_position(pd.Series(pd.to_datetime(["2024-06-21 13:00"])), np.array([36.0]), np.array([128.0]))


def test_outputs_have_physical_bounds_and_float32_dtype() -> None:
    ts = pd.Series(pd.date_range("2024-06-21 01:00", periods=24, freq="h", tz=KST))
    outputs = solar_position(ts, np.full(24, 36.0), np.full(24, 128.0))
    for output in outputs:
        assert output.shape == (24,)
        assert output.dtype == np.float32
        assert np.isfinite(output).all()
    elev, az, cosz = outputs
    assert ((elev >= -90) & (elev <= 90)).all()
    assert ((az >= 0) & (az < 360)).all()
    assert ((cosz >= -1) & (cosz <= 1)).all()
