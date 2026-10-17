import numpy as np
import pandas as pd
import pytest

from src.oibc.config import CalibrationConfig, CompetitionSpec
from src.oibc.models.calibrate import (
    CalibrationError,
    CloudCalibration,
    apply_calibration,
    cloud_index,
    fit_cloud_calibration,
    scale_factors,
)

KST = "Asia/Seoul"
DT = "datetime64[us, Asia/Seoul]"


def _cfg(**kwargs: object) -> CalibrationConfig:
    return CalibrationConfig(**kwargs)  # type: ignore[arg-type]


def _ts(hours: list[str]) -> pd.Series:
    return pd.Series(pd.DatetimeIndex(pd.to_datetime(hours).tz_localize(KST)), dtype=DT)


def _hourly(ts: pd.Series, f: np.ndarray, a: np.ndarray, smp: float = 100.0, k: int = 1) -> pd.DataFrame:
    return pd.DataFrame({
        "ts": ts.astype(DT),
        "forecast_kwh": np.asarray(f, dtype="float64"),
        "actual_kwh": np.asarray(a, dtype="float64"),
        "smp": np.full(len(ts), smp, dtype="float64"),
        "k": np.full(len(ts), k, dtype="float64"),
    })


def _index(ts_unique: pd.DatetimeIndex, values: list[float]) -> pd.Series:
    idx = pd.DatetimeIndex(ts_unique, dtype=DT)
    return pd.Series(np.asarray(values, dtype="float64"), index=idx, dtype="float64")


def test_cloud_index_nanmean_and_scale() -> None:
    ts = _ts(["2024-06-01 10:00", "2024-06-01 10:00", "2024-06-01 11:00", "2024-06-01 11:00"])
    frame = pd.DataFrame({
        "ts": ts.astype(DT),
        "wa_cloud_fleet": [20.0, 20.0, 40.0, 40.0],
        "wb_cloud_fleet": [np.nan, np.nan, 60.0, 60.0],
    })
    out = cloud_index(frame, _cfg())
    assert out.index.is_monotonic_increasing
    assert out.tolist() == pytest.approx([0.2, 0.5])


def test_cloud_index_all_nan_is_nan() -> None:
    ts = _ts(["2024-06-01 10:00", "2024-06-01 11:00"])
    frame = pd.DataFrame({
        "ts": ts.astype(DT),
        "wa_cloud_fleet": [np.nan, 30.0],
        "wb_cloud_fleet": [np.nan, 50.0],
    })
    out = cloud_index(frame, _cfg())
    assert np.isnan(out.iloc[0])
    assert out.iloc[1] == pytest.approx(0.4)


def test_cloud_index_contract() -> None:
    ts = _ts(["2024-06-01 10:00", "2024-06-01 10:00"])
    dup = pd.DataFrame({
        "ts": ts.astype(DT),
        "wa_cloud_fleet": [10.0, 20.0],
        "wb_cloud_fleet": [10.0, 10.0],
    })
    with pytest.raises(CalibrationError):
        cloud_index(dup, _cfg())
    bad_range = pd.DataFrame({
        "ts": ts.astype(DT),
        "wa_cloud_fleet": [120.0, 120.0],
        "wb_cloud_fleet": [10.0, 10.0],
    })
    with pytest.raises(CalibrationError):
        cloud_index(bad_range, _cfg())
    missing = pd.DataFrame({"ts": ts.astype(DT), "wa_cloud_fleet": [10.0, 10.0]})
    with pytest.raises(CalibrationError):
        cloud_index(missing, _cfg())
    nat = pd.DataFrame({
        "ts": pd.Series([pd.NaT, pd.NaT]),
        "wa_cloud_fleet": [10.0, 10.0],
        "wb_cloud_fleet": [10.0, 10.0],
    })
    with pytest.raises(CalibrationError):
        cloud_index(nat, _cfg())
    non_dt = pd.DataFrame({
        "ts": pd.Series([object(), object()]),
        "wa_cloud_fleet": [10.0, 10.0],
        "wb_cloud_fleet": [10.0, 10.0],
    })
    with pytest.raises(CalibrationError):
        cloud_index(non_dt, _cfg())
    non_numeric = pd.DataFrame({
        "ts": ts.astype(DT),
        "wa_cloud_fleet": ["x", "x"],
        "wb_cloud_fleet": [10.0, 10.0],
    })
    with pytest.raises(CalibrationError):
        cloud_index(non_numeric, _cfg())
    dup_cols = pd.DataFrame(
        [[ts.iloc[0], 10.0, 10.0, ts.iloc[0]], [ts.iloc[1], 10.0, 10.0, ts.iloc[1]]],
        columns=["ts", "wa_cloud_fleet", "wb_cloud_fleet", "ts"],
    )
    with pytest.raises(CalibrationError):
        cloud_index(dup_cols, _cfg())


def test_scale_factors_missing_ts_identity() -> None:
    ts_all = pd.DatetimeIndex(pd.to_datetime(["2024-06-01 10:00", "2024-06-01 11:00", "2024-06-01 12:00"]).tz_localize(KST), dtype=DT)
    index = _index(ts_all[:2], [0.5, np.nan])
    cal = CloudCalibration(a=0.97, b=0.04, at_boundary=False, n_fit_hours=2)
    ts = pd.Series(pd.DatetimeIndex([ts_all[0], ts_all[1], ts_all[2]]), dtype=DT)
    out = scale_factors(ts, index, cal)
    assert out[1] == pytest.approx(1.0)
    assert out[2] == pytest.approx(1.0)
    assert out[0] == pytest.approx(0.97 + 0.04 * 0.5)


def test_scale_factors_clipped_nonnegative() -> None:
    ts_all = pd.DatetimeIndex(pd.to_datetime(["2024-06-01 10:00"]).tz_localize(KST), dtype=DT)
    index = _index(ts_all, [0.5])
    cal = CloudCalibration(a=0.0, b=-1.0, at_boundary=True, n_fit_hours=1)
    ts = pd.Series(pd.DatetimeIndex(ts_all), dtype=DT)
    assert scale_factors(ts, index, cal).tolist() == [0.0]


def test_scale_factors_rejects_bad_ts_and_calibration() -> None:
    stamps = pd.DatetimeIndex(pd.to_datetime(["2024-06-01 10:00"]).tz_localize(KST), dtype=DT)
    index = _index(stamps, [0.5])
    good = CloudCalibration(a=1.0, b=0.0, at_boundary=False, n_fit_hours=1)
    ts = pd.Series(pd.DatetimeIndex(stamps), dtype=DT)
    with pytest.raises(CalibrationError):
        scale_factors(pd.Series([pd.NaT]), index, good)
    bad = CloudCalibration(a=float("inf"), b=0.0, at_boundary=False, n_fit_hours=1)
    with pytest.raises(CalibrationError):
        scale_factors(ts, index, bad)
    bad_index = pd.Series(["x"], index=pd.DatetimeIndex(stamps, dtype=DT))
    with pytest.raises(CalibrationError):
        scale_factors(ts, bad_index, good)


def _fit_frame(levels: list[float], per_level: int, scale: object) -> tuple[pd.DataFrame, pd.Series]:
    stamps: list[pd.Timestamp] = []
    cvals: list[float] = []
    fvals: list[float] = []
    avals: list[float] = []
    base = pd.Timestamp("2024-06-01 06:00", tz=KST)
    for li, c in enumerate(levels):
        for r in range(per_level):
            t = base + pd.Timedelta(hours=li * per_level + r)
            stamps.append(t)
            cvals.append(c)
            f = 100.0 + 5.0 * r
            fvals.append(f)
            s = scale(c) if callable(scale) else float(scale)  # type: ignore[operator]
            avals.append(f * s)
    ts = pd.Series(pd.DatetimeIndex(stamps, dtype=DT), dtype=DT)
    uniq = pd.DatetimeIndex(sorted(set(stamps)), dtype=DT)
    per_ts = dict(zip(stamps, cvals, strict=True))
    index = pd.Series(
        np.array([per_ts[t] for t in uniq], dtype="float64"),
        index=pd.DatetimeIndex(uniq, dtype=DT),
        dtype="float64",
    )
    hourly = _hourly(ts, np.array(fvals), np.array(avals))
    return hourly, index


def test_fit_recovers_known_linear_bias() -> None:
    hourly, index = _fit_frame([0.0, 0.5, 1.0], 4, lambda c: 0.97 + 0.04 * c)
    got = fit_cloud_calibration(hourly, index, _cfg(), CompetitionSpec())
    assert (got.a, got.b) == pytest.approx((0.97, 0.04))
    assert got.at_boundary is False


def test_fit_identity_when_unbiased() -> None:
    hourly, index = _fit_frame([0.0, 0.5, 1.0], 3, 1.0)
    got = fit_cloud_calibration(hourly, index, _cfg(), CompetitionSpec())
    assert (got.a, got.b) == (pytest.approx(1.0), pytest.approx(0.0))


def test_fit_never_worse_than_identity() -> None:
    rng = np.random.default_rng(7)
    n = 24
    base = pd.Timestamp("2024-06-01 06:00", tz=KST)
    stamps = [base + pd.Timedelta(hours=i) for i in range(n)]
    ts = pd.Series(pd.DatetimeIndex(stamps, dtype=DT), dtype=DT)
    cvals = rng.uniform(0.0, 1.0, n)
    f = rng.uniform(50.0, 150.0, n)
    a = np.abs(f + rng.normal(0.0, 10.0, n))
    hourly = _hourly(ts, f, a, smp=120.0, k=3)
    uniq = pd.DatetimeIndex(stamps, dtype=DT)
    index = pd.Series(cvals, index=uniq, dtype="float64")
    cfg = _cfg()
    got = fit_cloud_calibration(hourly, index, cfg, CompetitionSpec())
    s_best = scale_factors(ts, index, got)
    s_ident = scale_factors(ts, index, CloudCalibration(a=1.0, b=0.0, at_boundary=False, n_fit_hours=n))
    w = 0.10 * hourly["smp"].to_numpy(dtype="float64") * hourly["k"].to_numpy(dtype="float64")
    j_best = float((w * np.abs(s_best * f - a)).sum())
    j_ident = float((w * np.abs(s_ident * f - a)).sum())
    assert j_best <= j_ident


def test_fit_boundary_flag() -> None:
    hourly, index = _fit_frame([0.0, 0.5, 1.0], 3, 0.5)
    got = fit_cloud_calibration(hourly, index, _cfg(), CompetitionSpec())
    assert got.at_boundary is True


def test_fit_deterministic_tie_break() -> None:
    hourly, index = _fit_frame([0.0, 0.5], 3, 1.0)
    hourly = hourly.copy()
    hourly["smp"] = 0.0
    cfg = _cfg()
    spec = CompetitionSpec()
    first = fit_cloud_calibration(hourly, index, cfg, spec)
    second = fit_cloud_calibration(hourly, index, cfg, spec)
    assert (first.a, first.b) == (pytest.approx(cfg.a_min), pytest.approx(cfg.b_min))
    assert first == second


def test_fit_contract() -> None:
    cfg = _cfg()
    spec = CompetitionSpec()
    ts = _ts(["2024-06-01 10:00"])
    uniq = pd.DatetimeIndex(pd.to_datetime(["2024-06-01 10:00"]).tz_localize(KST), dtype=DT)
    index = pd.Series([0.5], index=uniq, dtype="float64")
    with pytest.raises(CalibrationError):
        fit_cloud_calibration(_hourly(ts[:0], np.array([]), np.array([])), index, cfg, spec)
    with pytest.raises(CalibrationError):
        fit_cloud_calibration(_hourly(ts, np.array([10.0]), np.array([np.nan])), index, cfg, spec)
    with pytest.raises(CalibrationError):
        fit_cloud_calibration(_hourly(ts, np.array([-1.0]), np.array([5.0])), index, cfg, spec)
    with pytest.raises(CalibrationError):
        fit_cloud_calibration(pd.DataFrame({"ts": ts.astype(DT)}), index, cfg, spec)
    nat_hourly = _hourly(ts, np.array([10.0]), np.array([9.0]))
    nat_hourly["ts"] = pd.Series([pd.NaT])
    with pytest.raises(CalibrationError):
        fit_cloud_calibration(nat_hourly, index, cfg, spec)
    non_numeric = _hourly(ts, np.array([10.0]), np.array([9.0]))
    non_numeric["forecast_kwh"] = ["x"]
    with pytest.raises(CalibrationError):
        fit_cloud_calibration(non_numeric, index, cfg, spec)
    bad_smp = _hourly(ts, np.array([10.0]), np.array([9.0]), smp=float("inf"))
    with pytest.raises(CalibrationError):
        fit_cloud_calibration(bad_smp, index, cfg, spec)
    overflow = _hourly(ts, np.array([10.0]), np.array([9.0]), smp=1e308, k=1e308)
    with pytest.raises(CalibrationError):
        fit_cloud_calibration(overflow, index, cfg, spec)


def test_apply_seed_axis_broadcast() -> None:
    stamps = pd.DatetimeIndex(pd.to_datetime(["2024-06-01 10:00", "2024-06-01 11:00", "2024-06-01 12:00"]).tz_localize(KST), dtype=DT)
    index = _index(stamps, [0.0, 0.5, 1.0])
    cal = CloudCalibration(a=0.9, b=0.1, at_boundary=False, n_fit_hours=3)
    ts = pd.Series(pd.DatetimeIndex(stamps), dtype=DT)
    preds = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0], [70.0, 80.0, 90.0]])
    before = preds.copy()
    out = apply_calibration(preds, ts, index, cal)
    s = np.array([0.9, 0.95, 1.0])
    np.testing.assert_allclose(out, preds * s[None, :])
    one = apply_calibration(preds[0], ts, index, cal)
    np.testing.assert_allclose(out[0], one)
    np.testing.assert_array_equal(preds, before)


def test_apply_identity_calibration() -> None:
    stamps = pd.DatetimeIndex(pd.to_datetime(["2024-06-01 10:00", "2024-06-01 11:00"]).tz_localize(KST), dtype=DT)
    index = _index(stamps, [0.3, 0.7])
    cal = CloudCalibration(a=1.0, b=0.0, at_boundary=False, n_fit_hours=2)
    ts = pd.Series(pd.DatetimeIndex(stamps), dtype=DT)
    preds = np.array([[10.0, 20.0], [30.0, 40.0]])
    out = apply_calibration(preds, ts, index, cal)
    assert np.array_equal(out, preds)


def test_apply_contract() -> None:
    stamps = pd.DatetimeIndex(pd.to_datetime(["2024-06-01 10:00", "2024-06-01 11:00"]).tz_localize(KST), dtype=DT)
    index = _index(stamps, [0.3, 0.7])
    cal = CloudCalibration(a=1.0, b=0.0, at_boundary=False, n_fit_hours=2)
    ts = pd.Series(pd.DatetimeIndex(stamps), dtype=DT)
    with pytest.raises(CalibrationError):
        apply_calibration(np.array([[1.0, 2.0, 3.0]]), ts, index, cal)
    with pytest.raises(CalibrationError):
        apply_calibration(np.array([-1.0, 2.0]), ts, index, cal)
    with pytest.raises(CalibrationError):
        apply_calibration(np.array([1.0, np.nan]), ts, index, cal)
    with pytest.raises(CalibrationError):
        apply_calibration(np.array([1.0]), ts, index, cal)
    with pytest.raises(CalibrationError):
        apply_calibration(np.zeros((2, 2, 2)), ts, index, cal)
    with pytest.raises(CalibrationError):
        apply_calibration(np.array([], dtype="float64"), ts[:0], index, cal)
    with pytest.raises(CalibrationError):
        apply_calibration(np.array(["x", "y"]), ts, index, cal)
    huge_index = _index(stamps, [1e308, 1e308])
    huge_cal = CloudCalibration(a=1.05, b=0.15, at_boundary=True, n_fit_hours=2)
    with pytest.raises(CalibrationError):
        apply_calibration(np.full(2, 1e308), ts, huge_index, huge_cal)
