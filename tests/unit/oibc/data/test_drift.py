import numpy as np
import pandas as pd
import pytest

from src.oibc.config import DriftConfig, FeatureConfig
from src.oibc.data.drift import DriftError, detect_breaks, splice_forecasts, used_raw_variables
from src.oibc.timeaxis import TimeAxisError

_KST = "Asia/Seoul"
_START = pd.Timestamp("2025-04-01", tz=_KST)
_N_BASE = 60
_N_TEST = 30
_TRAIN_END = _START + pd.Timedelta(days=_N_BASE)
_LEVELS = {"S1": 0.0, "S2": 30.0, "S3": 60.0}


def _panel(seed: int = 0, noise: float = 10.0) -> pd.DataFrame:
    """관측소마다 다른 수준(0/30/60)에 균일 잡음을 더한 합성 예보. 잡음은 범위를 넘지 않는다."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range(_START + pd.Timedelta(hours=1), periods=24 * (_N_BASE + _N_TEST), freq="h")
    frames = [
        pd.DataFrame(
            {
                "wa_station": station,
                "round": rnd,
                "ts": ts,
                "wa_cloud": level + rng.uniform(-noise, noise, len(ts)),
            }
        )
        for station, level in _LEVELS.items()
        for rnd in (10, 17)
    ]
    return pd.concat(frames, ignore_index=True)


def _test_day(df: pd.DataFrame, k: int) -> pd.Series:
    """시험 k번째 운영일(0부터)에 속한 행 마스크."""
    day = _TRAIN_END + pd.Timedelta(days=k)
    return (df["ts"] > day) & (df["ts"] <= day + pd.Timedelta(days=1))


def _detect(df: pd.DataFrame, variables=("wa_cloud",), cfg: DriftConfig | None = None):
    return detect_breaks(df, "wa", list(variables), _TRAIN_END, cfg or DriftConfig())


def test_detect_breaks_stationary_no_fire() -> None:
    assert _detect(_panel()) == {"wa_cloud": None}


def test_detect_breaks_dispersion_jump_first_date() -> None:
    df = _panel()
    jump = df["wa_station"].eq("S1") & (df["ts"] > _TRAIN_END + pd.Timedelta(days=10))
    df.loc[jump, "wa_cloud"] -= 40.0
    assert _detect(df)["wa_cloud"] == _TRAIN_END + pd.Timedelta(days=10)


def test_detect_breaks_range_share_threshold() -> None:
    df = _panel()
    idx_half = df.index[_test_day(df, 5)][:1]
    idx_two = df.index[_test_day(df, 7)][:3]
    df.loc[idx_half, "wa_cloud"] = 200.0
    df.loc[idx_two, "wa_cloud"] = 200.0
    assert _detect(df)["wa_cloud"] == _TRAIN_END + pd.Timedelta(days=7)


def test_detect_breaks_sticky_first() -> None:
    df = _panel()
    for k in (5, 20):
        idx = df.index[_test_day(df, k)][:3]
        df.loc[idx, "wa_cloud"] = 200.0
    assert _detect(df)["wa_cloud"] == _TRAIN_END + pd.Timedelta(days=5)


def test_detect_breaks_causal_invariance() -> None:
    df = _panel()
    after = df["ts"] > _TRAIN_END + pd.Timedelta(days=15)
    df.loc[after, "wa_cloud"] = 1.0e6
    result = _detect(df)["wa_cloud"]
    assert result == _TRAIN_END + pd.Timedelta(days=15)
    early = df["ts"] <= _TRAIN_END + pd.Timedelta(days=15)
    assert _detect(df[early].copy()) == {"wa_cloud": None}


def test_detect_breaks_right_closed_op_date() -> None:
    df = _panel()
    boundary = df["ts"].eq(_TRAIN_END + pd.Timedelta(days=4))
    df.loc[boundary, "wa_cloud"] = 1.0e6
    assert _detect(df)["wa_cloud"] == _TRAIN_END + pd.Timedelta(days=3)


def test_detect_breaks_duplicate_key_raises() -> None:
    df = _panel()
    with pytest.raises(DriftError):
        _detect(pd.concat([df, df.iloc[:1]], ignore_index=True))


def test_detect_breaks_missing_column_raises() -> None:
    with pytest.raises(DriftError):
        _detect(_panel().drop(columns=["wa_cloud"]))


def test_detect_breaks_zero_span_raises() -> None:
    df = _panel()
    base = df["ts"] <= _TRAIN_END
    df.loc[base, "wa_cloud"] = 5.0
    with pytest.raises(DriftError):
        _detect(df)


def test_detect_breaks_rejects_non_kst_timestamps() -> None:
    df = _panel()
    df["ts"] = df["ts"].dt.tz_convert("UTC")
    with pytest.raises(DriftError):
        _detect(df)


def test_used_raw_variables_splits_by_provider() -> None:
    default = FeatureConfig()
    cfg = FeatureConfig(wa_vars=tuple(v for v in default.wa_vars if v != "wa_ground_press"))
    split = used_raw_variables(cfg)
    assert "wa_ground_press" not in split["wa"]
    assert split["wa"] == cfg.wa_vars
    assert split["wb"] == default.wb_vars


def _fc(group: str, value: float, days: int = 4) -> pd.DataFrame:
    ts = pd.date_range(_START + pd.Timedelta(hours=1), periods=24 * days, freq="h")
    return pd.DataFrame({"group": group, "ts": ts, "forecast_kwh": value})


def _both(value: float, days: int = 4) -> pd.DataFrame:
    return pd.concat([_fc("G1", value, days), _fc("G2", value, days)], ignore_index=True)


def test_splice_switch_boundary() -> None:
    base = _both(1.0)
    fallback = _both(2.0)
    switch = _START + pd.Timedelta(days=2)
    out = splice_forecasts(base, fallback, switch)
    assert len(out) == len(base)
    assert set(zip(out["group"], out["ts"], strict=True)) == set(zip(base["group"], base["ts"], strict=True))
    op_days = out["ts"] - pd.Timedelta(hours=1)
    before = op_days < switch
    assert (out.loc[before.to_numpy(), "forecast_kwh"] == 1.0).all()
    assert (out.loc[~before.to_numpy(), "forecast_kwh"] == 2.0).all()


def test_splice_key_mismatch_raises() -> None:
    base = _both(1.0)
    fallback = _both(2.0).iloc[1:].reset_index(drop=True)
    with pytest.raises(DriftError):
        splice_forecasts(base, fallback, _START + pd.Timedelta(days=2))


def test_splice_rejects_non_midnight_switch() -> None:
    with pytest.raises((DriftError, TimeAxisError)):
        splice_forecasts(_both(1.0), _both(2.0), _START + pd.Timedelta(days=2, hours=5))


def test_detect_breaks_rejects_naive_train_end() -> None:
    with pytest.raises(DriftError):
        detect_breaks(_panel(), "wa", ["wa_cloud"], _TRAIN_END.tz_localize(None), DriftConfig())


def test_detect_breaks_empty_base_raises() -> None:
    df = _panel()
    with pytest.raises(DriftError):
        detect_breaks(df, "wa", ["wa_cloud"], _START - pd.Timedelta(days=1), DriftConfig())


def test_detect_breaks_all_nan_base_raises() -> None:
    df = _panel()
    df.loc[df["ts"] <= _TRAIN_END, "wa_cloud"] = np.nan
    with pytest.raises(DriftError):
        _detect(df)


def test_detect_breaks_single_station_disables_dispersion_only() -> None:
    df = _panel()
    df = df[df["wa_station"].eq("S1")].copy()
    assert _detect(df) == {"wa_cloud": None}
    shifted = df.copy()
    shifted.loc[shifted["ts"] > _TRAIN_END + pd.Timedelta(days=3), "wa_cloud"] = 1.0e4
    assert _detect(shifted)["wa_cloud"] == _TRAIN_END + pd.Timedelta(days=3)


def test_splice_missing_column_and_duplicate_key_raise() -> None:
    base = _both(1.0)
    switch = _START + pd.Timedelta(days=2)
    with pytest.raises(DriftError):
        splice_forecasts(base.drop(columns=["forecast_kwh"]), _both(2.0), switch)
    with pytest.raises(DriftError):
        splice_forecasts(pd.concat([base, base.iloc[:1]], ignore_index=True), _both(2.0), switch)


def test_splice_rejects_naive_switch() -> None:
    with pytest.raises(DriftError):
        splice_forecasts(_both(1.0), _both(2.0), pd.Timestamp("2025-04-03"))


def test_splice_rejects_non_kst_timestamps() -> None:
    base = _both(1.0)
    fallback = _both(2.0)
    base["ts"] = base["ts"].dt.tz_convert("UTC")
    fallback["ts"] = fallback["ts"].dt.tz_convert("UTC")
    with pytest.raises(DriftError):
        splice_forecasts(base, fallback, _START + pd.Timedelta(days=2))


def test_splice_rejects_negative_output() -> None:
    with pytest.raises(DriftError):
        splice_forecasts(_both(1.0), _both(-2.0), _START + pd.Timedelta(days=2))
