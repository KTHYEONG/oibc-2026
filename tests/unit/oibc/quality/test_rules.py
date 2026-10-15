import pandas as pd
import pytest

from src.oibc.config import QualityConfig
from src.oibc.quality.rules import generation_flags, observed_sentinel
from src.oibc.timeaxis import op_date

KST = "Asia/Seoul"
DT = "datetime64[us, Asia/Seoul]"


def _sites(n: int = 1, capacity: float = 100.0) -> pd.DataFrame:
    ids = [f"PV{i + 1:03d}" for i in range(n)]
    return pd.DataFrame(
        {
            "site_id": pd.Series(ids, dtype="string"),
            "capacity_kw": pd.Series([capacity] * n, dtype="float64"),
            "lat": pd.Series([36.0] * n, dtype="float64"),
            "lon": pd.Series([128.0] * n, dtype="float64"),
        }
    )


def _gen(site: str, start: str, periods: int, energy: float) -> pd.DataFrame:
    ts = pd.Series(pd.date_range(start, periods=periods, freq="h", tz=KST), dtype=DT)
    return pd.DataFrame(
        {
            "site_id": pd.Series([site] * periods, dtype="string"),
            "ts": ts,
            "op_date": op_date(ts),
            "energy": pd.Series([energy] * periods, dtype="float64"),
        }
    )


def test_flags_do_not_alter_rows_or_values() -> None:
    gen = _gen("PV001", "2025-06-01 01:00", 48, 10.0)
    before = gen.copy(deep=True)
    out = generation_flags(gen, _sites(), QualityConfig())
    assert len(out) == 48
    assert out["ts"].equals(before["ts"])
    assert (out["energy"] == before["energy"]).all()
    assert list(gen.columns) == list(before.columns)


def test_negative_energy_flagged() -> None:
    gen = _gen("PV001", "2025-06-01 12:00", 2, 0.0)
    gen.loc[0, "energy"] = -0.5
    out = generation_flags(gen, _sites(), QualityConfig())
    assert bool(out.loc[0, "dq_negative"])
    assert not bool(out.loc[1, "dq_negative"])


def test_over_capacity_flagged_at_ratio_boundary() -> None:
    gen = _gen("PV001", "2025-06-01 12:00", 2, 0.0)
    gen.loc[0, "energy"] = 105.0
    gen.loc[1, "energy"] = 105.01
    out = generation_flags(gen, _sites(capacity=100.0), QualityConfig())
    assert not bool(out.loc[0, "dq_over_cap"])
    assert bool(out.loc[1, "dq_over_cap"])


def test_night_generation_flagged_only_below_threshold() -> None:
    gen = pd.concat(
        [
            _gen("PV001", "2025-06-01 00:00", 1, 5.0),
            _gen("PV001", "2025-06-01 12:00", 1, 5.0),
        ],
        ignore_index=True,
    )
    out = generation_flags(gen, _sites(capacity=100.0), QualityConfig())
    assert bool(out.loc[0, "dq_night_gen"])
    assert not bool(out.loc[1, "dq_night_gen"])


def test_zero_day_flags_whole_site_day() -> None:
    first = _gen("PV001", "2025-06-01 01:00", 24, 0.0)
    second = _gen("PV001", "2025-06-02 01:00", 24, 0.0)
    second.loc[11, "energy"] = 50.0
    gen = pd.concat([first, second], ignore_index=True)
    out = generation_flags(gen, _sites(capacity=100.0), QualityConfig())
    assert bool(out.iloc[:24]["dq_zero_day"].all())
    assert not bool(out.iloc[24:]["dq_zero_day"].any())


def test_partial_zero_day_not_flagged() -> None:
    gen = _gen("PV001", "2025-06-01 01:00", 24, 0.0)
    gen.loc[11, "energy"] = 50.0
    out = generation_flags(gen, _sites(capacity=100.0), QualityConfig())
    assert not bool(out["dq_zero_day"].any())


def test_unknown_site_rejected() -> None:
    gen = _gen("PV999", "2025-06-01 01:00", 2, 0.0)
    with pytest.raises(ValueError, match="PV999"):
        generation_flags(gen, _sites(), QualityConfig())


def test_sentinel_detects_out_of_bounds_only() -> None:
    obs = pd.DataFrame(
        {
            "wb_station": pd.Series(["WB001"] * 3, dtype="string"),
            "wb_cloud": pd.Series([150.0, 100.0, float("nan")], dtype="float32"),
        }
    )
    flag = observed_sentinel(obs, QualityConfig())
    assert list(flag) == [True, False, False]


def test_unknown_daytime_energy_does_not_imply_zero_day() -> None:
    gen = _gen("PV001", "2025-06-01 01:00", 24, 0.0)
    gen.loc[11, "energy"] = float("nan")
    out = generation_flags(gen, _sites(), QualityConfig())
    assert not out["dq_zero_day"].any()
    assert pd.isna(out.loc[11, "energy"])


def test_no_daylight_and_repeated_index_preserve_row_flags() -> None:
    gen = _gen("PV001", "2025-06-01 01:00", 2, 0.0)
    gen.index = pd.Index([7, 7])
    before = gen.copy(deep=True)
    out = generation_flags(gen, _sites(), QualityConfig())
    assert not out["dq_zero_day"].any()
    pd.testing.assert_frame_equal(out[before.columns], before)
    pd.testing.assert_frame_equal(gen, before)
