import pandas as pd
import pytest

from src.oibc.config import CompetitionSpec
from src.oibc.timeaxis import TimeAxisError, cutoff, issued_at, k_weight, op_date, require_hourly_kst

KST = "Asia/Seoul"


def _s(vals: list[str]) -> pd.Series:
    return pd.Series(pd.DatetimeIndex(pd.to_datetime(vals).tz_localize(KST)))


def test_midnight_belongs_to_previous_op_date() -> None:
    out = op_date(_s(["2025-06-02 00:00"]))
    assert out.iloc[0] == pd.Timestamp("2025-06-01 00:00", tz=KST)


def test_first_hour_same_op_date() -> None:
    out = op_date(_s(["2025-06-01 01:00"]))
    assert out.iloc[0] == pd.Timestamp("2025-06-01 00:00", tz=KST)


def test_all_rounds_precede_cutoff() -> None:
    spec = CompetitionSpec()
    days = pd.date_range("2025-06-01", periods=365, freq="D", tz=KST)
    for r in (10, 17):
        op = pd.Series(days)
        cut = cutoff(op, spec)
        iss = issued_at(op, pd.Series([r] * 365), spec)
        assert bool((iss <= cut).all())
    op = pd.Series(days)
    iss17 = issued_at(op, pd.Series([17] * 365), spec)
    assert bool((iss17 == cutoff(op, spec)).all())


def test_k_weight_matches_peak_hours() -> None:
    spec = CompetitionSpec()
    ts = pd.Series(pd.date_range("2025-06-01 01:00", periods=24, freq="h", tz=KST))
    w = k_weight(ts, spec)
    assert w.dtype == "int8"
    assert int((w == 3).sum()) == 5
    assert set(ts[w == 3].dt.hour) == {11, 12, 13, 14, 15}


def test_naive_rejected() -> None:
    naive = pd.Series(pd.to_datetime(["2025-06-01 01:00"]))
    with pytest.raises(TimeAxisError, match="tz-naive"):
        op_date(naive)
    with pytest.raises(TimeAxisError, match="tz-naive"):
        k_weight(naive, CompetitionSpec())


def test_non_hourly_rejected() -> None:
    ts = _s(["2025-06-01 16:55"])
    with pytest.raises(TimeAxisError, match="hourly"):
        require_hourly_kst(ts, "ts")


def test_unknown_round_rejected() -> None:
    op = _s(["2025-06-01 00:00"])
    with pytest.raises(TimeAxisError, match="rounds"):
        issued_at(op, pd.Series([12]), CompetitionSpec())


def test_non_datetime_rejected() -> None:
    with pytest.raises(TimeAxisError, match="not datetime"):
        op_date(pd.Series(["not-a-timestamp-xyz"]))


def test_wrong_timezone_rejected() -> None:
    ts = pd.Series(pd.date_range("2025-06-01 01:00", periods=2, freq="h", tz="UTC"))
    with pytest.raises(TimeAxisError, match="Asia/Seoul"):
        op_date(ts)


@pytest.mark.parametrize("value", [10.9, 17.1, None, float("nan"), "10"])
def test_invalid_rounds_are_not_coerced(value: object) -> None:
    op = _s(["2025-06-01 00:00"])
    with pytest.raises(TimeAxisError, match="rounds"):
        issued_at(op, pd.Series([value]))


def test_round_index_mismatch_rejected() -> None:
    op = _s(["2025-06-01 00:00", "2025-06-02 00:00"])
    with pytest.raises(TimeAxisError, match="index"):
        issued_at(op, pd.Series([10, 17], index=[1, 0]))


def test_missing_timestamp_rejected() -> None:
    ts = pd.Series([pd.NaT], dtype="datetime64[ns, Asia/Seoul]")
    with pytest.raises(TimeAxisError, match="missing"):
        require_hourly_kst(ts, "ts")


def test_submicrosecond_rejected() -> None:
    ts = _s(["2025-06-01 01:00"]) + pd.Timedelta(nanoseconds=1)
    with pytest.raises(TimeAxisError, match="hourly"):
        require_hourly_kst(ts, "ts")


def test_operating_date_must_be_midnight() -> None:
    op = _s(["2025-06-01 01:00"])
    with pytest.raises(TimeAxisError, match="midnight"):
        cutoff(op, CompetitionSpec())
    with pytest.raises(TimeAxisError, match="midnight"):
        issued_at(op, pd.Series([17]))


def test_time_outputs_preserve_index_and_units() -> None:
    ts = _s(["2025-06-01 01:00", "2025-06-02 00:00"])
    ts.index = pd.Index([8, 3], name="row")
    dates = op_date(ts)
    spec = CompetitionSpec()
    for result in (dates, cutoff(dates, spec), issued_at(dates, pd.Series([10, 17], index=ts.index))):
        assert result.index.equals(ts.index)
        assert str(result.dtype) == "datetime64[us, Asia/Seoul]"
    assert k_weight(ts, spec).index.equals(ts.index)


def test_empty_typed_inputs_supported() -> None:
    ts = pd.Series([], dtype="datetime64[us, Asia/Seoul]")
    spec = CompetitionSpec()
    assert op_date(ts).empty
    assert cutoff(ts, spec).empty
    assert issued_at(ts, pd.Series([], dtype="int64")).empty
    assert k_weight(ts, spec).empty
