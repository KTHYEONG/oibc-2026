import pandas as pd
import pytest

from src.oibc.config import CompetitionSpec
from src.oibc.data.gate import LookAheadError, assert_causal, forecast_neighbors, labels
from src.oibc.timeaxis import cutoff, issued_at, op_date

KST = "Asia/Seoul"
DT = "datetime64[us, Asia/Seoul]"


def _forecast(
    station: str = "WA001",
    rounds: tuple[int, ...] = (10, 17),
    day: str = "2025-06-02",
    n: int = 24,
) -> pd.DataFrame:
    spec = CompetitionSpec()
    ts = pd.Series(pd.date_range(f"{day} 01:00", periods=n, freq="h", tz=KST), dtype=DT)
    recs = [(station, rnd, t) for rnd in rounds for t in ts]
    df = pd.DataFrame(recs, columns=["wa_station", "round", "ts"])
    df["wa_station"] = df["wa_station"].astype("string")
    df["round"] = df["round"].astype("int8")
    df["ts"] = df["ts"].astype(DT)
    df["op_date"] = op_date(df["ts"])
    df["issued_at"] = issued_at(df["op_date"], df["round"], spec)
    df["wa_temp"] = (df["ts"].dt.hour * 1.5).astype("float32")
    return df.sort_values(["wa_station", "round", "ts"]).reset_index(drop=True)


def _gen_table() -> pd.DataFrame:
    ts = pd.Series(pd.date_range("2025-06-01 01:00", periods=6, freq="h", tz=KST), dtype=DT)
    return pd.DataFrame({"site_id": "PV001", "ts": ts, "energy": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})


def test_labels_never_return_rows_after_until() -> None:
    table = _gen_table()
    before = table.copy(deep=True)
    out = labels(table, until=pd.Timestamp("2025-06-01 03:00", tz=KST))
    assert (out["ts"] <= pd.Timestamp("2025-06-01 03:00", tz=KST)).all()
    assert len(out) == 3
    pd.testing.assert_frame_equal(table, before)


def test_labels_require_timezone_aware_until() -> None:
    with pytest.raises(TypeError):
        labels(_gen_table(), until=pd.Timestamp("2025-06-01 03:00"))


def test_previous_hour_neighbor_across_op_date_boundary_is_allowed() -> None:
    df = _forecast(day="2025-06-01", n=26)
    out = forecast_neighbors(df, "wa_station", ["wa_temp"], [-1], CompetitionSpec())
    row = out[out["ts"] == pd.Timestamp("2025-06-02 01:00", tz=KST)].iloc[0]
    assert row["wa_temp_lag-1"] == pytest.approx(0.0 * 1.5)


def test_next_hour_neighbor_from_future_issuance_is_blocked() -> None:
    df = _forecast(day="2025-06-02", n=25)
    out = forecast_neighbors(df, "wa_station", ["wa_temp"], [+1], CompetitionSpec())
    row = out[out["ts"] == pd.Timestamp("2025-06-03 00:00", tz=KST)].iloc[0]
    assert pd.isna(row["wa_temp_lag+1"])


def test_next_hour_neighbor_within_same_issuance_is_kept() -> None:
    df = _forecast(day="2025-06-02")
    out = forecast_neighbors(df, "wa_station", ["wa_temp"], [+1], CompetitionSpec())
    row = out[out["ts"] == pd.Timestamp("2025-06-02 12:00", tz=KST)].iloc[0]
    assert row["wa_temp_lag+1"] == pytest.approx(13 * 1.5)


def test_zero_offset_equals_original() -> None:
    df = _forecast()
    out = forecast_neighbors(df, "wa_station", ["wa_temp"], [0], CompetitionSpec())
    pd.testing.assert_series_equal(
        out["wa_temp_lag+0"], df["wa_temp"], check_names=False, check_dtype=False
    )


def test_assert_causal_raises_on_violation() -> None:
    spec = CompetitionSpec()
    target = pd.Series(pd.date_range("2025-06-02 12:00", periods=3, freq="h", tz=KST))
    late = cutoff(op_date(target), spec) + pd.Timedelta(hours=1)
    with pytest.raises(LookAheadError, match="3 rows"):
        assert_causal(target, late, spec)


def test_assert_causal_ignores_missing_issuance_and_rejects_length_mismatch() -> None:
    spec = CompetitionSpec()
    target = pd.Series(pd.date_range("2025-06-02 12:00", periods=2, freq="h", tz=KST))
    good = cutoff(op_date(target), spec)
    mixed = good.copy()
    mixed.iloc[0] = pd.NaT
    assert_causal(target, mixed, spec)
    with pytest.raises(ValueError, match="length"):
        assert_causal(target, mixed.iloc[[0]], spec)


def test_row_count_and_order_preserved_with_many_missing_neighbors() -> None:
    df = _forecast(n=6).iloc[::-1].reset_index(drop=True)
    before = df.copy(deep=True)
    out = forecast_neighbors(df, "wa_station", ["wa_temp"], [-2, -1, +1, +2], CompetitionSpec())
    assert len(out) == len(df)
    assert out["ts"].tolist() == df["ts"].tolist()
    pd.testing.assert_frame_equal(df, before)


def test_causal_checks_are_positional_with_different_and_duplicate_indices() -> None:
    target = pd.Series(pd.date_range("2025-06-02 12:00", periods=2, freq="h", tz=KST), index=[7, 7])
    issuance = pd.Series([
        pd.Timestamp("2025-06-01 17:00", tz=KST),
        pd.Timestamp("2025-06-01 17:01", tz=KST),
    ], index=[9, 8])
    with pytest.raises(LookAheadError, match="1 rows, first ts=2025-06-02 13:00"):
        assert_causal(target, issuance, CompetitionSpec())
    issuance.iloc[1] = pd.NaT
    assert_causal(target, issuance, CompetitionSpec())
    with pytest.raises(TypeError, match="tz-aware"):
        assert_causal(target, issuance.dt.tz_localize(None), CompetitionSpec())


def test_neighbors_preserve_duplicate_index_precision_station_and_round() -> None:
    df = _forecast(n=3)
    other = df.copy()
    other["wa_station"] = "WA002"
    other["wa_temp"] += 100
    df = pd.concat([df, other], ignore_index=True).sample(frac=1, random_state=42)
    df["wa_temp"] = df["wa_temp"].astype("float64") + df["round"] + 0.123456789
    df.index = [4] * len(df)
    before = df.copy(deep=True)
    out = forecast_neighbors(df, "wa_station", ["wa_temp"], [0, 1], CompetitionSpec())
    pd.testing.assert_series_equal(out["wa_temp_lag+0"], df["wa_temp"], check_names=False)
    for pos in range(len(df)):
        row = df.iloc[pos]
        donor = df[(df["wa_station"] == row["wa_station"]) & (df["round"] == row["round"]) &
                   (df["ts"] == row["ts"] + pd.Timedelta(hours=1))]
        if donor.empty:
            assert pd.isna(out.iloc[pos]["wa_temp_lag+1"])
        else:
            assert out.iloc[pos]["wa_temp_lag+1"] == donor.iloc[0]["wa_temp"]
    pd.testing.assert_frame_equal(df, before)


def test_unknown_issuance_cannot_supply_values() -> None:
    df = _forecast(n=3)
    df.loc[df["ts"].dt.hour == 2, "issued_at"] = pd.NaT
    out = forecast_neighbors(df, "wa_station", ["wa_temp"], [1], CompetitionSpec())
    assert out.loc[out["ts"].dt.hour == 1, "wa_temp_lag+1"].isna().all()


@pytest.mark.parametrize("defect", ["duplicate", "missing", "round", "timezone"])
def test_invalid_forecast_keys_fail_closed(defect: str) -> None:
    df = _forecast(n=3)
    if defect == "duplicate":
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    elif defect == "missing":
        df.loc[0, "wa_station"] = pd.NA
    elif defect == "round":
        df.loc[0, "round"] = 9
    else:
        df["issued_at"] = df["issued_at"].dt.tz_localize(None)
    with pytest.raises(ValueError, match=r"key|round|tz-naive"):
        forecast_neighbors(df, "wa_station", ["wa_temp"], [1], CompetitionSpec())


def test_future_issuance_corruption_cannot_change_prior_day_features() -> None:
    df = _forecast(day="2025-06-01", n=50)
    spec = CompetitionSpec()
    original = forecast_neighbors(df, "wa_station", ["wa_temp"], [1, 2], spec)
    future = df["ts"] >= pd.Timestamp("2025-06-02 01:00", tz=KST)
    df.loc[future, "wa_temp"] = 1_000_000
    corrupted = forecast_neighbors(df, "wa_station", ["wa_temp"], [1, 2], spec)
    pd.testing.assert_frame_equal(original.loc[~future], corrupted.loc[~future])
