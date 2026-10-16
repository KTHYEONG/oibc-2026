import numpy as np
import pandas as pd
import pytest

from src.oibc.config import CompetitionSpec, FeatureConfig
from src.oibc.features.weather import (
    FeatureError,
    clear_sky_ghi,
    day_features,
    fleet_features,
    neighbor_features,
    round_revision_features,
    station_round_wide,
)
from src.oibc.timeaxis import issued_at, op_date

KST = "Asia/Seoul"
DT = "datetime64[us, Asia/Seoul]"
VAR = "wa_cloud"


def _forecast(
    stations: tuple[str, ...] = ("WA001",),
    days: int = 2,
    start: str = "2025-05-10",
    value: float | None = None,
) -> pd.DataFrame:
    spec = CompetitionSpec()
    ts = pd.Series(pd.date_range(f"{start} 01:00", periods=24 * days, freq="h", tz=KST), dtype=DT)
    rows = []
    for si, st in enumerate(stations):
        for r in spec.rounds:
            for t in ts:
                v = float(t.hour * 10 + r + si * 100) if value is None else value
                rows.append((st, r, t, v, v + 1.0))
    df = pd.DataFrame(rows, columns=["wa_station", "round", "ts", VAR, "wb_cloud"])
    df["round"] = df["round"].astype("int8")
    df["ts"] = df["ts"].astype(DT)
    df["op_date"] = op_date(df["ts"])
    df["issued_at"] = issued_at(df["op_date"], df["round"], spec)
    df["fc_source"] = pd.Series(["orig"] * len(df), dtype="string")
    for col in (VAR, "wb_cloud", "wb_precip_prob", "wb_uv_idx", "wa_rain", "wb_total_liq"):
        if col not in df.columns:
            df[col] = df[VAR]
        df[col] = df[col].astype("float32")
    return df.sort_values(["wa_station", "round", "ts"]).reset_index(drop=True)


def test_neighbor_blocks_issuance_boundary() -> None:
    spec = CompetitionSpec()
    df = _forecast()
    out = neighbor_features(df, "wa_station", (VAR,), (-1, 1), spec)
    for day in ("2025-05-10", "2025-05-11"):
        first = pd.Timestamp(f"{day} 01:00", tz=KST)
        last = pd.Timestamp(f"{day} 00:00", tz=KST) + pd.Timedelta(days=1)
        row_first = out[out["ts"] == first].iloc[0]
        row_last = out[out["ts"] == last].iloc[0]
        assert pd.isna(row_first[f"{VAR}_r17_n-1"])
        assert pd.isna(row_last[f"{VAR}_r17_n+1"])
    mid = pd.Timestamp("2025-05-10 12:00", tz=KST)
    row = out[out["ts"] == mid].iloc[0]
    assert row[f"{VAR}_r17_n+1"] == pytest.approx(float(13 * 10 + 17))
    assert row[f"{VAR}_r17_n-1"] == pytest.approx(float(11 * 10 + 17))


def test_day_aggregates_stay_within_op_day() -> None:
    spec = CompetitionSpec()
    cfg = FeatureConfig()
    df = _forecast(value=1.0)
    mask = df["op_date"] == pd.Timestamp("2025-05-11", tz=KST)
    for col in ("wa_cloud", "wb_cloud", "wb_precip_prob", "wb_uv_idx", "wa_rain", "wb_total_liq"):
        df.loc[mask, col] = 100.0
    out = day_features(df, "wa_station", cfg, spec)
    day = out[op_date(out["ts"]) == pd.Timestamp("2025-05-10", tz=KST)]
    assert (day[f"{VAR}_daymean"] == 1.0).all()
    assert ((day["wb_uv_idx_daymax"] == 1.0) | day["wb_uv_idx_daymax"].isna()).all()
    assert ((day[f"{VAR}_roll3"] == 1.0) | day[f"{VAR}_roll3"].isna()).all()
    assert ((day[f"{VAR}_d+2"] == 1.0) | day[f"{VAR}_d+2"].isna()).all()


def test_post_cutoff_mutation_leaves_past_features_unchanged() -> None:
    spec = CompetitionSpec()
    cfg = FeatureConfig()
    df = _forecast()
    tampered = df.copy()
    mask = tampered["op_date"] > pd.Timestamp("2025-05-10", tz=KST)
    tampered.loc[mask, VAR] = 9999.0
    tampered.loc[mask, "wb_cloud"] = -555.0
    past = op_date(df["ts"]) <= pd.Timestamp("2025-05-10", tz=KST)
    past_ts = df.loc[past, "ts"].drop_duplicates().sort_values().reset_index(drop=True)
    assert len(past_ts) > 0

    nb_a = neighbor_features(df, "wa_station", (VAR,), (-1, 1), spec)
    nb_b = neighbor_features(tampered, "wa_station", (VAR,), (-1, 1), spec)
    keep = op_date(nb_a["ts"]) <= pd.Timestamp("2025-05-10", tz=KST)
    pd.testing.assert_frame_equal(nb_a[keep].reset_index(drop=True), nb_b[keep].reset_index(drop=True))

    day_a = day_features(df, "wa_station", cfg, spec)
    day_b = day_features(tampered, "wa_station", cfg, spec)
    keep_d = op_date(day_a["ts"]) <= pd.Timestamp("2025-05-10", tz=KST)
    pd.testing.assert_frame_equal(day_a[keep_d].reset_index(drop=True), day_b[keep_d].reset_index(drop=True))

    fl_a = fleet_features(df, (VAR,), 17)
    fl_b = fleet_features(tampered, (VAR,), 17)
    keep_f = op_date(fl_a["ts"]) <= pd.Timestamp("2025-05-10", tz=KST)
    pd.testing.assert_frame_equal(fl_a[keep_f].reset_index(drop=True), fl_b[keep_f].reset_index(drop=True))

    wide_a = station_round_wide(df, "wa_station", (VAR,), spec)
    wide_b = station_round_wide(tampered, "wa_station", (VAR,), spec)
    keep_w = op_date(wide_a["ts"]) <= pd.Timestamp("2025-05-10", tz=KST)
    rd_a = round_revision_features(wide_a, (VAR,), spec)
    rd_b = round_revision_features(wide_b, (VAR,), spec)
    pd.testing.assert_frame_equal(rd_a[keep_w].reset_index(drop=True), rd_b[keep_w].reset_index(drop=True))


def test_daymean_uses_daytime_window() -> None:
    spec = CompetitionSpec()
    cfg = FeatureConfig()
    df = _forecast()
    hours = df["ts"].dt.hour
    df[VAR] = np.where(hours.between(9, 16), 10.0, 0.0).astype("float32")
    out = day_features(df, "wa_station", cfg, spec)
    assert (out[f"{VAR}_daymean"] == 10.0).all()


def test_fleet_averages_same_round_only() -> None:
    df = _forecast(stations=("WA001", "WA002"))
    df.loc[df["wa_station"] == "WA001", VAR] = 10.0
    df.loc[df["wa_station"] == "WA002", VAR] = 30.0
    df.loc[df["round"] == 10, VAR] = -100.0
    out = fleet_features(df, (VAR,), 17)
    assert (out[f"{VAR}_fleet"] == 20.0).all()


def test_round_revision_sign() -> None:
    spec = CompetitionSpec()
    df = _forecast(days=1)
    df.loc[df["round"] == 10, VAR] = 20.0
    df.loc[df["round"] == 17, VAR] = 50.0
    wide = station_round_wide(df, "wa_station", (VAR,), spec)
    out = round_revision_features(wide, (VAR,), spec)
    assert (out[f"{VAR}_rmean"] == 35.0).all()
    assert (out[f"{VAR}_rdiff"] == 30.0).all()


def test_clear_sky_ghi_bounds() -> None:
    cosz = np.array([-0.5, 0.0, 1.0])
    doy = np.array([172, 172, 172])
    ghi = clear_sky_ghi(cosz, doy)
    assert ghi.dtype == np.float32
    assert np.all(np.isfinite(ghi)) and (ghi >= 0).all()
    assert ghi[2] == ghi.max() and ghi[2] > 0


def test_all_nan_round_survives_wide_and_revision() -> None:
    df = _forecast(days=1)
    df.loc[df["round"] == 10, VAR] = np.nan
    original = df.copy(deep=True)
    wide = station_round_wide(df, "wa_station", (VAR,), CompetitionSpec())
    revision = round_revision_features(wide, (VAR,), CompetitionSpec())
    assert len(wide) == 24
    assert wide[f"{VAR}_r10"].isna().all()
    assert revision[[f"{VAR}_rmean", f"{VAR}_rdiff"]].isna().all().all()
    pd.testing.assert_frame_equal(df, original)


def test_day_sparse_hours_do_not_compress_time() -> None:
    df = _forecast(days=1)
    missing = pd.Timestamp("2025-05-10 12:00", tz=KST)
    df = df[df["ts"] != missing].copy()
    original = df.copy(deep=True)
    out = day_features(df, "wa_station", FeatureConfig(), CompetitionSpec()).set_index("ts")
    t = pd.Timestamp("2025-05-10 10:00", tz=KST)
    assert pd.isna(out.loc[t, f"{VAR}_d+2"])
    t = pd.Timestamp("2025-05-10 11:00", tz=KST)
    assert out.loc[t, f"{VAR}_roll3"] == pytest.approx((117 + 127) / 2)
    pd.testing.assert_frame_equal(df, original)


def test_all_missing_day_sum_stays_nan() -> None:
    df = _forecast(days=1)
    df["wa_rain"] = np.nan
    out = day_features(df, "wa_station", FeatureConfig(), CompetitionSpec())
    assert out["wa_rain_daysum"].isna().all()


@pytest.mark.parametrize("kind", ["late", "op_date", "no_issuance"])
def test_weather_rejects_causal_contract_violations(kind: str) -> None:
    from src.oibc.data.gate import LookAheadError

    df = _forecast(days=1)
    if kind == "late":
        df["issued_at"] = df["issued_at"] + pd.Timedelta(days=1)
    elif kind == "op_date":
        df["op_date"] = df["op_date"] - pd.Timedelta(days=1)
    else:
        df["issued_at"] = pd.NaT
    for transform in (
        lambda: station_round_wide(df, "wa_station", (VAR,), CompetitionSpec()),
        lambda: day_features(df, "wa_station", FeatureConfig(), CompetitionSpec()),
        lambda: fleet_features(df, (VAR,), 17),
    ):
        with pytest.raises((FeatureError, LookAheadError)):
            transform()


def test_fleet_requires_station_identity() -> None:
    df = _forecast(days=1).drop(columns="wa_station")
    with pytest.raises(FeatureError, match="station"):
        fleet_features(df, (VAR,), 17)


def test_wide_rejects_duplicate_keys() -> None:
    spec = CompetitionSpec()
    df = _forecast(days=1)
    dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    with pytest.raises(FeatureError):
        station_round_wide(dup, "wa_station", (VAR,), spec)


def test_key_contract_violations_rejected() -> None:
    spec = CompetitionSpec()
    df = _forecast(days=1)
    with pytest.raises(FeatureError):
        station_round_wide(df, "no_station", (VAR,), spec)
    with pytest.raises(FeatureError):
        station_round_wide(df.drop(columns=["ts"]), "wa_station", (VAR,), spec)
    bad = df.copy()
    bad.loc[bad.index[0], "wa_station"] = None
    with pytest.raises(FeatureError):
        station_round_wide(bad, "wa_station", (VAR,), spec)
    bad_round = df.copy()
    bad_round.loc[bad_round.index[0], "round"] = 11
    with pytest.raises(FeatureError):
        station_round_wide(bad_round, "wa_station", (VAR,), spec)
    with pytest.raises(FeatureError):
        station_round_wide(df, "wa_station", ("wa_missing_var",), spec)
    with pytest.raises(FeatureError):
        neighbor_features(df, "wa_station", ("wa_missing_var",), (-1,), spec)
    cut = df[~((df["round"] == 10) & (df["ts"] == df["ts"].iloc[0]))].reset_index(drop=True)
    with pytest.raises(FeatureError):
        station_round_wide(cut, "wa_station", (VAR,), spec)


def test_empty_offsets_return_keys_only() -> None:
    spec = CompetitionSpec()
    df = _forecast(days=1)
    out = neighbor_features(df, "wa_station", (VAR,), (), spec)
    assert list(out.columns) == ["wa_station", "ts"]


def test_neighbor_missing_round_yields_nan() -> None:
    spec = CompetitionSpec()
    df = _forecast(days=1)
    one_round = df[df["round"] == 10].reset_index(drop=True)
    out = neighbor_features(one_round, "wa_station", (VAR,), (-1,), spec)
    assert out[f"{VAR}_r17_n-1"].isna().all()
    assert out[f"{VAR}_r10_n-1"].notna().any()


def test_revision_and_fleet_contracts_rejected() -> None:
    spec = CompetitionSpec()
    df = _forecast(days=1)
    wide = station_round_wide(df, "wa_station", (VAR,), spec)
    with pytest.raises(FeatureError):
        round_revision_features(wide.drop(columns=[f"{VAR}_r10"]), (VAR,), spec)
    with pytest.raises(FeatureError):
        round_revision_features(wide[[f"{VAR}_r10", f"{VAR}_r17"]], (VAR,), spec)
    cfg = FeatureConfig()
    with pytest.raises(FeatureError):
        day_features(df.drop(columns=["wa_rain"]), "wa_station", cfg, spec)
    with pytest.raises(FeatureError):
        day_features(df[df["round"] == 99], "wa_station", cfg, spec)
    with pytest.raises(FeatureError):
        fleet_features(df.drop(columns=["ts"]), (VAR,), 17)
    with pytest.raises(FeatureError):
        fleet_features(df, ("wa_missing_var",), 17)
    with pytest.raises(FeatureError):
        fleet_features(df[df["round"] == 99], (VAR,), 17)
