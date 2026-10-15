from pathlib import Path

import pandas as pd
import pytest

from src.oibc.config import CompetitionSpec, DataPaths
from src.oibc.data.derived import (
    DerivedError,
    build_derived,
    build_grid,
    build_solar,
    fill_forecast_rounds,
)
from src.oibc.data.gate import LookAheadError, assert_causal
from src.oibc.timeaxis import issued_at, op_date

KST = "Asia/Seoul"
DT = "datetime64[us, Asia/Seoul]"


def _sites(n: int = 3) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "site_id": pd.Series([f"PV{i:03d}" for i in range(1, n + 1)], dtype="string"),
            "capacity_kw": pd.Series([100.0] * n, dtype="float32"),
            "lat": [36.0 + i * 0.1 for i in range(n)],
            "lon": [127.0 + i * 0.1 for i in range(n)],
            "wa_station": pd.Series(["WA001"] * n, dtype="string"),
            "wb_station": pd.Series(["WB001"] * n, dtype="string"),
        }
    )


def _spec(n_sites: int = 3, n_hours: int = 48) -> CompetitionSpec:
    start = pd.Timestamp("2025-05-10 01:00", tz=KST)
    return CompetitionSpec(
        n_sites=n_sites, train_end=start - pd.Timedelta(hours=1), eval_start=start,
        eval_end=start + pd.Timedelta(hours=n_hours - 1), n_eval_hours=n_hours,
    )


def _smp(day: str = "2025-05-10", n: int = 48, period: str = "eval") -> pd.DataFrame:
    ts = pd.Series(pd.date_range(f"{day} 01:00", periods=n, freq="h", tz=KST), dtype=DT)
    return pd.DataFrame({"ts": ts, "period": pd.Series([period] * n, dtype="string")})


def _forecast(day: str = "2025-05-10", n: int = 24) -> pd.DataFrame:
    spec = CompetitionSpec()
    ts = pd.Series(pd.date_range(f"{day} 01:00", periods=n, freq="h", tz=KST), dtype=DT)
    recs = [(rnd, t) for rnd in spec.rounds for t in ts]
    df = pd.DataFrame(recs, columns=["round", "ts"])
    df["wa_station"] = pd.Series(["WA001"] * len(df), dtype="string")
    df["round"] = df["round"].astype("int8")
    df["ts"] = df["ts"].astype(DT)
    df["op_date"] = op_date(df["ts"])
    df["issued_at"] = issued_at(df["op_date"], df["round"], spec)
    df["lead_h"] = ((df["ts"] - df["issued_at"]).dt.total_seconds() // 3600).astype("int16")
    df["period"] = pd.Series(["train"] * len(df), dtype="string")
    df["wa_temp"] = (df["ts"].dt.hour * 2.0 + df["round"]).astype("float32")
    df["wa_cloud"] = (df["ts"].dt.hour * 0.5).astype("float32")
    return df.sort_values(["wa_station", "round", "ts"]).reset_index(drop=True)


def test_grid_has_full_site_by_time_product() -> None:
    grid = build_grid(_sites(3), _smp(n=48), spec=_spec())
    assert len(grid) == 144
    assert not grid.duplicated(subset=["site_id", "ts"]).any()
    ordered = grid.sort_values(["site_id", "ts"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(grid, ordered)
    assert list(grid.columns) == ["site_id", "ts", "op_date", "period", "capacity_kw", "wa_station", "wb_station"]


def test_grid_rejects_incomplete_eval_window() -> None:
    with pytest.raises(DerivedError):
        build_grid(_sites(3), _smp(n=47, period="eval"), spec=_spec())


def test_missing_round_filled_from_other_round_with_donor_issuance() -> None:
    df = _forecast()
    missing_ts = pd.Timestamp("2025-05-10 12:00", tz=KST)
    cut = df[~((df["round"] == 17) & (df["ts"] == missing_ts))].reset_index(drop=True)
    filled = fill_forecast_rounds(cut, "wa_station", CompetitionSpec())
    row = filled[(filled["round"] == 17) & (filled["ts"] == missing_ts)].iloc[0]
    donor = df[(df["round"] == 10) & (df["ts"] == missing_ts)].iloc[0]
    assert row["fc_source"] == "other_round"
    assert row["wa_temp"] == pytest.approx(donor["wa_temp"])
    assert row["issued_at"] == donor["issued_at"]
    assert int(row["lead_h"]) == 26


def test_both_rounds_missing_stays_missing() -> None:
    df = _forecast()
    missing_ts = pd.Timestamp("2025-05-10 05:00", tz=KST)
    cut = df[df["ts"] != missing_ts].reset_index(drop=True)
    filled = fill_forecast_rounds(cut, "wa_station", CompetitionSpec())
    rows = filled[filled["ts"] == missing_ts]
    assert len(rows) == 2
    assert pd.isna(rows["wa_temp"]).all()
    assert (rows["fc_source"] == "missing").all()
    assert rows["issued_at"].isna().all()


def test_existing_nan_cells_untouched() -> None:
    df = _forecast()
    df.loc[df["wa_temp"].idxmax(), "wa_cloud"] = float("nan")
    filled = fill_forecast_rounds(df, "wa_station", CompetitionSpec())
    assert filled["fc_source"].eq("orig").all()
    assert filled["wa_cloud"].isna().sum() == 1


def test_filled_forecast_passes_causal_gate() -> None:
    df = _forecast()
    cut = df.iloc[1:].reset_index(drop=True)
    filled = fill_forecast_rounds(cut, "wa_station", CompetitionSpec())
    assert_causal(filled["ts"], filled["issued_at"], CompetitionSpec())


def test_derived_build_does_not_read_label_tables(tmp_path: Path) -> None:
    spec = _spec(n_sites=2, n_hours=24)
    canonical_dir = tmp_path / "canonical"
    canonical_dir.mkdir()
    sites = _sites(2)
    smp = _smp(n=24)
    sites.to_parquet(canonical_dir / "sites.parquet", index=False)
    smp.to_parquet(canonical_dir / "smp.parquet", index=False)
    for provider, station in (("wa", "wa_station"), ("wb", "wb_station")):
        fc = _forecast(n=24).rename(columns={"wa_station": station})
        if provider == "wb":
            fc = fc.rename(columns={"wa_temp": "wb_temp", "wa_cloud": "wb_cloud"})
        fc.to_parquet(canonical_dir / f"forecast_{provider}.parquet", index=False)
    paths = DataPaths(raw_dir=tmp_path / "raw", canonical_dir=canonical_dir, derived_dir=tmp_path / "derived")
    counts = build_derived(paths, spec)
    assert counts["grid"] == 48
    assert counts["solar"] == 48
    assert (tmp_path / "derived" / "grid.parquet").exists()
    assert (tmp_path / "derived" / "solar.parquet").exists()
    solar = pd.read_parquet(tmp_path / "derived" / "solar.parquet")
    assert len(solar) == 48
    assert solar[["elev", "azim", "cosz"]].dtypes.eq("float32").all()


@pytest.mark.parametrize("defect", ["empty_eval", "duplicate_time", "shifted_time", "period", "null_train_period", "site", "capacity", "station"])
def test_grid_rejects_invalid_keys_and_eval_alignment(defect: str) -> None:
    sites, smp = _sites(), _smp()
    if defect == "empty_eval":
        smp = smp.iloc[:0]
    elif defect == "duplicate_time":
        smp.loc[0, "ts"] = smp.loc[1, "ts"]
    elif defect == "shifted_time":
        smp.loc[0, "ts"] = smp["ts"].max() + pd.Timedelta(hours=1)
    elif defect == "period":
        smp.loc[0, "period"] = "train"
    elif defect == "null_train_period":
        train = smp.iloc[[0]].copy()
        train["ts"] -= pd.Timedelta(hours=1)
        train["period"] = pd.NA
        smp = pd.concat([train, smp], ignore_index=True)
    elif defect == "site":
        sites.loc[0, "site_id"] = sites.loc[1, "site_id"]
    elif defect == "capacity":
        sites.loc[0, "capacity_kw"] = float("nan")
    else:
        sites.loc[0, "wb_station"] = pd.NA
    with pytest.raises(DerivedError):
        build_grid(sites, smp, spec=_spec())


def test_fill_multiple_stations_whole_hour_gaps_and_all_nan_donors() -> None:
    df = _forecast(n=4)
    other = df.copy()
    other["wa_station"] = "WA002"
    other["wa_temp"] += 100
    df = pd.concat([df, other], ignore_index=True)
    absent = ((df["wa_station"] == "WA001") & (df["round"] == 10) & (df["ts"].dt.hour == 2))
    absent |= ((df["wa_station"] == "WA002") & (df["round"] == 17) & (df["ts"].dt.hour == 3))
    absent |= df["ts"].dt.hour == 4
    df.loc[(df["wa_station"] == "WA002") & (df["ts"].dt.hour == 3), ["wa_temp", "wa_cloud"]] = float("nan")
    # 양 끝 시각을 유지하고 내부의 완전 결측 시각을 만든다.
    tail = _forecast(n=5).query("ts.dt.hour == 5")
    tail_other = tail.copy()
    tail_other["wa_station"] = "WA002"
    df = pd.concat([df[~absent], tail, tail_other], ignore_index=True).sample(frac=1, random_state=42)
    before = df.copy(deep=True)
    filled = fill_forecast_rounds(df, "wa_station", CompetitionSpec())
    assert len(filled) == 20
    assert filled.loc[filled["ts"].dt.hour == 4, "fc_source"].eq("missing").all()
    gifts = filled[filled["fc_source"] == "other_round"].sort_values("wa_station")
    assert gifts["wa_station"].tolist() == ["WA001", "WA002"]
    assert gifts.iloc[0]["wa_temp"] == 21
    assert pd.isna(gifts.iloc[1]["wa_temp"])
    assert gifts.iloc[1]["issued_at"].hour == 10
    assert str(filled["lead_h"].dtype) == "Int16"
    assert str(filled["wa_temp"].dtype) == "float32"
    pd.testing.assert_frame_equal(df, before)


def test_empty_forecast_fails_with_clear_error() -> None:
    with pytest.raises(DerivedError, match="empty"):
        fill_forecast_rounds(_forecast().iloc[:0], "wa_station", CompetitionSpec())


@pytest.mark.parametrize("protected", ["raw", "canonical", "symlink"])
def test_derived_outputs_cannot_overwrite_sources(tmp_path: Path, protected: str) -> None:
    raw = tmp_path / "raw"
    canonical = tmp_path / "canonical"
    raw.mkdir()
    canonical.mkdir()
    original = canonical / "original.parquet"
    original.write_bytes(b"original")
    if protected == "symlink":
        output = tmp_path / "derived"
        output.mkdir()
        (output / "grid.parquet").symlink_to(original)
    else:
        output = (raw if protected == "raw" else canonical) / "nested"
    with pytest.raises(DerivedError, match="overwrite"):
        build_derived(DataPaths(raw_dir=raw, canonical_dir=canonical, derived_dir=output), _spec())
    assert original.read_bytes() == b"original"


def test_derived_rejects_missing_sites_before_writing_tables(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    _sites(2).to_parquet(canonical / "sites.parquet", index=False)
    _smp().to_parquet(canonical / "smp.parquet", index=False)
    output = tmp_path / "derived"
    with pytest.raises(DerivedError, match="sites: rows"):
        build_derived(DataPaths(canonical_dir=canonical, derived_dir=output), _spec())
    assert not list(output.glob("*.parquet"))


def test_solar_rejects_ambiguous_site_coordinates() -> None:
    sites = _sites()
    grid = build_grid(sites, _smp(), spec=_spec())
    with pytest.raises(DerivedError, match="duplicate site_id"):
        build_solar(grid, pd.concat([sites, sites.iloc[[0]]]))


def test_fill_rejects_late_source_issuance() -> None:
    df = _forecast()
    df.loc[df["round"] == 17, "issued_at"] += pd.Timedelta(hours=1)
    with pytest.raises(LookAheadError, match="24 rows"):
        fill_forecast_rounds(df, "wa_station", CompetitionSpec())
