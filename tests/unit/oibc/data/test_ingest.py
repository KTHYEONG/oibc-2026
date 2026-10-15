import json
from pathlib import Path

import pandas as pd
import pytest

from src.oibc.config import CompetitionSpec, DataPaths, QualityConfig
from src.oibc.data import schemas
from src.oibc.data.ingest import (
    IngestError,
    build_canonical,
    canonical_forecast,
    canonical_generation,
    canonical_observed,
    canonical_sites,
    canonical_smp,
)
from src.oibc.timeaxis import cutoff

KST = "Asia/Seoul"
DT = "datetime64[us, Asia/Seoul]"

WA_VARS = [
    "wa_cloud",
    "wa_temp",
    "wa_temp_max",
    "wa_temp_min",
    "wa_humidity",
    "wa_ground_press",
    "wa_wind_speed",
    "wa_wind_dir",
    "wa_rain",
    "wa_snow",
]

WB_VARS = [
    "wb_temp",
    "wb_feel_temp",
    "wb_wet_bulb_temp",
    "wb_dew_point",
    "wb_wind_speed",
    "wb_wind_dir",
    "wb_wind_gust_spd",
    "wb_rel_hum",
    "wb_vis",
    "wb_ceiling",
    "wb_uv_idx",
    "wb_precip_prob",
    "wb_rain_prob",
    "wb_snow_prob",
    "wb_ice_prob",
    "wb_total_liq",
    "wb_rain",
    "wb_snow",
    "wb_ice",
    "wb_cloud",
]

WB_OB_VARS = [
    "wb_temp",
    "wb_feel_temp",
    "wb_feel_temp_shade",
    "wb_rel_hum",
    "wb_dew_point",
    "wb_wind_dir",
    "wb_wind_speed",
    "wb_uv_idx",
    "wb_vis",
    "wb_cloud",
    "wb_ceiling",
    "wb_pressure",
    "wb_appr_temp",
    "wb_wind_chill_temp",
    "wb_precip_1h",
]


def _hours(day: str, n: int = 24) -> pd.Series:
    return pd.Series(pd.date_range(f"{day} 01:00", periods=n, freq="h", tz=KST), dtype=DT)


def _forecast_frame(
    provider: str, station: str, days: list[str], rounds: tuple[int, ...] = (10, 17)
) -> pd.DataFrame:
    variables = WA_VARS if provider == "wa" else WB_VARS
    rows = [
        (station, rnd, ts, *([1.0] * len(variables)))
        for day in days
        for rnd in rounds
        for ts in _hours(day)
    ]
    cols = [f"{provider}_station", "round", "ts", *variables]
    df = pd.DataFrame(rows, columns=cols)
    df[f"{provider}_station"] = df[f"{provider}_station"].astype("string")
    return df


def test_forecast_issued_before_cutoff_for_all_rows() -> None:
    train = _forecast_frame("wa", "WA001", ["2025-06-01"])
    eval_ = _forecast_frame("wa", "WA001", ["2025-06-02"])
    out, _ = canonical_forecast(train, eval_, "wa", CompetitionSpec())
    cut = cutoff(out["op_date"], CompetitionSpec())
    assert bool((out["issued_at"] <= cut).all())
    late = out[out["round"] == 17]["lead_h"]
    assert int(late.min()) == 8
    assert int(late.max()) == 31


def test_midnight_forecast_row_maps_to_previous_op_date_issuance() -> None:
    df = _forecast_frame("wa", "WA001", ["2025-06-01"], rounds=(10,))
    row = df[df["ts"] == pd.Timestamp("2025-06-02 00:00", tz=KST)].iloc[[0]]
    out, _ = canonical_forecast(row, row.iloc[0:0], "wa", CompetitionSpec())
    assert out.iloc[0]["issued_at"] == pd.Timestamp("2025-05-31 10:00", tz=KST)


def test_overlapping_train_eval_forecast_keys_rejected() -> None:
    df = _forecast_frame("wa", "WA001", ["2025-06-01"])
    with pytest.raises(IngestError, match="overlap"):
        canonical_forecast(df, df, "wa", CompetitionSpec())


def test_non_hourly_rows_excluded_and_counted() -> None:
    df = _forecast_frame("wa", "WA001", ["2025-06-01"])
    bad = df.iloc[[0]].copy()
    bad["ts"] = pd.Series(
        pd.DatetimeIndex(["2025-06-01 16:55"]).tz_localize(KST), dtype=DT
    )
    df = pd.concat([df, bad], ignore_index=True)
    out, report = canonical_forecast(df, df.iloc[0:0], "wa", CompetitionSpec())
    assert len(out) == len(df) - 1
    assert report.excluded["non_hourly"] == 1


def test_duplicate_observed_keys_drop_all_copies() -> None:
    ts = pd.Series(pd.DatetimeIndex(["2025-06-01 01:00"]).tz_localize(KST), dtype=DT)
    raw = pd.DataFrame(
        {
            "wa_station": pd.Series(["WA001", "WA001"], dtype="string"),
            "ts": pd.concat([ts, ts], ignore_index=True),
            **{c: pd.Series([1.0, 2.0], dtype="float64") for c in WA_VARS},
        }
    )
    out, report = canonical_observed(raw, "wa", QualityConfig())
    assert len(out) == 0
    assert report.excluded["duplicate_key"] == 2


def test_wind_speed_converted_to_si_for_provider_b() -> None:
    df = _forecast_frame("wb", "WB001", ["2025-06-01"], rounds=(10,))
    df["wb_wind_speed"] = 36.0
    df["wb_wind_gust_spd"] = 18.0
    out, _ = canonical_forecast(df, df.iloc[0:0], "wb", CompetitionSpec())
    assert out["wb_wind_speed"].iloc[0] == pytest.approx(10.0, rel=1e-6)
    wa = _forecast_frame("wa", "WA001", ["2025-06-01"], rounds=(10,))
    wa["wa_wind_speed"] = 10.0
    out_wa, _ = canonical_forecast(wa, wa.iloc[0:0], "wa", CompetitionSpec())
    assert out_wa["wa_wind_speed"].iloc[0] == pytest.approx(10.0, rel=1e-6)


def test_missing_forecast_rows_are_not_filled() -> None:
    df = _forecast_frame("wa", "WA001", ["2025-06-01"])
    df = df.iloc[1:].reset_index(drop=True)
    out, _ = canonical_forecast(df, df.iloc[0:0], "wa", CompetitionSpec())
    assert len(out) == len(df)


def _smp_frames() -> tuple[pd.DataFrame, pd.DataFrame, CompetitionSpec]:
    spec = CompetitionSpec(
        train_end=pd.Timestamp("2025-06-01 00:00", tz=KST),
        eval_start=pd.Timestamp("2025-06-01 01:00", tz=KST),
        eval_end=pd.Timestamp("2025-06-02 00:00", tz=KST),
        n_eval_hours=24,
    )
    train_ts = pd.Series(pd.date_range("2025-05-31 01:00", periods=24, freq="h", tz=KST))
    eval_ts = pd.Series(pd.date_range("2025-06-01 01:00", periods=24, freq="h", tz=KST))
    train = pd.DataFrame({"ts": train_ts.astype(str), "smp": [50.0] * 24})
    k = [3 if t.hour in (11, 12, 13, 14, 15) else 1 for t in eval_ts]
    eval_ = pd.DataFrame({"ts": eval_ts.astype(str), "smp": [60.0] * 24, "k": k})
    return train, eval_, spec


def test_smp_k_mismatch_rejected() -> None:
    train, eval_, spec = _smp_frames()
    eval_.loc[eval_["ts"].str.contains("13:00"), "k"] = 1
    with pytest.raises(IngestError, match="k mismatch"):
        canonical_smp(train, eval_, spec)


def test_smp_period_split_at_train_end() -> None:
    train, eval_, spec = _smp_frames()
    out, _ = canonical_smp(train, eval_, spec)
    assert out[out["ts"] == pd.Timestamp("2025-06-01 00:00", tz=KST)]["period"].iloc[0] == "train"
    assert out[out["ts"] == pd.Timestamp("2025-06-01 01:00", tz=KST)]["period"].iloc[0] == "eval"


def test_naive_ts_rejected() -> None:
    train, eval_, spec = _smp_frames()
    train["ts"] = ["2025-05-31 01:00"] * 24
    with pytest.raises(IngestError, match="tz-naive"):
        canonical_smp(train, eval_, spec)


def test_smp_train_eval_overlap_rejected() -> None:
    train, eval_, spec = _smp_frames()
    train = pd.concat([train, eval_[["ts", "smp"]]], ignore_index=True)
    with pytest.raises(IngestError, match="overlap"):
        canonical_smp(train, eval_, spec)


def test_smp_eval_grid_gap_rejected() -> None:
    train, eval_, spec = _smp_frames()
    with pytest.raises(IngestError, match="grid"):
        canonical_smp(train, eval_.iloc[1:].reset_index(drop=True), spec)


def _mini_gen() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "site_id": pd.Series(["PV001"], dtype="string"),
            "ts": pd.Series(
                pd.DatetimeIndex(["2025-06-01 01:00"]).tz_localize(KST), dtype=DT
            ),
            "energy": [1.0],
        }
    )


def test_sites_duplicate_rejected() -> None:
    raw = _raw_sites(2)
    raw.loc[1, "site_id"] = "PV001"
    with pytest.raises(IngestError, match="duplicate site_id"):
        canonical_sites(raw, _mini_gen(), CompetitionSpec(n_sites=2))


def test_sites_nonpositive_capacity_rejected() -> None:
    raw = _raw_sites(2)
    raw.loc[0, "capacity_kw"] = 0.0
    with pytest.raises(IngestError, match="capacity"):
        canonical_sites(raw, _mini_gen(), CompetitionSpec(n_sites=2))


def test_forecast_round_outside_spec_rejected() -> None:
    df = _forecast_frame("wa", "WA001", ["2025-06-01"])
    df.loc[0, "round"] = 12
    with pytest.raises(IngestError, match="round"):
        canonical_forecast(df, df.iloc[0:0], "wa", CompetitionSpec())


def test_forecast_issued_after_cutoff_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.oibc.data.ingest as ingest_mod

    monkeypatch.setattr(ingest_mod, "cutoff", lambda op, spec: op - pd.Timedelta(days=1))
    df = _forecast_frame("wa", "WA001", ["2025-06-01"])
    with pytest.raises(IngestError, match="cutoff"):
        canonical_forecast(df, df.iloc[0:0], "wa", CompetitionSpec())


def test_git_rev_failure_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.oibc.data.ingest as ingest_mod

    def _boom(*args: object, **kwargs: object) -> object:
        raise OSError("no git")

    monkeypatch.setattr(ingest_mod.subprocess, "run", _boom)
    assert ingest_mod._git_rev() is None

    class _Proc:
        returncode = 1
        stdout = ""

    monkeypatch.setattr(ingest_mod.subprocess, "run", lambda *a, **k: _Proc())
    assert ingest_mod._git_rev() is None


def _raw_sites(n: int = 2) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "site_id": [f"PV{i + 1:03d}" for i in range(n)],
            "current_group": ["A"] * n,
            "capacity_kw": [100.0] * n,
            "lat": [36.0] * n,
            "lon": [128.0] * n,
            "wa_station": ["WA001"] * n,
            "wb_station": ["WB001"] * n,
            "wa_dist": [1000.0] * n,
            "wb_dist": [1000.0] * n,
        }
    )


def test_sites_without_generation_keep_nat_first_ts() -> None:
    spec = CompetitionSpec(n_sites=2)
    gen = pd.DataFrame(
        {
            "site_id": pd.Series(["PV001"], dtype="string"),
            "ts": pd.Series(
                pd.DatetimeIndex(["2025-06-01 01:00"]).tz_localize(KST), dtype=DT
            ),
            "energy": [1.0],
        }
    )
    out, _ = canonical_sites(_raw_sites(2), gen, spec)
    assert len(out) == 2
    assert out[out["site_id"] == "PV001"]["first_ts"].notna().all()
    assert out[out["site_id"] == "PV002"]["first_ts"].isna().all()


def test_wrong_site_count_rejected() -> None:
    spec = CompetitionSpec(n_sites=2)
    gen = pd.DataFrame(
        {
            "site_id": pd.Series(["PV001"], dtype="string"),
            "ts": pd.Series(
                pd.DatetimeIndex(["2025-06-01 01:00"]).tz_localize(KST), dtype=DT
            ),
            "energy": [1.0],
        }
    )
    with pytest.raises(IngestError, match="rows"):
        canonical_sites(_raw_sites(3), gen, spec)


def test_generation_unknown_site_rejected() -> None:
    raw = pd.DataFrame(
        {
            "site_id": pd.Series(["PV999"], dtype="string"),
            "ts": pd.Series(
                pd.DatetimeIndex(["2025-06-01 01:00"]).tz_localize(KST), dtype=DT
            ),
            "energy": [1.0],
        }
    )
    with pytest.raises(IngestError, match="unknown site"):
        canonical_generation(raw, _raw_sites(2), QualityConfig())


def _mini_spec() -> CompetitionSpec:
    return CompetitionSpec(
        n_sites=2,
        train_end=pd.Timestamp("2025-06-01 00:00", tz=KST),
        eval_start=pd.Timestamp("2025-06-01 01:00", tz=KST),
        eval_end=pd.Timestamp("2025-06-02 00:00", tz=KST),
        n_eval_hours=24,
    )


def _write_mini_raw(raw_dir: Path, spec: CompetitionSpec) -> None:
    (raw_dir / "train").mkdir(parents=True, exist_ok=True)
    (raw_dir / "test").mkdir(parents=True, exist_ok=True)
    sites = _raw_sites(spec.n_sites)
    sites.to_csv(raw_dir / "sites.csv", index=False)
    train_ts = pd.Series(pd.date_range("2025-05-31 01:00", periods=24, freq="h", tz=KST))
    eval_ts = pd.Series(pd.date_range("2025-06-01 01:00", periods=24, freq="h", tz=KST))
    train_smp = pd.DataFrame({"ts": train_ts.astype(str), "smp": [50.0] * 24})
    k = [3 if t.hour in (11, 12, 13, 14, 15) else 1 for t in eval_ts]
    eval_smp = pd.DataFrame({"ts": eval_ts.astype(str), "smp": [60.0] * 24, "k": k})
    train_smp.to_csv(raw_dir / "train/smp.csv", index=False)
    eval_smp.to_csv(raw_dir / "smp.csv", index=False)
    gen_rows = [
        (f"PV{i + 1:03d}", ts, 10.0) for i in range(spec.n_sites) for ts in train_ts
    ]
    pd.DataFrame(gen_rows, columns=["site_id", "ts", "energy"]).to_parquet(
        raw_dir / "train/generation.parquet", index=False
    )
    for provider, fc_vars, ob_vars in (
        ("wa", WA_VARS, WA_VARS),
        ("wb", WB_VARS, WB_OB_VARS),
    ):
        station = f"{provider.upper()}001"
        fc_rows = [
            (station, rnd, ts, *([1.0] * len(fc_vars)))
            for rnd in (10, 17)
            for ts in train_ts
        ]
        pd.DataFrame(
            fc_rows, columns=[f"{provider}_station", "round", "ts", *fc_vars]
        ).to_parquet(raw_dir / f"train/forecast_{provider}.parquet", index=False)
        fc_rows = [
            (station, rnd, ts, *([1.0] * len(fc_vars)))
            for rnd in (10, 17)
            for ts in eval_ts
        ]
        pd.DataFrame(
            fc_rows, columns=[f"{provider}_station", "round", "ts", *fc_vars]
        ).to_parquet(raw_dir / f"test/forecast_{provider}.parquet", index=False)
        obs_rows = [(station, ts, *([1.0] * len(ob_vars))) for ts in train_ts]
        pd.DataFrame(
            obs_rows, columns=[f"{provider}_station", "ts", *ob_vars]
        ).to_parquet(raw_dir / f"train/observed_{provider}.parquet", index=False)


def test_build_canonical_writes_all_tables_and_deterministic_manifest(
    tmp_path: Path,
) -> None:
    spec = _mini_spec()
    raw_dir = tmp_path / "raw"
    canonical_dir = tmp_path / "canonical"
    _write_mini_raw(raw_dir, spec)
    before = {p: p.stat().st_mtime_ns for p in raw_dir.rglob("*") if p.is_file()}
    paths = DataPaths(raw_dir=raw_dir, canonical_dir=canonical_dir)
    build_canonical(paths, spec, QualityConfig())
    with open(canonical_dir / "manifest.json", encoding="utf-8") as handle:
        first = json.load(handle)
    build_canonical(paths, spec, QualityConfig())
    with open(canonical_dir / "manifest.json", encoding="utf-8") as handle:
        second = json.load(handle)
    assert first["tables"] == second["tables"]
    assert first["files"] == second["files"]
    check = {
        "sites": schemas.SITES,
        "smp": schemas.SMP,
        "generation": schemas.GENERATION,
        "forecast_wa": schemas.FORECAST_WA,
        "forecast_wb": schemas.FORECAST_WB,
        "observed_wa": schemas.OBSERVED_WA,
        "observed_wb": schemas.OBSERVED_WB,
    }
    for name, schema in check.items():
        frame = pd.read_parquet(canonical_dir / f"{name}.parquet")
        schemas.validate_frame(frame, schema)
    after = {p: p.stat().st_mtime_ns for p in raw_dir.rglob("*") if p.is_file()}
    assert before == after


@pytest.mark.parametrize("invalid", [pd.NaT, pd.Timestamp("2025-06-01 01:00", tz="UTC")])
def test_timestamp_contract_rejects_missing_or_wrong_timezone(invalid: pd.Timestamp) -> None:
    raw = _mini_gen()
    raw["ts"] = pd.Series([invalid], dtype="datetime64[ns, UTC]" if invalid is not pd.NaT else DT)
    with pytest.raises(IngestError, match="ts"):
        canonical_generation(raw, _raw_sites(), QualityConfig())


def test_submicrosecond_timestamp_is_excluded_without_rounding() -> None:
    raw = _mini_gen()
    raw["ts"] = raw["ts"].astype("datetime64[ns, Asia/Seoul]") + pd.Timedelta(nanoseconds=1)
    out, report = canonical_generation(raw, _raw_sites(), QualityConfig())
    assert out.empty
    assert report.excluded == {"non_hourly": 1}


@pytest.mark.parametrize("missing", [True, False])
def test_eval_k_is_required_and_null_is_rejected(missing: bool) -> None:
    train, eval_, spec = _smp_frames()
    if missing:
        eval_ = eval_.drop(columns="k")
    else:
        eval_["k"] = eval_["k"].astype("Int8")
        eval_.loc[0, "k"] = pd.NA
    with pytest.raises(IngestError, match="k"):
        canonical_smp(train, eval_, spec)


def test_train_rows_cannot_patch_missing_eval_smp() -> None:
    train, eval_, spec = _smp_frames()
    train = pd.concat([train, eval_.iloc[[0]][["ts", "smp"]]], ignore_index=True)
    with pytest.raises(IngestError, match="grid"):
        canonical_smp(train, eval_.iloc[1:], spec)


def test_duplicate_null_station_keys_cannot_be_silently_excluded() -> None:
    raw = pd.DataFrame({"wa_station": [None, None], "ts": [_hours("2025-06-01", 1).iloc[0]] * 2})
    for col in WA_VARS:
        raw[col] = 0.0
    with pytest.raises(IngestError, match="key"):
        canonical_observed(raw, "wa", QualityConfig())


def test_unknown_generation_site_rejected_even_if_row_would_be_excluded() -> None:
    raw = _mini_gen()
    raw["site_id"] = "PV999"
    raw["ts"] += pd.Timedelta(minutes=5)
    with pytest.raises(IngestError, match="unknown site"):
        canonical_generation(raw, _raw_sites(), QualityConfig())


@pytest.mark.parametrize("capacity", [float("nan"), float("inf"), -1.0])
def test_sites_capacity_must_be_finite_positive(capacity: float) -> None:
    raw = _raw_sites()
    raw.loc[0, "capacity_kw"] = capacity
    with pytest.raises(IngestError, match="capacity"):
        canonical_sites(raw, _mini_gen(), CompetitionSpec(n_sites=2))


def test_first_ts_uses_only_retained_generation_rows() -> None:
    raw = _mini_gen()
    early = raw.copy()
    early["ts"] -= pd.Timedelta(hours=2)
    non_hour = raw.copy()
    non_hour["ts"] -= pd.Timedelta(minutes=5)
    generation = pd.concat([early, early, non_hour, raw], ignore_index=True)
    sites, _ = canonical_sites(_raw_sites(), generation, CompetitionSpec(n_sites=2))
    gen, report = canonical_generation(generation, sites, QualityConfig())
    assert sites.loc[0, "first_ts"] == gen["ts"].min()
    assert report.excluded == {"non_hourly": 1, "duplicate_key": 2}


@pytest.mark.parametrize("nested", [False, True])
def test_build_rejects_outputs_inside_raw_without_writing(tmp_path: Path, nested: bool) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    output = raw_dir / "nested" if nested else raw_dir
    with pytest.raises(IngestError, match="raw"):
        build_canonical(DataPaths(raw_dir=raw_dir, canonical_dir=output), _mini_spec(), QualityConfig())
    assert list(raw_dir.iterdir()) == []


def test_build_rejects_output_symlink_into_raw(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    output = tmp_path / "canonical"
    raw_dir.mkdir()
    output.mkdir()
    original = raw_dir / "generation.parquet"
    original.write_bytes(b"original")
    (output / "generation.parquet").symlink_to(original)
    with pytest.raises(IngestError, match="raw"):
        build_canonical(DataPaths(raw_dir=raw_dir, canonical_dir=output), _mini_spec(), QualityConfig())
    assert original.read_bytes() == b"original"


def test_observed_wind_bounds_use_si_and_preserve_missing_cells() -> None:
    raw = pd.DataFrame({"wb_station": ["WB001"] * 3, "ts": _hours("2025-06-01", 3)})
    for col in WB_OB_VARS:
        raw[col] = 0.0
    raw["wb_pressure"] = 1000.0
    raw["wb_wind_speed"] = [36.0, 216.0, float("nan")]
    before = raw.copy(deep=True)
    out, report = canonical_observed(raw, "wb", QualityConfig())
    assert out["wb_wind_speed"].iloc[:2].tolist() == pytest.approx([10.0, 60.0])
    assert pd.isna(out["wb_wind_speed"].iloc[2])
    assert out["dq_sentinel"].tolist() == [False, True, False]
    assert report.flags == {"dq_sentinel": 1}
    pd.testing.assert_frame_equal(raw, before)
