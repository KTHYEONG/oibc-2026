from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.oibc.config import CompetitionSpec, DataPaths, FeatureConfig
from src.oibc.features.panel import PanelError, build_features, build_panel, feature_columns
from src.oibc.timeaxis import issued_at, op_date

KST = "Asia/Seoul"
DT = "datetime64[us, Asia/Seoul]"
ST = "string"


def _spec(n_sites: int = 3) -> CompetitionSpec:
    return CompetitionSpec(
        n_sites=n_sites,
        train_end=pd.Timestamp("2025-05-11 00:00", tz=KST),
        eval_start=pd.Timestamp("2025-05-11 01:00", tz=KST),
        eval_end=pd.Timestamp("2025-05-13 00:00", tz=KST),
        n_eval_hours=48,
    )


def _env(tmp_path: Path, spec: CompetitionSpec, cfg: FeatureConfig) -> DataPaths:
    canonical = tmp_path / "canonical"
    derived = tmp_path / "derived"
    canonical.mkdir(parents=True, exist_ok=True)
    derived.mkdir(parents=True, exist_ok=True)
    sites = pd.DataFrame(
        {
            "site_id": pd.Series([f"PV{i:03d}" for i in range(1, spec.n_sites + 1)], dtype=ST),
            "current_group": pd.Series(["A"] * spec.n_sites, dtype=ST),
            "capacity_kw": pd.Series([100.0] * spec.n_sites, dtype="float32"),
            "lat": [36.0 + i * 0.1 for i in range(spec.n_sites)],
            "lon": [127.0 + i * 0.1 for i in range(spec.n_sites)],
            "wa_station": pd.Series(["WA001"] * spec.n_sites, dtype=ST),
            "wb_station": pd.Series(["WB001"] * spec.n_sites, dtype=ST),
        }
    )
    ts_all = pd.Series(pd.date_range("2025-05-09 01:00", periods=96, freq="h", tz=KST), dtype=DT)
    train_ts = ts_all[ts_all <= spec.train_end]
    eval_ts = ts_all[ts_all > spec.train_end]
    generation = pd.DataFrame(
        {
            "site_id": pd.Series(np.repeat(sites["site_id"].to_numpy(), len(train_ts)), dtype=ST),
            "ts": pd.Series(np.tile(train_ts.to_numpy(), len(sites)), dtype=DT),
            "energy": pd.Series(np.full(len(sites) * len(train_ts), 5.0), dtype="float32"),
        }
    )
    generation["op_date"] = op_date(generation["ts"])
    smp = pd.DataFrame(
        {
            "ts": ts_all,
            "smp": pd.Series(np.full(len(ts_all), 100.0), dtype="float32"),
            "k": pd.Series(np.ones(len(ts_all)), dtype="int8"),
        }
    )
    grid = pd.DataFrame(
        {
            "site_id": pd.Series(np.repeat(sites["site_id"].to_numpy(), len(ts_all)), dtype=ST),
            "ts": pd.Series(np.tile(ts_all.to_numpy(), len(sites)), dtype=DT),
            "capacity_kw": pd.Series(np.full(len(sites) * len(ts_all), 100.0), dtype="float32"),
            "wa_station": pd.Series(["WA001"] * len(sites) * len(ts_all), dtype=ST),
            "wb_station": pd.Series(["WB001"] * len(sites) * len(ts_all), dtype=ST),
        }
    )
    grid["op_date"] = op_date(grid["ts"])
    grid["period"] = pd.Series(np.where(grid["ts"] <= spec.train_end, "train", "eval"), dtype=ST)
    solar = pd.DataFrame(
        {
            "site_id": grid["site_id"].copy(),
            "ts": grid["ts"].copy(),
            "elev": pd.Series(np.full(len(grid), 45.0), dtype="float32"),
            "azim": pd.Series(np.full(len(grid), 180.0), dtype="float32"),
            "cosz": pd.Series(np.full(len(grid), 0.7), dtype="float32"),
        }
    )
    sites.to_parquet(canonical / "sites.parquet", index=False)
    generation.to_parquet(canonical / "generation.parquet", index=False)
    smp.to_parquet(canonical / "smp.parquet", index=False)
    grid.to_parquet(derived / "grid.parquet", index=False)
    solar.to_parquet(derived / "solar.parquet", index=False)
    for provider, station, variables in (
        ("wa", "wa_station", cfg.wa_vars),
        ("wb", "wb_station", cfg.wb_vars),
    ):
        st_name = "WA001" if provider == "wa" else "WB001"
        recs = [(st_name, r, t) for r in spec.rounds for t in ts_all]
        fc = pd.DataFrame(recs, columns=[station, "round", "ts"])
        fc["round"] = fc["round"].astype("int8")
        fc["ts"] = fc["ts"].astype(DT)
        fc["op_date"] = op_date(fc["ts"])
        fc["issued_at"] = issued_at(fc["op_date"], fc["round"], spec)
        fc["fc_source"] = pd.Series(["orig"] * len(fc), dtype=ST)
        hours = fc["ts"].dt.hour.to_numpy(dtype="float64")
        for i, var in enumerate(variables):
            fc[var] = ((hours + i * 7.0 + fc["round"].to_numpy(dtype="float64")) % 100).astype("float32")
        fc.to_parquet(derived / f"forecast_filled_{provider}.parquet", index=False)
    return DataPaths(canonical_dir=canonical, derived_dir=derived, features_dir=tmp_path / "features")


def test_join_preserves_rows_with_nan_for_missing_forecast(tmp_path: Path) -> None:
    spec, cfg = _spec(), FeatureConfig()
    paths = _env(tmp_path, spec, cfg)
    filled = pd.read_parquet(paths.derived_dir / "forecast_filled_wa.parquet")
    missing_ts = pd.Timestamp("2025-05-09 12:00", tz=KST)
    filled = filled[filled["ts"] != missing_ts].reset_index(drop=True)
    filled.to_parquet(paths.derived_dir / "forecast_filled_wa.parquet", index=False)
    panel = build_panel(paths, spec, cfg)
    n_train_ts = 48
    assert len(panel) == spec.n_sites * (n_train_ts + spec.n_eval_hours)
    hit = panel[panel["ts"] == missing_ts]
    assert len(hit) == spec.n_sites
    assert hit["wa_cloud_r17"].isna().all()


def test_night_rows_excluded(tmp_path: Path) -> None:
    spec, cfg = _spec(), FeatureConfig()
    paths = _env(tmp_path, spec, cfg)
    solar = pd.read_parquet(paths.derived_dir / "solar.parquet")
    solar.loc[solar["ts"].dt.hour == 12, "elev"] = -10.0
    solar.to_parquet(paths.derived_dir / "solar.parquet", index=False)
    panel = build_panel(paths, spec, cfg)
    assert (panel["elev"] > cfg.day_min_elev_deg).all()


def test_feature_columns_bundle_toggle(tmp_path: Path) -> None:
    spec = _spec()
    paths = _env(tmp_path, spec, FeatureConfig())
    panel = build_panel(paths, spec, FeatureConfig())
    cols = feature_columns(panel, FeatureConfig(groups=frozenset({"rd"})))
    assert any(c.endswith("_rdiff") for c in cols)
    assert not any(c.endswith("_fleet") or c.endswith("_daymean") for c in cols)
    assert "cs_ghi" not in cols
    for banned in ("smp", "k", "energy", "site_id"):
        assert banned not in cols


def test_feature_columns_all_groups(tmp_path: Path) -> None:
    spec = _spec()
    paths = _env(tmp_path, spec, FeatureConfig())
    panel = build_panel(paths, spec, FeatureConfig())
    cols = feature_columns(panel, FeatureConfig())
    assert "cs_ghi" in cols
    assert any(c.endswith("_fleet") for c in cols)
    assert any(c.endswith("_daymean") for c in cols)
    assert any(c.endswith("_rdiff") for c in cols)
    with pytest.raises(PanelError):
        feature_columns(panel.drop(columns=["cs_ghi"]), FeatureConfig())


def test_eval_only_nan_rejected(tmp_path: Path) -> None:
    spec, cfg = _spec(), FeatureConfig()
    paths = _env(tmp_path, spec, cfg)
    filled = pd.read_parquet(paths.derived_dir / "forecast_filled_wa.parquet")
    missing_ts = pd.Timestamp("2025-05-12 12:00", tz=KST)
    filled = filled[filled["ts"] != missing_ts].reset_index(drop=True)
    filled.to_parquet(paths.derived_dir / "forecast_filled_wa.parquet", index=False)
    with pytest.raises(PanelError):
        build_panel(paths, spec, cfg)


def test_eval_grid_gap_rejected(tmp_path: Path) -> None:
    spec, cfg = _spec(), FeatureConfig()
    paths = _env(tmp_path, spec, cfg)
    grid = pd.read_parquet(paths.derived_dir / "grid.parquet")
    drop = (grid["site_id"] == "PV001") & (grid["ts"] == pd.Timestamp("2025-05-12 12:00", tz=KST))
    grid = grid[~drop].reset_index(drop=True)
    grid.to_parquet(paths.derived_dir / "grid.parquet", index=False)
    with pytest.raises(PanelError):
        build_panel(paths, spec, cfg)


def test_eval_grid_shift_rejected(tmp_path: Path) -> None:
    spec, cfg = _spec(), FeatureConfig()
    paths = _env(tmp_path, spec, cfg)
    grid = pd.read_parquet(paths.derived_dir / "grid.parquet")
    sel = (grid["site_id"] == "PV001") & (grid["ts"] == pd.Timestamp("2025-05-12 12:00", tz=KST))
    grid.loc[sel, "ts"] = pd.Timestamp("2025-05-12 13:00", tz=KST)
    grid.to_parquet(paths.derived_dir / "grid.parquet", index=False)
    with pytest.raises(PanelError):
        build_panel(paths, spec, cfg)


def test_contract_violations_rejected(tmp_path: Path) -> None:
    spec, cfg = _spec(), FeatureConfig()
    paths = _env(tmp_path, spec, cfg)
    sites = pd.read_parquet(paths.canonical_dir / "sites.parquet")
    sites.iloc[[0]].to_parquet(paths.canonical_dir / "sites.parquet", index=False)
    with pytest.raises(PanelError):
        build_panel(paths, spec, cfg)

    paths = _env(tmp_path, spec, cfg)
    solar = pd.read_parquet(paths.derived_dir / "solar.parquet")
    solar = solar.iloc[1:].reset_index(drop=True)
    solar.to_parquet(paths.derived_dir / "solar.parquet", index=False)
    with pytest.raises(PanelError):
        build_panel(paths, spec, cfg)

    paths = _env(tmp_path, spec, cfg)
    solar = pd.read_parquet(paths.derived_dir / "solar.parquet")
    solar = solar[solar["ts"] != pd.Timestamp("2025-05-12 12:00", tz=KST)].reset_index(drop=True)
    solar.to_parquet(paths.derived_dir / "solar.parquet", index=False)
    with pytest.raises(PanelError):
        build_panel(paths, spec, cfg)

    paths = _env(tmp_path, spec, cfg)
    smp = pd.read_parquet(paths.canonical_dir / "smp.parquet")
    smp = pd.concat([smp, smp.iloc[[0]]], ignore_index=True)
    smp.to_parquet(paths.canonical_dir / "smp.parquet", index=False)
    with pytest.raises(PanelError):
        build_panel(paths, spec, cfg)

    paths = _env(tmp_path, spec, cfg)
    smp = pd.read_parquet(paths.canonical_dir / "smp.parquet")
    smp = smp.iloc[1:].reset_index(drop=True)
    smp.to_parquet(paths.canonical_dir / "smp.parquet", index=False)
    with pytest.raises(PanelError):
        build_panel(paths, spec, cfg)

    paths = _env(tmp_path, spec, cfg)
    grid = pd.read_parquet(paths.derived_dir / "grid.parquet")
    grid = grid[grid["site_id"] != "PV003"].reset_index(drop=True)
    grid.to_parquet(paths.derived_dir / "grid.parquet", index=False)
    with pytest.raises(PanelError):
        build_panel(paths, spec, cfg)


def test_build_panel_deterministic(tmp_path: Path) -> None:
    spec, cfg = _spec(), FeatureConfig()
    paths = _env(tmp_path, spec, cfg)
    first = build_panel(paths, spec, cfg)
    second = build_panel(paths, spec, cfg)
    assert first.equals(second)


def test_build_features_writes_panel(tmp_path: Path) -> None:
    spec, cfg = _spec(), FeatureConfig()
    paths = _env(tmp_path, spec, cfg)
    out = build_features(paths, spec, cfg)
    assert out == paths.features_dir / "panel.parquet"
    assert out.exists()


@pytest.mark.parametrize("kind", [
    "eval_labels", "unknown_site", "op_date", "duplicate_forecast", "dark_site", "metadata", "joint_gap",
])
def test_panel_rejects_corrupt_contracts(tmp_path: Path, kind: str) -> None:
    spec, cfg = _spec(), FeatureConfig()
    paths = _env(tmp_path, spec, cfg)
    if kind in {"eval_labels", "unknown_site", "op_date"}:
        path = paths.canonical_dir / "generation.parquet"
        frame = pd.read_parquet(path)
        if kind == "eval_labels":
            frame.loc[0, "ts"] = spec.eval_start
            frame["op_date"] = op_date(frame["ts"])
        elif kind == "unknown_site":
            frame.loc[0, "site_id"] = "UNKNOWN"
        else:
            frame.loc[0, "op_date"] = frame.loc[0, "op_date"] + pd.Timedelta(days=1)
        frame.to_parquet(path, index=False)
    elif kind == "duplicate_forecast":
        path = paths.derived_dir / "forecast_filled_wa.parquet"
        frame = pd.read_parquet(path)
        pd.concat([frame, frame.iloc[[0]]], ignore_index=True).to_parquet(path, index=False)
    elif kind == "dark_site":
        path = paths.derived_dir / "solar.parquet"
        frame = pd.read_parquet(path)
        frame.loc[(frame["site_id"] == "PV003") & (frame["ts"] > spec.train_end), "elev"] = -10.0
        frame.to_parquet(path, index=False)
    elif kind == "metadata":
        path = paths.canonical_dir / "sites.parquet"
        frame = pd.read_parquet(path)
        frame.loc[0, "lat"] = np.nan
        frame.to_parquet(path, index=False)
    else:
        for name in ("grid", "solar"):
            path = paths.derived_dir / f"{name}.parquet"
            frame = pd.read_parquet(path)
            frame = frame[~((frame["site_id"] == "PV001") & (frame["ts"] == spec.eval_start))]
            frame.to_parquet(path, index=False)
    with pytest.raises(PanelError):
        build_panel(paths, spec, cfg)


@pytest.mark.parametrize("layer", ["raw_dir", "canonical_dir", "derived_dir"])
def test_features_output_cannot_alias_inputs(tmp_path: Path, layer: str) -> None:
    spec, cfg = _spec(), FeatureConfig()
    paths = _env(tmp_path, spec, cfg)
    protected = getattr(paths, layer)
    bad_paths = paths.model_copy(update={"features_dir": protected})
    with pytest.raises(PanelError, match="overwrite"):
        build_features(bad_paths, spec, cfg)


def test_base_only_panel_still_enforces_issuance_cutoff(tmp_path: Path) -> None:
    from src.oibc.data.gate import LookAheadError

    cfg = FeatureConfig(
        groups=frozenset(), neighbor_vars=(), rd_vars=(), day_mean_vars=(), day_max_vars=(), day_sum_vars=(),
        fleet_vars=(), vsfleet_vars=(),
    )
    spec = _spec()
    paths = _env(tmp_path, spec, cfg)
    path = paths.derived_dir / "forecast_filled_wa.parquet"
    frame = pd.read_parquet(path)
    frame["issued_at"] = frame["issued_at"] + pd.Timedelta(days=1)
    frame.to_parquet(path, index=False)
    with pytest.raises(LookAheadError):
        build_panel(paths, spec, cfg)


def test_eval_energy_cannot_enter_panel_from_grid(tmp_path: Path) -> None:
    spec, cfg = _spec(), FeatureConfig()
    paths = _env(tmp_path, spec, cfg)
    path = paths.derived_dir / "grid.parquet"
    grid = pd.read_parquet(path)
    grid["energy"] = np.float32(999.0)
    grid.to_parquet(path, index=False)
    panel = build_panel(paths, spec, cfg)
    assert panel.loc[panel["period"] == "eval", "energy"].isna().all()
    assert panel.loc[panel["period"] == "train", "energy"].eq(5.0).all()
