"""실행기 불변식 가드."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.oibc.config import (
    ArtifactPaths,
    CompetitionSpec,
    DataPaths,
    FeatureConfig,
    FoldSpec,
    ModelConfig,
    RunConfig,
    ValidationConfig,
)
from src.oibc.data.schemas import _DT
from src.oibc.evaluate.compare import compare_runs
from src.oibc.timeaxis import op_date
from src.oibc.train import runner as runner_mod
from src.oibc.train.runner import (
    RunnerError,
    forecast_eval,
    load_daily_runs,
    load_oof,
    reference_assignment,
    run_validation,
    site_frame,
)

KST = "Asia/Seoul"
SITES = ["PV001", "PV002", "PV003", "PV004"]
GROUPS = {"PV001": "A", "PV002": "A", "PV003": "B", "PV004": "B"}


def _spec() -> CompetitionSpec:
    return CompetitionSpec(
        n_sites=4, min_group_capacity_kw=200.0,
        train_end=pd.Timestamp("2025-06-01 00:00", tz=KST),
        eval_start=pd.Timestamp("2025-06-01 01:00", tz=KST),
        eval_end=pd.Timestamp("2025-06-02 00:00", tz=KST), n_eval_hours=24,
    )


@pytest.fixture(autouse=True)
def isolated_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> None:
    if request.node.get_closest_marker("slow"):
        return
    data = DataPaths(canonical_dir=tmp_path / "canonical", derived_dir=tmp_path / "derived", features_dir=tmp_path)
    data.canonical_dir.mkdir()
    data.derived_dir.mkdir()
    monkeypatch.setattr(runner_mod, "DataPaths", lambda: data)
    monkeypatch.setattr(runner_mod, "CompetitionSpec", _spec)


def _vcfg() -> ValidationConfig:
    folds = (
        FoldSpec(
            name="A",
            test_start=pd.Timestamp("2023-01-10", tz=KST),
            test_end=pd.Timestamp("2023-01-14", tz=KST),
        ),
        FoldSpec(
            name="B",
            test_start=pd.Timestamp("2023-01-15", tz=KST),
            test_end=pd.Timestamp("2023-01-19", tz=KST),
        ),
    )
    return ValidationConfig(
        folds=folds, embargo_days=1, min_warm_days=1, seeds=(0, 1), bootstrap_n=100
    )


def _feat() -> FeatureConfig:
    return FeatureConfig(
        groups=frozenset(),
        neighbor_vars=(),
        rd_vars=(),
        day_mean_vars=(),
        day_max_vars=(),
        day_sum_vars=(),
        fleet_vars=(),
        vsfleet_vars=(),
        wa_vars=("wa_cloud",),
        wb_vars=("wb_cloud", "wb_uv_idx"),
    )


def _cfg(run_id: str, **over: object) -> RunConfig:
    base: dict[str, object] = {
        "run_id": run_id,
        "hypothesis": "synthetic guard",
        "features": _feat(),
        "model": ModelConfig(n_rounds=3, num_leaves=4, n_jobs=1, min_child_samples=5, seeds=(0, 1)),
    }
    base.update(over)
    return RunConfig(**base)  # type: ignore[arg-type]


def _paths(tmp_path: Path) -> ArtifactPaths:
    return ArtifactPaths(
        root=tmp_path,
        registry=tmp_path / "registry.jsonl",
        submissions=tmp_path / "subs",
        runs=tmp_path / "runs",
    )


def _make_panel(tmp_path: Path) -> Path:
    feat = _feat()
    extra = [f"{v}_r{r}" for v in (*feat.wa_vars, *feat.wb_vars) for r in (10, 17)]
    rows: list[dict[str, object]] = []
    train_days = pd.date_range("2023-01-01", "2023-01-25", freq="D", tz=KST)
    eval_stamps = [
        pd.Timestamp("2025-06-01 11:00", tz=KST),
        pd.Timestamp("2025-06-01 12:00", tz=KST),
        pd.Timestamp("2025-06-01 13:00", tz=KST),
    ]
    for day in train_days:
        for hour in (11, 12, 13):
            ts = day + pd.Timedelta(hours=hour)
            for idx, site in enumerate(SITES):
                row: dict[str, object] = {
                        "site_id": site,
                        "ts": ts,
                        "period": "train",
                        "energy": np.float32(40.0 + hour + idx * 3.0),
                        "current_group": GROUPS[site],
                        "smp": 100.0,
                        "k": 3,
                        "elev": np.float32(30.0),
                        "azim": np.float32(180.0),
                        "cosz": np.float32(0.5),
                        "doy_sin": np.float32(0.0),
                        "doy_cos": np.float32(1.0),
                        "hour": np.int8(hour),
                        "capacity_kw": np.float32(100.0),
                        "lat": 36.0 + idx / 100.0,
                        "lon": 129.0,
                    }
                for col in extra:
                    row[col] = np.float32(0.5)
                rows.append(row)  # noqa: PERF401
    for ts in eval_stamps:
        for site in SITES:
            row = {
                        "site_id": site,
                        "ts": ts,
                        "period": "eval",
                        "energy": np.float32(np.nan),
                        "current_group": GROUPS[site],
                        "smp": 100.0,
                        "k": 3,
                        "elev": np.float32(30.0),
                        "azim": np.float32(180.0),
                        "cosz": np.float32(0.5),
                        "doy_sin": np.float32(0.0),
                        "doy_cos": np.float32(1.0),
                        "hour": np.int8(ts.hour),
                        "capacity_kw": np.float32(100.0),
                        "lat": 36.0 + SITES.index(site) / 100.0,
                        "lon": 129.0,
                    }
            for col in extra:
                row[col] = np.float32(0.5)
            rows.append(row)  # noqa: PERF401
    panel = pd.DataFrame(rows)
    panel["site_id"] = panel["site_id"].astype("string")
    panel["ts"] = pd.Series(pd.DatetimeIndex(panel["ts"]), dtype=_DT)
    panel["op_date"] = op_date(panel["ts"]).astype(_DT)
    panel["period"] = panel["period"].astype("string")
    panel["current_group"] = panel["current_group"].astype("string")
    path = tmp_path / "panel.parquet"
    panel.to_parquet(path, index=False)
    canonical = tmp_path / "canonical"
    derived = tmp_path / "derived"
    sites = pd.DataFrame({"site_id": SITES, "current_group": [GROUPS[s] for s in SITES], "capacity_kw": [100.0] * 4})
    sites.to_parquet(canonical / "sites.parquet", index=False)
    generation = panel.loc[panel["period"] == "train", ["site_id", "ts", "op_date", "energy"]].copy()
    night_ts = pd.DatetimeIndex([day + pd.Timedelta(hours=1) for day in train_days]).as_unit("us")
    night = pd.MultiIndex.from_product([SITES, night_ts], names=["site_id", "ts"]).to_frame(index=False)
    night["op_date"] = op_date(night["ts"])
    night["energy"] = 2.0
    generation = pd.concat([generation, night], ignore_index=True)
    generation.to_parquet(canonical / "generation.parquet", index=False)
    eval_grid = pd.MultiIndex.from_product(
        [SITES, pd.date_range(_spec().eval_start, _spec().eval_end, freq="h").as_unit("us")], names=["site_id", "ts"]
    ).to_frame(index=False)
    eval_grid["period"] = "eval"
    eval_grid.to_parquet(derived / "grid.parquet", index=False)
    solar = pd.concat([generation[["site_id", "ts"]], eval_grid[["site_id", "ts"]]], ignore_index=True)
    solar["elev"] = np.where(solar["ts"].dt.hour.isin([11, 12, 13]), 30.0, -10.0)
    solar.to_parquet(derived / "solar.parquet", index=False)
    stamps = pd.DatetimeIndex(solar["ts"].unique()).sort_values()
    smp = pd.DataFrame({"ts": stamps, "smp": 100.0})
    smp["k"] = np.where(smp["ts"].dt.hour.isin([11, 12, 13, 14, 15]), 3, 1)
    smp["period"] = np.where(smp["ts"] <= _spec().train_end, "train", "eval")
    smp.to_parquet(canonical / "smp.parquet", index=False)
    return path


def _day_night() -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    ts = pd.DatetimeIndex(
        ["2023-01-10 11:00", "2023-01-10 12:00", "2023-01-10 13:00"], tz=KST
    ).as_unit("us")
    day = pd.DataFrame({"site_id": ["PV001"] * 3, "ts": ts, "energy": [10.0, 20.0, 30.0]})
    night_ts = pd.DatetimeIndex(["2023-01-10 00:00", "2023-01-10 01:00"], tz=KST).as_unit("us")
    night = pd.DataFrame({"site_id": ["PV001"] * 2, "ts": night_ts, "energy": [5.0, 0.0]})
    return day, np.array([9.0, 19.0, 29.0]), night


def test_reference_assignment() -> None:
    sites = pd.DataFrame({"site_id": ["a", "b"], "current_group": ["A", "B"]})
    out = reference_assignment(sites)
    assert out.set_index("site_id")["group"].to_dict() == {"a": "G1", "b": "G2"}
    with pytest.raises(RunnerError):
        reference_assignment(pd.DataFrame({"site_id": ["a"], "current_group": ["C"]}))


def test_site_frame_night_zero_and_preserved() -> None:
    day, pred, night = _day_night()
    out = site_frame(day, pred, night)
    assert len(out) == 5
    assert bool((out.loc[out["ts"].dt.hour.isin([0, 1]), "pred_kwh"] == 0.0).all())
    assert out["actual_kwh"].sum() == pytest.approx(65.0)


def test_site_frame_overlap_rejected() -> None:
    day, pred, _ = _day_night()
    dup = day.iloc[:1].copy()
    with pytest.raises(RunnerError):
        site_frame(day, pred, dup)


def test_run_validation_artifacts(tmp_path: Path) -> None:
    panel = _make_panel(tmp_path)
    paths = _paths(tmp_path)
    vcfg = _vcfg()
    rec = run_validation(_cfg("r1"), panel_path=panel, vcfg=vcfg, paths=paths)
    assert rec.run_id == "r1"
    for fold in ("A", "B"):
        for seed in (0, 1):
            for subset in ("all", "warm"):
                assert (paths.runs / "r1" / f"{fold}_s{seed}_{subset}.parquet").exists()
        for subset in ("all", "warm"):
            assert (paths.runs / "r1" / f"{fold}_bag_{subset}.parquet").exists()
    assert set(rec.metrics) == {f"{f}/{s}" for f in ("A", "B") for s in (0, 1)}


def test_run_overwrite_rejected(tmp_path: Path) -> None:
    panel = _make_panel(tmp_path)
    paths = _paths(tmp_path)
    vcfg = _vcfg()
    run_validation(_cfg("r1"), panel_path=panel, vcfg=vcfg, paths=paths)
    from src.oibc.evaluate.registry import read_records

    n = len(read_records(paths.registry))
    with pytest.raises(RunnerError):
        run_validation(_cfg("r1"), panel_path=panel, vcfg=vcfg, paths=paths)
    assert len(read_records(paths.registry)) == n


def test_baseline_missing_fail_closed(tmp_path: Path) -> None:
    panel = _make_panel(tmp_path)
    with pytest.raises(RunnerError):
        run_validation(
            _cfg("r2", baseline_run_id="nope"), panel_path=panel, vcfg=_vcfg(), paths=_paths(tmp_path)
        )


def test_seed_mismatch_rejected(tmp_path: Path) -> None:
    panel = _make_panel(tmp_path)
    vcfg = _vcfg()
    bad = RunConfig(
        run_id="r3",
        hypothesis="h",
        features=_feat(),
        model=ModelConfig(n_rounds=3, num_leaves=4, n_jobs=1, min_child_samples=5, seeds=(0, 1, 2)),
    )
    with pytest.raises(RunnerError):
        run_validation(bad, panel_path=panel, vcfg=vcfg, paths=_paths(tmp_path))


def test_daily_runs_roundtrip_compare(tmp_path: Path) -> None:
    panel = _make_panel(tmp_path)
    paths = _paths(tmp_path)
    vcfg = _vcfg()
    run_validation(_cfg("a"), panel_path=panel, vcfg=vcfg, paths=paths)
    run_validation(_cfg("b"), panel_path=panel, vcfg=vcfg, paths=paths)
    cand = load_daily_runs("a", "all", paths=paths, vcfg=vcfg)
    base = load_daily_runs("b", "all", paths=paths, vcfg=vcfg)
    cmp = compare_runs(cand, base, vcfg)
    assert {f.fold for f in cmp.folds} == {"A", "B"}


def test_forecast_eval_format(tmp_path: Path) -> None:
    panel = _make_panel(tmp_path)
    out = forecast_eval(_cfg("r1"), panel_path=panel, spec=_spec())
    assert set(out["group"].tolist()) == {"G1", "G2"}
    assert (out["forecast_kwh"] >= 0).all() and np.isfinite(out["forecast_kwh"].to_numpy()).all()
    assert str(out["forecast_kwh"].dtype) == "float64"
    assert str(out["ts"].dtype) == _DT


def test_forecast_eval_uses_panel_eval(tmp_path: Path) -> None:
    panel = _make_panel(tmp_path)
    out = forecast_eval(_cfg("r9"), panel_path=panel, spec=_spec())
    assert len(out) == 2 * _spec().n_eval_hours
    night = out.loc[out["ts"] == pd.Timestamp("2025-06-01 01:00", tz=KST)]
    assert len(night) == 2 and bool((night["forecast_kwh"] == 0.0).all())
    assert out["forecast_kwh"].sum() > 0.0


def test_no_leakage_from_future_actuals(tmp_path: Path) -> None:
    panel_path = _make_panel(tmp_path)
    paths = _paths(tmp_path)
    vcfg = _vcfg()
    run_validation(_cfg("base"), panel_path=panel_path, vcfg=vcfg, paths=paths)
    before = load_daily_runs("base", "all", paths=paths, vcfg=vcfg)["A"][0]
    panel = pd.read_parquet(panel_path)
    future = (panel["period"] == "train") & (panel["op_date"] >= pd.Timestamp("2023-01-15", tz=KST))
    panel.loc[future, "energy"] = panel.loc[future, "energy"] * 100.0
    tampered = tmp_path / "panel_tampered.parquet"
    panel.to_parquet(tampered, index=False)
    run_validation(_cfg("tampered"), panel_path=tampered, vcfg=vcfg, paths=paths)
    after = load_daily_runs("tampered", "all", paths=paths, vcfg=vcfg)["A"][0]
    pd.testing.assert_frame_equal(before, after)


@pytest.mark.slow
def test_real_panel_seed_deterministic() -> None:
    from src.oibc.cv.splits import fold_masks
    from src.oibc.evaluate.settlement import aggregate_groups, daily_totals, settle
    from src.oibc.features.panel import feature_columns
    from src.oibc.models.gbm import fit_seed_models, predict_seed_kwh

    panel_path = DataPaths().features_dir / "panel.parquet"
    if not panel_path.exists():
        pytest.skip("실데이터 패널 없음")
    panel = pd.read_parquet(panel_path)
    panel = panel.loc[panel["period"] == "train"]
    features = feature_columns(panel, FeatureConfig())
    assignment, _ = runner_mod._resolve_assignment(panel)
    smp = runner_mod._resolve_smp(panel)
    default = ValidationConfig()
    vcfg = ValidationConfig(folds=(default.folds[1],), seeds=(0,), bootstrap_n=100)
    fold = vcfg.folds[0]
    tr, te = fold_masks(panel["op_date"], fold, vcfg)
    train, test = panel.loc[tr], panel.loc[te]
    model_cfg = ModelConfig(n_rounds=50, seeds=(0,))
    first: list[float] = []
    second: list[float] = []
    for store in (first, second):
        models = fit_seed_models(train, features, model_cfg)
        preds = predict_seed_kwh(models, test, features)
        frame = site_frame(test, preds[0], test.iloc[:0])
        daily = daily_totals(settle(aggregate_groups(frame, assignment), smp))
        store.extend(daily["penalty"].tolist())
        del models, preds
    assert first == second
    zero_frame = site_frame(test, np.zeros(len(test), dtype="float64"), test.iloc[:0])
    zero_w = float(daily_totals(settle(aggregate_groups(zero_frame, assignment), smp))["penalty"].sum())
    assert sum(first) < zero_w


def test_run_id_rule_rejected() -> None:
    with pytest.raises(ValueError, match="run_id"):
        _cfg("bad id!")
    with pytest.raises(ValueError, match="hypothesis"):
        _cfg("blank", hypothesis=" ")
    for value in (".", "..", "../outside"):
        with pytest.raises(ValueError, match="run_id"):
            _cfg(value)
        with pytest.raises(ValueError, match="run_id"):
            _cfg("safe", baseline_run_id=value)


def test_reference_assignment_guards() -> None:
    with pytest.raises(RunnerError):
        reference_assignment(pd.DataFrame({"site_id": ["a"]}))
    with pytest.raises(RunnerError):
        reference_assignment(pd.DataFrame({"site_id": [], "current_group": []}))
    with pytest.raises(RunnerError):
        reference_assignment(pd.DataFrame({"site_id": ["a", "a"], "current_group": ["A", "A"]}))
    with pytest.raises(RunnerError):
        reference_assignment(pd.DataFrame({"site_id": [None], "current_group": ["A"]}))


def test_site_frame_guards() -> None:
    day, pred, night = _day_night()
    with pytest.raises(RunnerError):
        site_frame(day, np.array([1.0, 2.0]), night)
    with pytest.raises(RunnerError):
        site_frame(day, np.array([1.0, -1.0, 3.0]), night)
    with pytest.raises(RunnerError):
        site_frame(day, np.array([1.0, np.inf, 3.0]), night)
    with pytest.raises(RunnerError):
        site_frame(day.drop(columns=["ts"]), pred, night)
    with pytest.raises(RunnerError):
        site_frame(
            pd.concat([day, day.iloc[:1]], ignore_index=True),
            np.array([1.0, 2.0, 3.0, 4.0]),
            night,
        )
    bad_ts = day.copy()
    bad_ts["ts"] = pd.Series(
        pd.DatetimeIndex(["2023-01-10 11:30", "2023-01-10 12:30", "2023-01-10 13:30"], tz=KST)
    )
    with pytest.raises(RunnerError):
        site_frame(bad_ts, pred, night)
    bad_energy = day.copy()
    bad_energy["energy"] = [1.0, np.nan, 3.0]
    with pytest.raises(RunnerError):
        site_frame(bad_energy, pred, night)
    with pytest.raises(RunnerError):
        site_frame(day.drop(columns=["energy"]), pred, night.drop(columns=["energy"]))
    with pytest.raises(RunnerError):
        site_frame(day, ["x", "y", "z"], night)  # type: ignore[arg-type]
    renamed = day.rename(columns={"energy": "actual_kwh"})
    out = site_frame(renamed, pred, night.rename(columns={"energy": "actual_kwh"}))
    assert len(out) == 5


def test_load_daily_runs_guards(tmp_path: Path) -> None:
    panel = _make_panel(tmp_path)
    paths = _paths(tmp_path)
    vcfg = _vcfg()
    run_validation(_cfg("g"), panel_path=panel, vcfg=vcfg, paths=paths)
    with pytest.raises(RunnerError):
        load_daily_runs("g", "nope", paths=paths, vcfg=vcfg)  # type: ignore[arg-type]
    with pytest.raises(RunnerError):
        load_daily_runs("missing", "all", paths=paths, vcfg=vcfg)
    bagged = load_daily_runs("g", "all", bagged=True, paths=paths, vcfg=vcfg)
    assert set(bagged) == {"A", "B"} and all(len(v) == 1 for v in bagged.values())


def test_run_validation_guards(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    vcfg = _vcfg()
    with pytest.raises(RunnerError):
        run_validation(_cfg("x"), panel_path=tmp_path / "none.parquet", vcfg=vcfg, paths=paths)
    empty = tmp_path / "empty.parquet"
    pd.DataFrame({"period": [], "op_date": []}).to_parquet(empty, index=False)
    with pytest.raises(RunnerError):
        run_validation(_cfg("x"), panel_path=empty, vcfg=vcfg, paths=paths)
    garbage = tmp_path / "garbage.parquet"
    garbage.write_bytes(b"not a parquet file")
    with pytest.raises(RunnerError):
        run_validation(_cfg("x"), panel_path=garbage, vcfg=vcfg, paths=paths)
    panel = _make_panel(tmp_path)
    noperiod = pd.read_parquet(panel).drop(columns=["period"])
    noperiod_path = tmp_path / "noperiod.parquet"
    noperiod.to_parquet(noperiod_path, index=False)
    with pytest.raises(RunnerError):
        run_validation(_cfg("x"), panel_path=noperiod_path, vcfg=vcfg, paths=paths)


def test_run_validation_empty_test_fold(tmp_path: Path) -> None:
    panel = _make_panel(tmp_path)
    paths = _paths(tmp_path)
    far_folds = (
        FoldSpec(name="A", test_start=pd.Timestamp("2020-01-10", tz=KST), test_end=pd.Timestamp("2020-01-14", tz=KST)),
    )
    vcfg = ValidationConfig(folds=far_folds, embargo_days=1, min_warm_days=1, seeds=(0, 1), bootstrap_n=100)
    cfg = RunConfig(
        run_id="far",
        hypothesis="h",
        features=_feat(),
        model=ModelConfig(n_rounds=3, num_leaves=4, n_jobs=1, min_child_samples=5, seeds=(0, 1)),
    )
    from src.oibc.cv.splits import SplitError  # noqa: F401

    with pytest.raises(RunnerError):
        run_validation(cfg, panel_path=panel, vcfg=vcfg, paths=paths)


def test_run_validation_empty_warm_subset(tmp_path: Path) -> None:
    panel = _make_panel(tmp_path)
    paths = _paths(tmp_path)
    vcfg = ValidationConfig(
        folds=_vcfg().folds, embargo_days=1, min_warm_days=100000, seeds=(0, 1), bootstrap_n=100
    )
    with pytest.raises(RunnerError):
        run_validation(_cfg("nowarm"), panel_path=panel, vcfg=vcfg, paths=paths)


def test_run_validation_missing_smp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.oibc.train import runner as runner_mod

    panel = _make_panel(tmp_path)
    paths = _paths(tmp_path)
    monkeypatch.setattr(runner_mod, "_resolve_smp", lambda *args: pd.DataFrame({"ts": [], "smp": [], "k": []}))
    with pytest.raises(RunnerError):
        run_validation(_cfg("nosmp"), panel_path=panel, vcfg=_vcfg(), paths=paths)


def test_run_validation_with_baseline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    panel = _make_panel(tmp_path)
    paths = _paths(tmp_path)
    vcfg = _vcfg()
    monkeypatch.setattr(runner_mod, "_git_tree_dirty", lambda: False)
    monkeypatch.setattr(runner_mod, "_git_commit", lambda: "aaa")
    run_validation(_cfg("base0"), panel_path=panel, vcfg=vcfg, paths=paths)
    rec = run_validation(_cfg("cand0", baseline_run_id="base0"), panel_path=panel, vcfg=vcfg, paths=paths)
    assert rec.baseline_run_id == "base0"
    assert "vs_baseline" in (rec.comparison or {})


def test_forecast_eval_guards(tmp_path: Path) -> None:
    with pytest.raises(RunnerError):
        forecast_eval(_cfg("x"), panel_path=tmp_path / "none.parquet")
    panel = _make_panel(tmp_path)
    train_only = pd.read_parquet(panel).query("period == 'train'")
    train_path = tmp_path / "train_only.parquet"
    train_only.to_parquet(train_path, index=False)
    with pytest.raises(RunnerError):
        forecast_eval(_cfg("x"), panel_path=train_path)
    noperiod = pd.read_parquet(panel).drop(columns=["period"])
    noperiod_path = tmp_path / "noperiod_fc.parquet"
    noperiod.to_parquet(noperiod_path, index=False)
    with pytest.raises(RunnerError):
        forecast_eval(_cfg("x"), panel_path=noperiod_path)


def test_forecast_eval_smp_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import pandas as pd

    from src.oibc.train import runner as runner_mod

    panel = _make_panel(tmp_path)
    real_read = pd.read_parquet

    def _selective(path: object, *a: object, **k: object) -> pd.DataFrame:
        if str(path).endswith("smp.parquet"):
            raise FileNotFoundError("gone")
        return real_read(path, *a, **k)  # type: ignore[arg-type]

    monkeypatch.setattr(pd, "read_parquet", _selective)
    with pytest.raises(RunnerError, match=r"smp\.parquet"):
        runner_mod.forecast_eval(_cfg("fb"), panel_path=panel)


def test_forecast_eval_smp_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import pandas as pd

    from src.oibc.train import runner as runner_mod

    panel = _make_panel(tmp_path)
    real_read = pd.read_parquet

    def _short_smp(path: object, *a: object, **k: object) -> pd.DataFrame:
        if str(path).endswith("smp.parquet"):
            return pd.DataFrame(
                {
                    "ts": pd.Series(pd.DatetimeIndex(["2025-06-01 01:00"], tz=KST)),
                    "smp": [100.0],
                    "k": [1],
                    "period": ["eval"],
                }
            )
        return real_read(path, *a, **k)  # type: ignore[arg-type]

    monkeypatch.setattr(pd, "read_parquet", _short_smp)
    with pytest.raises(RunnerError):
        runner_mod.forecast_eval(_cfg("mm"), panel_path=panel)


def test_forecast_eval_group_mismatch(tmp_path: Path) -> None:
    from src.oibc.config import CompetitionSpec

    panel = _make_panel(tmp_path)
    spec = _spec().model_copy(update={"groups": ("G1", "G3")})
    with pytest.raises(RunnerError):
        forecast_eval(_cfg("gm"), panel_path=panel, spec=spec)


def test_forecast_eval_postcondition_guards(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import numpy as np

    from src.oibc.train import runner as runner_mod

    panel = _make_panel(tmp_path)
    monkeypatch.setattr(runner_mod, "bag", lambda preds: np.full(preds.shape[1], np.inf))
    with pytest.raises(RunnerError):
        runner_mod.forecast_eval(_cfg("inf"), panel_path=panel)
    monkeypatch.setattr(runner_mod, "bag", lambda preds: np.full(preds.shape[1], np.nan))
    with pytest.raises(RunnerError):
        runner_mod.forecast_eval(_cfg("tot"), panel_path=panel)


def test_git_commit_none_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    from src.oibc.train import runner as runner_mod

    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(OSError("no git"))
    )
    assert runner_mod._git_commit() is None

    class _Bad:
        returncode = 1
        stdout = "abc"

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Bad())
    assert runner_mod._git_commit() is None
    assert runner_mod._default_panel_path().name == "panel.parquet"


def test_normalize_daily_branches() -> None:
    from src.oibc.train import runner as runner_mod

    idx = pd.DatetimeIndex(pd.date_range("2023-01-10", periods=3, freq="D", tz=KST), dtype=_DT)
    col_frame = pd.DataFrame({"penalty": [1.0, 2.0, 3.0], "revenue": [4.0, 5.0, 6.0], "op_date": idx})
    out = runner_mod._normalize_daily(col_frame, "x")
    assert isinstance(out.index, pd.DatetimeIndex)
    with pytest.raises(RunnerError):
        runner_mod._normalize_daily(pd.DataFrame({"penalty": [1.0]}, index=["bad"]), "x")


def test_night_frame_guards(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import pandas as pd

    from src.oibc.train import runner as runner_mod

    _make_panel(tmp_path)
    real_read = pd.read_parquet

    def _bad_gen(path: object, *a: object, **k: object) -> pd.DataFrame:
        if str(path).endswith("generation.parquet"):
            return pd.DataFrame({"site_id": ["s"], "ts": [pd.Timestamp("2023-01-01", tz=KST)]})
        return real_read(path, *a, **k)  # type: ignore[arg-type]

    monkeypatch.setattr(pd, "read_parquet", _bad_gen)
    with pytest.raises(RunnerError):
        runner_mod._night_frame(pd.Series([], dtype="object"), 0.0)


def test_forecast_preserves_row_prediction_pairing(tmp_path: Path) -> None:
    from src.oibc.features.panel import feature_columns
    from src.oibc.models.gbm import bag, fit_seed_models, predict_seed_kwh

    path = _make_panel(tmp_path)
    panel = pd.read_parquet(path)
    cfg = _cfg("aligned")
    features = feature_columns(panel, cfg.features)
    train = panel.loc[panel["period"] == "train"]
    day = panel.loc[panel["period"] == "eval"].copy()
    models = fit_seed_models(train, features, cfg.model)
    day["forecast_kwh"] = bag(predict_seed_kwh(models, day, features))
    assert day["forecast_kwh"].nunique() > 1
    day["group"] = day["current_group"].map({"A": "G1", "B": "G2"})
    expected = day.groupby(["group", "ts"])["forecast_kwh"].sum().sort_index()
    shuffled = pd.concat([train, panel.loc[panel["period"] == "eval"].sample(frac=1.0, random_state=37)], ignore_index=True)
    shuffled["ts"] = shuffled["ts"].astype("datetime64[ms, Asia/Seoul]")
    shuffled.to_parquet(path, index=False)
    actual = forecast_eval(cfg, panel_path=path).set_index(["group", "ts"])["forecast_kwh"].sort_index()
    np.testing.assert_allclose(actual.reindex(expected.index), expected, rtol=1e-12)
    assert actual.sum() == pytest.approx(expected.sum())
    assert (actual.loc[~actual.index.get_level_values("ts").isin(day["ts"])] == 0.0).all()


@pytest.mark.parametrize("defect", ["duplicate", "op_date", "eval_label", "future_train", "night", "missing_day"])
def test_panel_contract_rejected_before_training(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, defect: str) -> None:
    path = _make_panel(tmp_path)
    panel = pd.read_parquet(path)
    if defect == "duplicate":
        panel = pd.concat([panel, panel.iloc[:1]], ignore_index=True)
    elif defect == "op_date":
        panel.loc[0, "op_date"] += pd.Timedelta(days=1)
    elif defect == "eval_label":
        panel.loc[panel["period"] == "eval", "energy"] = 1.0
    elif defect == "future_train":
        panel.loc[panel["period"] == "eval", "period"] = "train"
        panel.loc[panel["energy"].isna(), "energy"] = 1.0
    elif defect == "night":
        panel.loc[0, "elev"] = -10.0
    else:
        panel = panel.drop(panel.index[panel["period"] == "eval"][0])
    panel.to_parquet(path, index=False)

    def no_fit(*args, **kwargs):
        pytest.fail("계약 위반 입력으로 학습 실행")

    monkeypatch.setattr(runner_mod, "fit_seed_models", no_fit)
    with pytest.raises(RunnerError):
        forecast_eval(_cfg("invalid"), panel_path=path)


@pytest.mark.parametrize(("file", "defect"), [
    ("grid", "missing"), ("solar", "missing"), ("solar", "duplicate"),
    ("smp", "price"), ("smp", "duplicate"), ("sites", "capacity"), ("sites", "group_drift"),
])
def test_auxiliary_contracts_fail_closed(tmp_path: Path, file: str, defect: str) -> None:
    path = _make_panel(tmp_path)
    folder = "derived" if file in {"grid", "solar"} else "canonical"
    target = tmp_path / folder / f"{file}.parquet"
    frame = pd.read_parquet(target)
    if defect == "missing":
        frame = frame.drop(frame.index[-1])
    elif defect == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
    elif defect == "price":
        frame.loc[0, "smp"] = np.nan
    elif defect == "group_drift":
        frame["current_group"] = frame["current_group"].map({"A": "B", "B": "A"})
    else:
        frame.loc[0, "capacity_kw"] = 1.0
    frame.to_parquet(target, index=False)
    with pytest.raises(RunnerError):
        forecast_eval(_cfg("bad-aux"), panel_path=path)


@pytest.mark.parametrize("defect", ["future", "solar_missing", "unknown_site", "missing_train_day"])
def test_night_observation_integrity(tmp_path: Path, defect: str) -> None:
    panel_path = _make_panel(tmp_path)
    gen_path = tmp_path / "canonical" / "generation.parquet"
    solar_path = tmp_path / "derived" / "solar.parquet"
    gen = pd.read_parquet(gen_path)
    solar = pd.read_parquet(solar_path)
    if defect == "future":
        future = gen.iloc[:1].copy()
        future["ts"] = _spec().eval_start
        future["op_date"] = op_date(future["ts"])
        gen = pd.concat([gen, future], ignore_index=True)
    elif defect == "solar_missing":
        solar = solar.iloc[1:]
    elif defect == "unknown_site":
        gen.loc[gen["ts"].dt.hour == 1, "site_id"] = "alien"
        solar.loc[solar["ts"].dt.hour == 1, "site_id"] = "alien"
        gen = gen.drop_duplicates(["site_id", "ts"])
        solar = solar.drop_duplicates(["site_id", "ts"])
    else:
        panel = pd.read_parquet(panel_path).iloc[1:]
        panel.to_parquet(panel_path, index=False)
    gen.to_parquet(gen_path, index=False)
    solar.to_parquet(solar_path, index=False)
    with pytest.raises(RunnerError):
        run_validation(_cfg("bad-night"), panel_path=panel_path, vcfg=_vcfg(), paths=_paths(tmp_path))


def test_day_night_overlap_across_timestamp_units() -> None:
    day, pred, _ = _day_night()
    night = day.iloc[:1].copy()
    night["ts"] = night["ts"].astype("datetime64[ms, Asia/Seoul]")
    with pytest.raises(RunnerError, match="overlap"):
        site_frame(day, pred, night)


def test_validation_night_revenue_and_registry_identity(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    import logging
    from src.oibc.evaluate.registry import read_records

    path = _make_panel(tmp_path)
    paths = _paths(tmp_path)
    with caplog.at_level(logging.INFO):
        rec = run_validation(_cfg("night-check"), panel_path=path, vcfg=_vcfg(), paths=paths)
    panel = pd.read_parquet(path)
    test_day = panel.loc[panel["op_date"].between(_vcfg().folds[0].test_start, _vcfg().folds[0].test_end)]
    assert rec.metrics["A/0"]["revenue_all"] == pytest.approx(float(test_day["energy"].sum()) * 100.0 + 5 * 4 * 2 * 100.0)
    log = next(r.message for r in caplog.records if "run=night-check fold=A seed=0" in r.message)
    assert f"W_warm={rec.metrics['A/0']['W_warm']!r}" in log
    assert read_records(paths.registry) == [rec]
    for subset in ("all", "warm"):
        for file in (paths.runs / rec.run_id).glob(f"*_{subset}.parquet"):
            daily = pd.read_parquet(file)
            assert len(daily) == 5
    for file in (paths.runs / rec.run_id).iterdir():
        file.unlink()
    (paths.runs / rec.run_id).rmdir()
    with pytest.raises(RunnerError, match="registry"):
        run_validation(_cfg(rec.run_id), panel_path=path, vcfg=_vcfg(), paths=paths)


def test_corrupt_baseline_rejected_before_training(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _make_panel(tmp_path)
    paths = _paths(tmp_path)
    baseline = paths.runs / "broken"
    baseline.mkdir(parents=True)
    idx = pd.DatetimeIndex(["2023-01-10"], tz=KST).as_unit("us")
    pd.DataFrame({"penalty": [-1.0], "revenue": [1.0]}, index=idx).to_parquet(baseline / "A_s0_all.parquet")
    with pytest.raises(RunnerError, match="invalid daily"):
        run_validation(_cfg("candidate", baseline_run_id="broken"), panel_path=path, vcfg=_vcfg(), paths=paths)
    for value in ("..", "../escape"):
        with pytest.raises(RunnerError, match="unsafe"):
            load_daily_runs(value, "all", paths=paths, vcfg=_vcfg())


def test_mutating_panel_during_fit_not_recorded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _make_panel(tmp_path)
    paths = _paths(tmp_path)
    real_fit = runner_mod.fit_seed_models

    def mutate(train, features, cfg):
        models = real_fit(train, features, cfg)
        panel = pd.read_parquet(path)
        panel["doy_sin"] = 0.4
        panel.to_parquet(path, index=False)
        return models

    monkeypatch.setattr(runner_mod, "fit_seed_models", mutate)
    with pytest.raises(RunnerError, match="changed during"):
        run_validation(_cfg("mutable"), panel_path=path, vcfg=_vcfg(), paths=paths)
    assert not paths.registry.exists()


def test_group_prediction_overflow_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _make_panel(tmp_path)
    monkeypatch.setattr(runner_mod, "bag", lambda preds: np.full(preds.shape[1], 1e308))
    with pytest.raises(RunnerError, match="overflow"):
        forecast_eval(_cfg("overflow"), panel_path=path)


def test_night_solar_schema_and_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _make_panel(tmp_path)
    real_read = pd.read_parquet

    def _bad_solar(path: object, *a: object, **k: object) -> pd.DataFrame:
        if str(path).endswith("generation.parquet"):
            return pd.DataFrame(
                {
                    "site_id": ["s"],
                    "ts": [pd.Timestamp("2023-01-10 11:00", tz=KST)],
                    "energy": [1.0],
                    "op_date": [pd.Timestamp("2023-01-10", tz=KST)],
                }
            )
        if str(path).endswith("solar.parquet"):
            return pd.DataFrame({"site_id": ["s"]})
        return real_read(path, *a, **k)  # type: ignore[arg-type]

    monkeypatch.setattr(pd, "read_parquet", _bad_solar)
    with pytest.raises(RunnerError):
        runner_mod._night_frame(pd.Series([], dtype="object"), 0.0)

    def _high_elev(path: object, *a: object, **k: object) -> pd.DataFrame:
        if str(path).endswith("generation.parquet"):
            return pd.DataFrame(
                {
                    "site_id": ["s"],
                    "ts": [pd.Timestamp("2023-01-10 11:00", tz=KST)],
                    "energy": [1.0],
                    "op_date": [pd.Timestamp("2023-01-10", tz=KST)],
                }
            )
        if str(path).endswith("solar.parquet"):
            return pd.DataFrame(
                {
                    "site_id": ["s"],
                    "ts": [pd.Timestamp("2023-01-10 11:00", tz=KST)],
                    "elev": [45.0],
                }
            )
        return real_read(path, *a, **k)  # type: ignore[arg-type]

    monkeypatch.setattr(pd, "read_parquet", _high_elev)
    assert runner_mod._night_frame(pd.Series([], dtype="object"), 0.0).empty


def test_assignment_fallback_without_canonical(monkeypatch: pytest.MonkeyPatch) -> None:
    import pandas as pd

    from src.oibc.train import runner as runner_mod

    def _missing(*a: object, **k: object) -> pd.DataFrame:
        raise FileNotFoundError("gone")

    monkeypatch.setattr(pd, "read_parquet", _missing)
    panel = pd.DataFrame({"site_id": ["s1"], "current_group": ["A"]})
    with pytest.raises(RunnerError, match=r"sites\.parquet"):
        runner_mod._resolve_assignment(panel)
    with pytest.raises(RunnerError):
        runner_mod._resolve_assignment(pd.DataFrame({"site_id": ["s1"]}))
    with pytest.raises(RunnerError):
        runner_mod._resolve_smp(pd.DataFrame({"ts": []}))
    with pytest.raises(RunnerError, match=r"smp\.parquet"):
        runner_mod._resolve_smp(panel)
    with pytest.raises(RunnerError, match=r"generation\.parquet"):
        runner_mod._night_frame(pd.Series([], dtype="object"), 0.0)


def _clean_git(monkeypatch: pytest.MonkeyPatch, commit: str = "aaa") -> None:
    monkeypatch.setattr(runner_mod, "_git_commit", lambda: commit)
    monkeypatch.setattr(runner_mod, "_git_tree_dirty", lambda: False)


def test_comparable_commit_drift_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.oibc.evaluate.registry import read_records

    panel = _make_panel(tmp_path)
    paths = _paths(tmp_path)
    vcfg = _vcfg()
    _clean_git(monkeypatch, commit="aaa")
    run_validation(_cfg("base"), panel_path=panel, vcfg=vcfg, paths=paths)
    assert len(read_records(paths.registry)) == 1
    monkeypatch.setattr(runner_mod, "_git_commit", lambda: "bbb")
    with pytest.raises(RunnerError):
        run_validation(_cfg("cand", baseline_run_id="base"), panel_path=panel, vcfg=vcfg, paths=paths)
    assert len(read_records(paths.registry)) == 1


def test_comparable_drift_allowed_and_recorded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    panel = _make_panel(tmp_path)
    paths = _paths(tmp_path)
    vcfg = _vcfg()
    _clean_git(monkeypatch, commit="aaa")
    run_validation(_cfg("base"), panel_path=panel, vcfg=vcfg, paths=paths)
    monkeypatch.setattr(runner_mod, "_git_commit", lambda: "bbb")
    cfg = _cfg("cand", allow_commit_drift=True, baseline_run_id="base")
    rec = run_validation(cfg, panel_path=panel, vcfg=vcfg, paths=paths)
    assert "commit_drift_allowed" in rec.note
    assert "볼록성" in rec.note
    assert "예외" not in rec.note


def test_comparable_dirty_tree_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    panel = _make_panel(tmp_path)
    paths = _paths(tmp_path)
    vcfg = _vcfg()
    _clean_git(monkeypatch, commit="aaa")
    run_validation(_cfg("base"), panel_path=panel, vcfg=vcfg, paths=paths)
    monkeypatch.setattr(runner_mod, "_git_tree_dirty", lambda: True)
    with pytest.raises(RunnerError):
        run_validation(_cfg("cand", baseline_run_id="base"), panel_path=panel, vcfg=vcfg, paths=paths)


def test_comparable_panel_hash_mismatch_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    panel = _make_panel(tmp_path)
    paths = _paths(tmp_path)
    vcfg = _vcfg()
    _clean_git(monkeypatch, commit="aaa")
    run_validation(_cfg("base"), panel_path=panel, vcfg=vcfg, paths=paths)
    other = pd.read_parquet(panel)
    other["doy_sin"] = 0.9
    tampered = tmp_path / "panel_other.parquet"
    other.to_parquet(tampered, index=False)
    with pytest.raises(RunnerError):
        run_validation(_cfg("cand", baseline_run_id="base"), panel_path=tampered, vcfg=vcfg, paths=paths)


def test_comparable_config_diff_recorded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    panel = _make_panel(tmp_path)
    paths = _paths(tmp_path)
    vcfg = _vcfg()
    _clean_git(monkeypatch, commit="aaa")
    run_validation(_cfg("base"), panel_path=panel, vcfg=vcfg, paths=paths)
    heavier = RunConfig(
        run_id="cand",
        hypothesis="synthetic guard",
        features=_feat(),
        model=ModelConfig(n_rounds=3, num_leaves=4, n_jobs=1, min_child_samples=5, seeds=(0, 1), weight="none"),
        baseline_run_id="base",
    )
    rec = run_validation(heavier, panel_path=panel, vcfg=vcfg, paths=paths)
    assert (rec.comparison or {})["config_diff"] == ["model.weight"]


def test_comparable_identical_config_null_diff(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    panel = _make_panel(tmp_path)
    paths = _paths(tmp_path)
    vcfg = _vcfg()
    _clean_git(monkeypatch, commit="aaa")
    run_validation(_cfg("base"), panel_path=panel, vcfg=vcfg, paths=paths)
    rec = run_validation(_cfg("cand", baseline_run_id="base"), panel_path=panel, vcfg=vcfg, paths=paths)
    assert (rec.comparison or {})["config_diff"] == []


def test_seed_bag_note_rationale(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    panel = _make_panel(tmp_path)
    paths = _paths(tmp_path)
    _clean_git(monkeypatch, commit="aaa")
    rec = run_validation(_cfg("note-check"), panel_path=panel, vcfg=_vcfg(), paths=paths)
    assert "볼록성" in rec.note
    assert "예외" not in rec.note


def test_git_tree_dirty_none_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    from src.oibc.train import runner as runner_mod

    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(OSError("no git"))
    )
    assert runner_mod._git_tree_dirty() is None

    class _Bad:
        returncode = 1
        stdout = " M x"

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Bad())
    assert runner_mod._git_tree_dirty() is None


def test_comparable_seed_fold_config_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    panel = _make_panel(tmp_path)
    paths = _paths(tmp_path)
    vcfg = _vcfg()
    _clean_git(monkeypatch, commit="aaa")
    run_validation(_cfg("base"), panel_path=panel, vcfg=vcfg, paths=paths)
    other_seeds = ValidationConfig(folds=vcfg.folds, embargo_days=1, min_warm_days=1, seeds=(0, 9), bootstrap_n=100)
    mismatched = RunConfig(
        run_id="bad-seed",
        hypothesis="synthetic guard",
        features=_feat(),
        model=ModelConfig(n_rounds=3, num_leaves=4, n_jobs=1, min_child_samples=5, seeds=(0, 9)),
        baseline_run_id="base",
    )
    with pytest.raises(RunnerError):
        run_validation(mismatched, panel_path=panel, vcfg=other_seeds, paths=paths)
    other_folds = (
        FoldSpec(name="A", test_start=pd.Timestamp("2023-01-10", tz=KST), test_end=pd.Timestamp("2023-01-14", tz=KST)),
        FoldSpec(name="C", test_start=pd.Timestamp("2023-01-15", tz=KST), test_end=pd.Timestamp("2023-01-19", tz=KST)),
    )
    other_val = ValidationConfig(folds=other_folds, embargo_days=1, min_warm_days=1, seeds=(0, 1), bootstrap_n=100)
    with pytest.raises(RunnerError):
        run_validation(_cfg("bad-fold", baseline_run_id="base"), panel_path=panel, vcfg=other_val, paths=paths)
    other_comp = _spec().model_copy(update={"penalty_rate": 0.2})
    with pytest.raises(RunnerError):
        run_validation(
            _cfg("bad-comp", baseline_run_id="base"), panel_path=panel, vcfg=vcfg, paths=paths, spec=other_comp
        )
    other_bootstrap = ValidationConfig(
        folds=vcfg.folds, embargo_days=1, min_warm_days=1, seeds=(0, 1), bootstrap_n=500
    )
    with pytest.raises(RunnerError):
        run_validation(
            _cfg("bad-val", baseline_run_id="base"), panel_path=panel, vcfg=other_bootstrap, paths=paths
        )


def test_config_diff_value_normalization() -> None:
    import copy

    assert runner_mod._normalize_diff_value(frozenset({"b", "a"})) == ["a", "b"]
    assert runner_mod._normalize_diff_value((1, 2)) == [1, 2]
    assert runner_mod._normalize_diff_value("x") == "x"
    rec = _cfg("r").model_dump(mode="json")
    base = RunConfig(
        run_id="b",
        hypothesis="synthetic guard",
        features=_feat(),
        model=ModelConfig(n_rounds=3, num_leaves=4, n_jobs=1, min_child_samples=5, seeds=(0, 1), weight="none"),
    )
    stripped = copy.deepcopy(rec)
    stripped.pop("features")
    fake = type("R", (), {"config": stripped, "seeds": (0, 1), "folds": ("A", "B"), "git_commit": "aaa"})()
    with pytest.raises(RunnerError, match="features config"):
        runner_mod._config_diff_keys(fake, base)


def test_check_comparable_seed_fold_mismatch_direct(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import hashlib

    from src.oibc.evaluate.registry import read_records

    panel = _make_panel(tmp_path)
    paths = _paths(tmp_path)
    vcfg = _vcfg()
    _clean_git(monkeypatch, commit="aaa")
    run_validation(_cfg("base"), panel_path=panel, vcfg=vcfg, paths=paths)
    record = next(r for r in read_records(paths.registry) if r.run_id == "base")
    panel_hash = hashlib.sha256(panel.read_bytes()).hexdigest()
    comp = _spec()
    cfg = _cfg("cand", baseline_run_id="base")
    other_seeds = ValidationConfig(folds=vcfg.folds, embargo_days=1, min_warm_days=1, seeds=(0, 9), bootstrap_n=100)
    with pytest.raises(RunnerError):
        runner_mod._check_comparable(record, cfg, panel_hash, other_seeds, comp)
    other_folds = (
        FoldSpec(name="A", test_start=pd.Timestamp("2023-01-10", tz=KST), test_end=pd.Timestamp("2023-01-14", tz=KST)),
        FoldSpec(name="C", test_start=pd.Timestamp("2023-01-15", tz=KST), test_end=pd.Timestamp("2023-01-19", tz=KST)),
    )
    other_val = ValidationConfig(folds=other_folds, embargo_days=1, min_warm_days=1, seeds=(0, 1), bootstrap_n=100)
    with pytest.raises(RunnerError):
        runner_mod._check_comparable(record, cfg, panel_hash, other_val, comp)


def test_baseline_files_without_registry_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import shutil

    panel = _make_panel(tmp_path)
    paths = _paths(tmp_path)
    vcfg = _vcfg()
    _clean_git(monkeypatch, commit="aaa")
    run_validation(_cfg("base"), panel_path=panel, vcfg=vcfg, paths=paths)
    shutil.copytree(paths.runs / "base", paths.runs / "ghost")
    paths.registry.unlink()
    with pytest.raises(RunnerError, match="unknown baseline"):
        run_validation(_cfg("cand", baseline_run_id="ghost"), panel_path=panel, vcfg=vcfg, paths=paths)


def test_dirty_baseline_rejected_before_fit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    panel = _make_panel(tmp_path)
    paths = _paths(tmp_path)
    _clean_git(monkeypatch)
    monkeypatch.setattr(runner_mod, "_git_tree_dirty", lambda: True)
    baseline = run_validation(_cfg("base"), panel_path=panel, vcfg=_vcfg(), paths=paths)
    assert baseline.config["git_dirty"] is True
    monkeypatch.setattr(runner_mod, "_git_tree_dirty", lambda: False)

    def no_fit(*args, **kwargs):
        pytest.fail("변경된 코드로 학습한 기준과 비교 학습 실행")

    monkeypatch.setattr(runner_mod, "fit_seed_models", no_fit)
    with pytest.raises(RunnerError, match="dirty work tree"):
        run_validation(_cfg("cand", baseline_run_id="base"), panel_path=panel, vcfg=_vcfg(), paths=paths)
    assert not (paths.runs / "cand").exists()


def test_comparable_metadata_boundaries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    panel = _make_panel(tmp_path)
    paths = _paths(tmp_path)
    _clean_git(monkeypatch)
    val = _vcfg()
    record = run_validation(_cfg("base"), panel_path=panel, vcfg=val, paths=paths)
    cfg = _cfg("cand", baseline_run_id="base")
    with pytest.raises(RunnerError, match="panel hash"):
        runner_mod._check_comparable(record, cfg, None, val, _spec())
    for dirty in (None, True):
        config = {**record.config, "git_dirty": dirty}
        changed = record.model_copy(update={"config": config})
        with pytest.raises(RunnerError, match="dirty work tree"):
            runner_mod._check_comparable(changed, cfg, config["panel_sha256"], val, _spec())
        allowed = cfg.model_copy(update={"allow_commit_drift": True})
        assert runner_mod._check_comparable(changed, allowed, config["panel_sha256"], val, _spec()) == []
    cfg = cfg.model_copy(update={"features": FeatureConfig()})
    config = cfg.model_dump(mode="json")
    config["features"]["groups"].reverse()
    changed = record.model_copy(update={"config": config})
    assert runner_mod._config_diff_keys(changed, cfg) == []


def _sites4(cap: float = 100.0) -> pd.DataFrame:
    return pd.DataFrame(
        {"site_id": SITES, "current_group": [GROUPS[s] for s in SITES], "capacity_kw": [cap] * 4}
    )


def _swapped_assignment() -> pd.DataFrame:
    return pd.DataFrame(
        {"site_id": SITES, "group": ["G2", "G1", "G1", "G2"]},
    )


def _comp_paths(tmp_path: Path) -> ArtifactPaths:
    return ArtifactPaths(
        root=tmp_path,
        registry=tmp_path / "registry.jsonl",
        submissions=tmp_path / "subs",
        runs=tmp_path / "runs",
        compositions=tmp_path / "comps",
    )


def test_resolve_composition_reference_matches_current_group() -> None:
    from src.oibc.compose.store import assignment_sha256

    assignment, sha = runner_mod.resolve_composition("current_group", _sites4(), spec=_spec())
    assert assignment.set_index("site_id")["group"].to_dict() == {
        "PV001": "G1", "PV002": "G1", "PV003": "G2", "PV004": "G2",
    }
    assert sha == assignment_sha256(runner_mod.reference_assignment(_sites4()))


def test_resolve_composition_loads_saved(tmp_path: Path) -> None:
    from src.oibc.compose.store import assignment_sha256, load_composition, write_composition
    from src.oibc.config import ComposeConfig

    paths = _comp_paths(tmp_path)
    want = _swapped_assignment()
    write_composition(want, "t1", ComposeConfig(), _sites4(), paths=paths, spec=_spec())
    got, sha = runner_mod.resolve_composition("t1", _sites4(), paths=paths, spec=_spec())
    pd.testing.assert_frame_equal(got, load_composition("t1", _sites4(), paths=paths, spec=_spec())[0])
    assert sha == assignment_sha256(want)


def test_resolve_composition_unknown_id_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(RunnerError):
        runner_mod.resolve_composition("nope", _sites4(), paths=_comp_paths(tmp_path), spec=_spec())


def test_resolve_composition_reference_below_capacity_fails() -> None:
    with pytest.raises(RunnerError):
        runner_mod.resolve_composition("current_group", _sites4(cap=10.0), spec=_spec())


def test_run_validation_records_composition(tmp_path: Path) -> None:
    from src.oibc.compose.store import load_composition, write_composition
    from src.oibc.config import ComposeConfig
    from src.oibc.evaluate.registry import read_records

    panel = _make_panel(tmp_path)
    paths = _comp_paths(tmp_path)
    write_composition(_swapped_assignment(), "t1", ComposeConfig(), _sites4(), paths=paths, spec=_spec())
    cfg = _cfg("comp-rec", composition_id="t1")
    rec = run_validation(cfg, panel_path=panel, vcfg=_vcfg(), paths=paths, spec=_spec())
    assert rec.config["composition_id"] == "t1"
    assert rec.config["composition_sha256"] == load_composition("t1", _sites4(), paths=paths, spec=_spec())[1]
    assert read_records(paths.registry) == [rec]


def test_run_validation_composition_changes_settlement_only(tmp_path: Path) -> None:
    from src.oibc.compose.store import write_composition
    from src.oibc.config import ComposeConfig
    from src.oibc.cv.splits import fold_masks
    from src.oibc.features.panel import feature_columns
    from src.oibc.models.gbm import fit_seed_models, predict_seed_kwh

    panel = _make_panel(tmp_path)
    paths = _comp_paths(tmp_path)
    vcfg = _vcfg()
    write_composition(_swapped_assignment(), "t1", ComposeConfig(), _sites4(), paths=paths, spec=_spec())
    run_validation(_cfg("ref-leg"), panel_path=panel, vcfg=vcfg, paths=paths, spec=_spec())
    run_validation(_cfg("swap-leg", composition_id="t1"), panel_path=panel, vcfg=vcfg, paths=paths, spec=_spec())
    ref = load_daily_runs("ref-leg", "all", paths=paths, vcfg=vcfg)
    swap = load_daily_runs("swap-leg", "all", paths=paths, vcfg=vcfg)
    ref_rev = sum(float(f["revenue"].sum()) for frames in ref.values() for f in frames)
    swap_rev = sum(float(f["revenue"].sum()) for frames in swap.values() for f in frames)
    assert ref_rev == pytest.approx(swap_rev)
    train = pd.read_parquet(panel).query("period == 'train'")
    cfg = _cfg("expected")
    features = feature_columns(train, cfg.features)
    for fold in vcfg.folds:
        tr, te = fold_masks(train["op_date"], fold, vcfg)
        day = train.loc[te].reset_index(drop=True)
        preds = predict_seed_kwh(fit_seed_models(train.loc[tr], features, cfg.model), day, features)
        for assignment, daily_runs in ((reference_assignment(_sites4()), ref), (_swapped_assignment(), swap)):
            group_map = assignment.set_index("site_id")["group"]
            for idx, pred in enumerate(preds):
                residual = day[["ts", "op_date"]].assign(
                    group=day["site_id"].map(group_map), error=pred - day["energy"].to_numpy(),
                )
                net = residual.groupby(["op_date", "ts", "group"])["error"].sum().abs()
                expected = (net * 100.0 * 3 * 0.10).groupby(level="op_date").sum()
                expected += 4 * 2.0 * 100.0 * 0.10
                np.testing.assert_allclose(daily_runs[fold.name][idx]["penalty"], expected)


def test_config_diff_flags_composition(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    panel = _make_panel(tmp_path)
    paths = _comp_paths(tmp_path)
    _clean_git(monkeypatch)
    run_validation(_cfg("base"), panel_path=panel, vcfg=_vcfg(), paths=paths, spec=_spec())
    from src.oibc.evaluate.registry import read_records

    record = next(r for r in read_records(paths.registry) if r.run_id == "base")
    legacy = record.model_copy(update={"config": {k: v for k, v in record.config.items() if k != "composition_id"}})
    cand = _cfg("cand", composition_id="geo64-s0")
    assert "composition_id" in runner_mod._config_diff_keys(legacy, cand)


def test_forecast_eval_uses_run_composition(tmp_path: Path) -> None:
    from src.oibc.compose.store import write_composition
    from src.oibc.config import ComposeConfig
    from src.oibc.features.panel import feature_columns
    from src.oibc.models.gbm import bag, fit_seed_models, predict_seed_kwh

    panel = _make_panel(tmp_path)
    paths = _comp_paths(tmp_path)
    write_composition(_swapped_assignment(), "t1", ComposeConfig(), _sites4(), paths=paths, spec=_spec())
    ref = forecast_eval(_cfg("r1"), panel_path=panel, spec=_spec())
    swap = forecast_eval(_cfg("r2", composition_id="t1"), panel_path=panel, spec=_spec(), paths=paths)
    assert float(ref["forecast_kwh"].sum()) == pytest.approx(float(swap["forecast_kwh"].sum()))
    ref_g1 = ref.loc[ref["group"] == "G1"].sort_values("ts")["forecast_kwh"].to_numpy()
    swap_g1 = swap.loc[swap["group"] == "G1"].sort_values("ts")["forecast_kwh"].to_numpy()
    assert not bool((ref_g1 == swap_g1).all())
    frame = pd.read_parquet(panel)
    cfg = _cfg("expected")
    features = feature_columns(frame, cfg.features)
    day = frame.loc[frame["period"] == "eval"].copy()
    models = fit_seed_models(frame.loc[frame["period"] == "train"], features, cfg.model)
    day["pred"] = bag(predict_seed_kwh(models, day, features))
    day["group"] = day["site_id"].map(_swapped_assignment().set_index("site_id")["group"])
    expected = day.groupby(["group", "ts"])["pred"].sum()
    actual = swap.set_index(["group", "ts"])["forecast_kwh"]
    np.testing.assert_allclose(actual.reindex(expected.index), expected, rtol=1e-12)


def test_validation_accepts_panel_site_subset(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    import logging

    path = _make_panel(tmp_path)
    panel = pd.read_parquet(path)
    panel.loc[panel["site_id"] != "PV004"].to_parquet(path, index=False)
    gen_path = tmp_path / "canonical" / "generation.parquet"
    generation = pd.read_parquet(gen_path)
    generation.loc[~(generation["site_id"].eq("PV004") & generation["ts"].dt.hour.ne(1))].to_parquet(
        gen_path, index=False,
    )
    with caplog.at_level(logging.INFO):
        rec = run_validation(_cfg("subset"), panel_path=path, vcfg=_vcfg(), paths=_comp_paths(tmp_path))
    assert rec.config["composition_id"] == "current_group"
    message = next(r.message for r in caplog.records if "run=subset composition_id=" in r.message)
    assert "caps={'G1': 200.0, 'G2': 200.0}" in message


def test_resolve_assignment_explicit_spec(tmp_path: Path) -> None:
    panel = _make_panel(tmp_path)
    assignment, sha = runner_mod._resolve_assignment(pd.read_parquet(panel), spec=_spec())
    assert set(assignment["group"].tolist()) == {"G1", "G2"}
    assert len(sha) == 64


def test_resolve_assignment_none_composition_rejected(tmp_path: Path) -> None:
    panel = _make_panel(tmp_path)
    with pytest.raises(RunnerError, match="unsafe comp_id"):
        runner_mod._resolve_assignment(pd.read_parquet(panel), None)  # type: ignore[arg-type]


@pytest.mark.parametrize("legacy", [False, True])
def test_pipeline_submission_composition_roundtrip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, legacy: bool,
) -> None:
    import json

    from src.oibc import pipeline
    from src.oibc.compose import store
    from src.oibc.config import ComposeConfig
    from src.oibc.evaluate import registry
    from src.oibc.submit import build, validate

    panel = _make_panel(tmp_path)
    paths = _comp_paths(tmp_path)
    comp_id = "current_group" if legacy else "t1"
    assignment = reference_assignment(_sites4()) if legacy else _swapped_assignment()
    if not legacy:
        store.write_composition(assignment, comp_id, ComposeConfig(), _sites4(), paths=paths, spec=_spec())
    cfg = _cfg("roundtrip", composition_id=comp_id)
    rec = run_validation(cfg, panel_path=panel, vcfg=_vcfg(), paths=paths, spec=_spec())
    digest = rec.config["composition_sha256"]
    if legacy:
        config = {k: v for k, v in rec.config.items() if k not in {"composition_id", "composition_sha256"}}
        rec = rec.model_copy(update={"config": config})
        paths.registry.write_text(rec.model_dump_json() + "\n", encoding="utf-8")
    raw = tmp_path / "raw"
    raw.mkdir()
    _sites4().to_csv(raw / "sites.csv", index=False)
    stamps = pd.date_range(_spec().eval_start, _spec().eval_end, freq="h")
    pd.DataFrame({"ts": stamps, "smp": 100.0}).to_csv(raw / "smp.csv", index=False)
    data = runner_mod.DataPaths().model_copy(update={"raw_dir": raw})
    monkeypatch.setattr(pipeline, "DataPaths", lambda: data)
    monkeypatch.setattr(validate, "DataPaths", lambda: data)
    for module in (registry, store, build):
        monkeypatch.setattr(module, "ArtifactPaths", lambda: paths)
    assert pipeline.main(["submit", "--run-id", cfg.run_id, "--sub-id", "s1"]) == 0
    folder = paths.submissions / "s1"
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["composition_id"] == comp_id
    assert manifest["composition_sha256"] == digest
    got = pd.read_csv(folder / "composition.csv", dtype="string")
    pd.testing.assert_frame_equal(got, assignment.astype("string"))
    report = validate.validate_submission(folder / "composition.csv", folder / "forecast.csv", spec=_spec())
    assert report.ok
    assert sum(report.group_forecast_total_kwh.values()) > 0


def _oof_seed_cols(seeds: tuple[int, ...]) -> list[str]:
    return [f"pred_s{s}" for s in seeds]


def test_run_validation_default_does_not_write_oof(tmp_path: Path) -> None:
    panel = _make_panel(tmp_path)
    paths = _paths(tmp_path)
    run_validation(_cfg("no-oof"), panel_path=panel, vcfg=_vcfg(), paths=paths)
    assert list((paths.runs / "no-oof").glob("*_oof.parquet")) == []
    from src.oibc.evaluate.registry import read_records

    rec = next(r for r in read_records(paths.registry) if r.run_id == "no-oof")
    assert rec.config["oof_saved"] is False
    daily = sorted(p.name for p in (paths.runs / "no-oof").glob("*.parquet"))
    assert daily == sorted(
        f"{f}_{tag}_{s}.parquet"
        for f in ("A", "B") for tag in ("s0", "s1", "bag") for s in ("all", "warm")
    )


def test_run_validation_save_oof_schema(tmp_path: Path) -> None:
    panel = _make_panel(tmp_path)
    paths = _paths(tmp_path)
    vcfg = _vcfg()
    run_validation(_cfg("oof1"), panel_path=panel, vcfg=vcfg, paths=paths, save_oof=True)
    seeds = (0, 1)
    want_cols = ["site_id", "ts", "op_date", "actual_kwh", *_oof_seed_cols(seeds), "pred_bag", "is_warm"]
    for fold in ("A", "B"):
        frame = pd.read_parquet(paths.runs / "oof1" / f"{fold}_oof.parquet")
        assert frame.columns.tolist() == want_cols
        assert str(frame["site_id"].dtype) == "string"
        assert str(frame["ts"].dtype) == _DT
        assert str(frame["op_date"].dtype) == _DT
        assert all(str(frame[c].dtype) == "float64" for c in ["actual_kwh", *want_cols[4:-1]])
        assert str(frame["is_warm"].dtype) == "bool"
        assert not frame.duplicated(subset=["site_id", "ts"]).any()
        ordered = frame.sort_values(["site_id", "ts"], kind="mergesort").reset_index(drop=True)
        pd.testing.assert_frame_equal(frame, ordered)


def test_run_validation_oof_conserves_daily_settlement(tmp_path: Path) -> None:
    from src.oibc.evaluate.settlement import aggregate_groups, daily_totals, settle

    panel = _make_panel(tmp_path)
    paths = _paths(tmp_path)
    vcfg = _vcfg()
    run_validation(_cfg("oof2"), panel_path=panel, vcfg=vcfg, paths=paths, save_oof=True)
    sites = pd.read_parquet(tmp_path / "canonical" / "sites.parquet")
    assignment = reference_assignment(sites)
    smp = pd.read_parquet(tmp_path / "canonical" / "smp.parquet")
    for fold in ("A", "B"):
        oof = load_oof("oof2", fold, paths=paths)
        tag_of = {f"pred_s{s}": f"s{s}" for s in (0, 1)} | {"pred_bag": "bag"}
        for col, tag in tag_of.items():
            site = oof[["site_id", "ts"]].assign(
                pred_kwh=oof[col].to_numpy(dtype="float64"),
                actual_kwh=oof["actual_kwh"].to_numpy(dtype="float64"),
            )
            daily = daily_totals(settle(aggregate_groups(site, assignment, spec=_spec()), smp, spec=_spec()))
            saved = runner_mod._normalize_daily(
                pd.read_parquet(paths.runs / "oof2" / f"{fold}_{tag}_all.parquet"), "x",
            )
            np.testing.assert_allclose(daily["penalty"], saved["penalty"], rtol=1e-12)
            np.testing.assert_allclose(daily["revenue"], saved["revenue"], rtol=1e-12)


def test_run_validation_oof_night_rows_zero(tmp_path: Path) -> None:
    panel = _make_panel(tmp_path)
    paths = _paths(tmp_path)
    run_validation(_cfg("oof3"), panel_path=panel, vcfg=_vcfg(), paths=paths, save_oof=True)
    for fold in ("A", "B"):
        oof = load_oof("oof3", fold, paths=paths)
        night = oof.loc[~oof["ts"].dt.hour.isin([11, 12, 13])]
        assert len(night) > 0
        assert bool((oof.loc[oof["ts"].dt.hour.eq(1), [c for c in oof.columns if c.startswith("pred_")]] == 0.0).all().all())


def _make_cold_panel(tmp_path: Path) -> Path:
    panel_path = _make_panel(tmp_path)
    cutoff = pd.Timestamp("2023-01-09", tz=KST)
    panel = pd.read_parquet(panel_path)
    panel.loc[~((panel["site_id"] == "PV004") & (panel["op_date"] < cutoff))].to_parquet(panel_path, index=False)
    for name in ("generation", "solar"):
        path = tmp_path / ("canonical" if name == "generation" else "derived") / f"{name}.parquet"
        frame = pd.read_parquet(path)
        if "site_id" in frame.columns and "ts" in frame.columns:
            frame.loc[~((frame["site_id"] == "PV004") & (frame["ts"] < cutoff))].to_parquet(path, index=False)
    return panel_path


def test_run_validation_oof_warm_flag_matches_warm_subset(tmp_path: Path) -> None:
    from src.oibc.evaluate.settlement import aggregate_groups, daily_totals, settle

    panel = _make_cold_panel(tmp_path)
    paths = _paths(tmp_path)
    run_validation(_cfg("oof4"), panel_path=panel, vcfg=_vcfg(), paths=paths, save_oof=True)
    sites = pd.read_parquet(tmp_path / "canonical" / "sites.parquet")
    assignment = reference_assignment(sites)
    smp = pd.read_parquet(tmp_path / "canonical" / "smp.parquet")
    oof_a = load_oof("oof4", "A", paths=paths)
    assert set(oof_a.loc[oof_a["site_id"] == "PV004", "is_warm"]) == {False}
    assert set(oof_a.loc[oof_a["site_id"] != "PV004", "is_warm"]) == {True}
    for fold in ("A", "B"):
        oof = load_oof("oof4", fold, paths=paths)
        warm = oof.loc[oof["is_warm"]].reset_index(drop=True)
        assert len(warm) < len(oof) if fold == "A" else len(warm) == len(oof)
        site = warm[["site_id", "ts"]].assign(
            pred_kwh=warm["pred_bag"].to_numpy(dtype="float64"),
            actual_kwh=warm["actual_kwh"].to_numpy(dtype="float64"),
        )
        daily = daily_totals(settle(aggregate_groups(site, assignment, spec=_spec()), smp, spec=_spec()))
        saved = runner_mod._normalize_daily(
            pd.read_parquet(paths.runs / "oof4" / f"{fold}_bag_warm.parquet"), "x",
        )
        np.testing.assert_allclose(daily["penalty"], saved["penalty"], rtol=1e-12)
        np.testing.assert_allclose(daily["revenue"], saved["revenue"], rtol=1e-12)


def test_run_validation_save_oof_not_in_config_diff(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    panel = _make_panel(tmp_path)
    paths = _paths(tmp_path)
    vcfg = _vcfg()
    _clean_git(monkeypatch)
    run_validation(_cfg("base"), panel_path=panel, vcfg=vcfg, paths=paths)
    rec = run_validation(_cfg("cand", baseline_run_id="base"), panel_path=panel, vcfg=vcfg, paths=paths, save_oof=True)
    assert (rec.comparison or {})["config_diff"] == []


def test_run_validation_refactor_preserves_daily_outputs(tmp_path: Path) -> None:
    panel = _make_panel(tmp_path)
    paths = _paths(tmp_path)
    vcfg = _vcfg()
    run_validation(_cfg("plain"), panel_path=panel, vcfg=vcfg, paths=paths, save_oof=False)
    run_validation(_cfg("with-oof"), panel_path=panel, vcfg=vcfg, paths=paths, save_oof=True)
    for fold in ("A", "B"):
        for tag in ("s0", "s1", "bag"):
            for subset in ("all", "warm"):
                name = f"{fold}_{tag}_{subset}.parquet"
                left = runner_mod._normalize_daily(pd.read_parquet(paths.runs / "plain" / name), "x")
                right = runner_mod._normalize_daily(pd.read_parquet(paths.runs / "with-oof" / name), "x")
                pd.testing.assert_frame_equal(left, right)


def test_load_oof_roundtrip(tmp_path: Path) -> None:
    panel = _make_panel(tmp_path)
    paths = _paths(tmp_path)
    run_validation(_cfg("oof5"), panel_path=panel, vcfg=_vcfg(), paths=paths, save_oof=True)
    for fold in ("A", "B"):
        pd.testing.assert_frame_equal(
            load_oof("oof5", fold, paths=paths),
            pd.read_parquet(paths.runs / "oof5" / f"{fold}_oof.parquet"),
        )


def test_load_oof_missing_file_fails_closed(tmp_path: Path) -> None:
    panel = _make_panel(tmp_path)
    paths = _paths(tmp_path)
    run_validation(_cfg("oof6"), panel_path=panel, vcfg=_vcfg(), paths=paths)
    with pytest.raises(RunnerError, match="--save-oof"):
        load_oof("oof6", "A", paths=paths)


def test_load_oof_rejects_unsafe_component(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    with pytest.raises(RunnerError):
        load_oof("../x", "A", paths=paths)


def test_load_oof_rejects_negative_prediction(tmp_path: Path) -> None:
    panel = _make_panel(tmp_path)
    paths = _paths(tmp_path)
    run_validation(_cfg("oof7"), panel_path=panel, vcfg=_vcfg(), paths=paths, save_oof=True)
    target = paths.runs / "oof7" / "A_oof.parquet"
    frame = pd.read_parquet(target)
    frame.loc[frame.index[0], "pred_s0"] = -1.0
    frame.to_parquet(target, index=False)
    with pytest.raises(RunnerError):
        load_oof("oof7", "A", paths=paths)


@pytest.mark.parametrize("defect", ["unreadable", "columns", "seeds", "duplicate", "hour", "actual"])
def test_load_oof_rejects_corrupt_files(tmp_path: Path, defect: str) -> None:
    panel = _make_panel(tmp_path)
    paths = _paths(tmp_path)
    run_validation(_cfg("oof8"), panel_path=panel, vcfg=_vcfg(), paths=paths, save_oof=True)
    target = paths.runs / "oof8" / "A_oof.parquet"
    if defect == "unreadable":
        target.write_bytes(b"not a parquet file")
    else:
        frame = pd.read_parquet(target)
        if defect == "columns":
            frame = frame.drop(columns=["pred_bag"])
        elif defect == "seeds":
            frame = frame.drop(columns=[c for c in frame.columns if c.startswith("pred_s")])
        elif defect == "duplicate":
            frame = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
        elif defect == "hour":
            bad = frame.copy()
            bad["ts"] = pd.Series(
                pd.DatetimeIndex(bad["ts"]) + pd.to_timedelta(range(len(bad)), unit="m")
            )
            frame = bad
        else:
            frame.loc[frame.index[0], "actual_kwh"] = float("nan")
        frame.to_parquet(target, index=False)
    with pytest.raises(RunnerError):
        load_oof("oof8", "A", paths=paths)


@pytest.mark.parametrize("defect", [None, "warm_string", "warm_null", "date", "date_dtype", "numeric", "site"])
def test_load_oof_preserves_schema_and_row_pairing(tmp_path: Path, defect: str | None) -> None:
    paths = _paths(tmp_path)
    folder = paths.runs / "schema"
    folder.mkdir(parents=True)
    ts = pd.Series(pd.DatetimeIndex(["2023-01-11 00:00", "2023-01-10 11:00"], tz=KST), dtype=_DT)
    frame = pd.DataFrame({
        "site_id": pd.Series(["PV002", "PV001"], dtype="string"),
        "ts": ts,
        "op_date": op_date(ts).astype(_DT),
        "actual_kwh": [2.0, 4.0],
        "pred_s1": [3.0, 5.0],
        "pred_s0": [1.0, 6.0],
        "pred_bag": [2.0, 5.5],
        "is_warm": [True, False],
    })
    if defect == "warm_string":
        frame["is_warm"] = ["False", "True"]
    elif defect == "warm_null":
        frame["is_warm"] = pd.Series([True, pd.NA], dtype="boolean")
    elif defect == "date":
        frame.loc[0, "op_date"] += pd.Timedelta(days=1)
    elif defect == "date_dtype":
        frame["op_date"] = frame["op_date"].astype("str")
    elif defect == "numeric":
        frame["pred_bag"] = frame["pred_bag"].astype("str")
    elif defect == "site":
        frame["site_id"] = [1, 2]
    frame.index = [9, 3]
    frame.to_parquet(folder / "A_oof.parquet")
    if defect is not None:
        with pytest.raises(RunnerError, match=r"schema|op_date"):
            load_oof("schema", "A", paths=paths)
    else:
        expected = frame.sort_values(["site_id", "ts"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(load_oof("schema", "A", paths=paths), expected)
