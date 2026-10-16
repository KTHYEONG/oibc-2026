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
def test_real_panel_weekly_w_within_tolerance() -> None:
    from src.oibc.config import repo_root
    from src.oibc.cv.splits import fold_masks
    from src.oibc.evaluate.settlement import aggregate_groups, settle
    from src.oibc.features.panel import feature_columns
    from src.oibc.models.gbm import fit_seed_models, predict_seed_kwh

    panel_path = DataPaths().features_dir / "panel.parquet"
    refs = repo_root() / "scratch" / "fe_runs"
    if not panel_path.exists() or not (refs / "all_A_s0_all.parquet").exists():
        pytest.skip("실데이터 패널 또는 probe 진단 산출물 없음")
    panel = pd.read_parquet(panel_path)
    panel = panel.loc[panel["period"] == "train"]
    features = feature_columns(panel, FeatureConfig())
    assignment = runner_mod._resolve_assignment(panel)
    smp = runner_mod._resolve_smp(panel)
    vcfg = ValidationConfig()
    for fold in vcfg.folds:
        tr, te = fold_masks(panel["op_date"], fold, vcfg)
        train, test = panel.loc[tr], panel.loc[te]
        models = fit_seed_models(train, features, ModelConfig())
        preds = predict_seed_kwh(models, test, features)
        for seed, pred in zip(vcfg.seeds, preds, strict=True):
            frame = site_frame(test, pred, test.iloc[:0])
            penalty = settle(aggregate_groups(frame, assignment), smp).penalty
            reference = float(pd.read_parquet(refs / f"all_{fold.name}_s{seed}_all.parquet")["penalty"].sum())
            assert penalty == pytest.approx(reference, rel=0.02)
        del models, preds, train, test


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


def test_run_validation_with_baseline(tmp_path: Path) -> None:
    panel = _make_panel(tmp_path)
    paths = _paths(tmp_path)
    vcfg = _vcfg()
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
