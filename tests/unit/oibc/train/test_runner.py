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


_PANEL_BYTES: dict[str, bytes] = {}
_PANEL_REL_PATHS = (
    "panel.parquet",
    "canonical/sites.parquet",
    "canonical/generation.parquet",
    "canonical/smp.parquet",
    "derived/grid.parquet",
    "derived/solar.parquet",
)


def _make_panel(tmp_path: Path) -> Path:
    path = tmp_path / "panel.parquet"
    if _PANEL_BYTES:
        for rel, data in _PANEL_BYTES.items():
            target = tmp_path / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        return path
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
    for rel in _PANEL_REL_PATHS:
        _PANEL_BYTES[rel] = (tmp_path / rel).read_bytes()
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


def test_site_frame_columnar_equivalence_shuffled() -> None:
    ts11 = pd.Timestamp("2023-01-10 11:00", tz=KST)
    ts12 = pd.Timestamp("2023-01-10 12:00", tz=KST)
    ts00 = pd.Timestamp("2023-01-10 00:00", tz=KST)
    day = pd.DataFrame(
        {
            "site_id": ["PV002", "PV001", "PV002", "PV001"],
            "ts": [ts11, ts12, ts12, ts11],
            "energy": [30.0, 20.0, 40.0, 10.0],
        }
    )
    pred = np.array([29.0, 19.0, 39.0, 9.0])
    night = pd.DataFrame(
        {"site_id": ["PV002", "PV001"], "ts": [ts00, ts00], "energy": [1.0, 5.0]}
    )
    day.index = [8, 2, 6, 4]
    night.index = [6, 2]
    out = site_frame(day, pred, night)
    expected = pd.DataFrame(
        {
            "site_id": pd.Series(["PV001", "PV001", "PV001", "PV002", "PV002", "PV002"], dtype="string"),
            "ts": pd.Series(
                pd.DatetimeIndex([ts00, ts11, ts12, ts00, ts11, ts12]).as_unit("us"), dtype=_DT
            ),
            "pred_kwh": pd.Series([0.0, 9.0, 19.0, 0.0, 29.0, 39.0], dtype="float64"),
            "actual_kwh": pd.Series([5.0, 10.0, 20.0, 1.0, 30.0, 40.0], dtype="float64"),
        }
    )
    pd.testing.assert_frame_equal(out, expected, check_exact=True)


def test_site_frame_mixed_timestamp_units() -> None:
    day, pred, night = _day_night()
    day_ms = day.copy()
    day_ms["ts"] = pd.Series(pd.DatetimeIndex(day["ts"]).as_unit("ms"))
    night_us = night.copy()
    night_us["ts"] = pd.Series(pd.DatetimeIndex(night["ts"]).as_unit("us"))
    out = site_frame(day_ms, pred, night_us)
    expected = pd.DataFrame(
        {
            "site_id": pd.Series(["PV001"] * 5, dtype="string"),
            "ts": pd.Series(
                pd.DatetimeIndex(
                    ["2023-01-10 00:00", "2023-01-10 01:00", "2023-01-10 11:00",
                     "2023-01-10 12:00", "2023-01-10 13:00"], tz=KST,
                ), dtype=_DT,
            ),
            "pred_kwh": [0.0, 0.0, 9.0, 19.0, 29.0],
            "actual_kwh": [5.0, 0.0, 10.0, 20.0, 30.0],
        }
    )
    pd.testing.assert_frame_equal(out, expected, check_exact=True)


@pytest.mark.parametrize("empty_part", ["day", "night", "both"])
@pytest.mark.parametrize("empty_ts_dtype", ["object", "datetime64[ns]", _DT])
def test_site_frame_empty_part(empty_part: str, empty_ts_dtype: str) -> None:
    day, pred, night = _day_night()
    if empty_part in ("day", "both"):
        day = pd.DataFrame({"site_id": pd.Series(dtype="string"), "ts": pd.Series(dtype=empty_ts_dtype)})
        pred = np.empty(0, dtype="float64")
    if empty_part in ("night", "both"):
        night = pd.DataFrame({"site_id": pd.Series(dtype="string"), "ts": pd.Series(dtype=empty_ts_dtype)})
    expected = pd.DataFrame(
        {
            "site_id": pd.Series(dtype="string"),
            "ts": pd.Series(dtype=_DT),
            "pred_kwh": pd.Series(dtype="float64"),
            "actual_kwh": pd.Series(dtype="float64"),
        }
    )
    if len(day):
        expected = day.rename(columns={"energy": "actual_kwh"}).assign(pred_kwh=pred)
    elif len(night):
        expected = night.rename(columns={"energy": "actual_kwh"}).assign(pred_kwh=0.0)
    expected["site_id"] = expected["site_id"].astype("string")
    expected = expected[["site_id", "ts", "pred_kwh", "actual_kwh"]].reset_index(drop=True)
    pd.testing.assert_frame_equal(site_frame(day, pred, night), expected, check_exact=True)


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


def _netting_run(tmp_path: Path, run_id: str = "net1") -> tuple[Path, ArtifactPaths]:
    from src.oibc.compose.store import write_composition
    from src.oibc.config import ComposeConfig

    panel = _make_panel(tmp_path)
    paths = _comp_paths(tmp_path)
    write_composition(_swapped_assignment(), "t1", ComposeConfig(), _sites4(), paths=paths, spec=_spec())
    run_validation(_cfg(run_id), panel_path=panel, vcfg=_vcfg(), paths=paths, spec=_spec(), save_oof=True)
    return panel, paths


def test_netting_report_structure_and_identity(tmp_path: Path) -> None:
    from src.oibc.train.runner import netting_report

    _, paths = _netting_run(tmp_path)
    report = netting_report("net1", ["current_group", "t1"], n_random=4, seed=0, paths=paths, spec=_spec())
    assert report["run_id"] == "net1" and report["n_random"] == 4 and report["seed"] == 0
    assert set(report["compositions"]) == {"current_group", "t1"}
    for fold in ("A", "B"):
        entry = report["folds"][fold]
        assert set(entry) == {"pool_penalty", "compositions", "random", "capacity_balanced", "composition_percentile_vs_random"}
        for vals in entry["compositions"].values():
            assert set(vals) == {"penalty", "netting_loss", "netting_share"}
            assert vals["penalty"] == pytest.approx(entry["pool_penalty"] + vals["netting_loss"])
        assert set(entry["random"]) == {"mean", "sd", "p05", "p50", "p95"}


def test_netting_report_reference_matches_run_bag(tmp_path: Path) -> None:
    from src.oibc.train.runner import netting_report

    _, paths = _netting_run(tmp_path, "netbag")
    report = netting_report("netbag", ["current_group"], n_random=2, seed=1, paths=paths, spec=_spec())
    for fold in ("A", "B"):
        saved = runner_mod._normalize_daily(
            pd.read_parquet(paths.runs / "netbag" / f"{fold}_bag_all.parquet"), "x"
        )
        assert report["folds"][fold]["compositions"]["current_group"]["penalty"] == pytest.approx(
            float(saved["penalty"].sum()), rel=1e-9
        )


def test_netting_report_deterministic(tmp_path: Path) -> None:
    from src.oibc.train.runner import netting_report

    _, paths = _netting_run(tmp_path, "netdet")
    first = netting_report("netdet", ["current_group"], n_random=4, seed=2, paths=paths, spec=_spec())
    second = netting_report("netdet", ["current_group"], n_random=4, seed=2, paths=paths, spec=_spec())
    assert first == second


def test_netting_report_requires_oof(tmp_path: Path) -> None:
    from src.oibc.train.runner import netting_report

    panel = _make_panel(tmp_path)
    paths = _comp_paths(tmp_path)
    run_validation(_cfg("nooof"), panel_path=panel, vcfg=_vcfg(), paths=paths, spec=_spec())
    with pytest.raises(RunnerError):
        netting_report("nooof", ["current_group"], n_random=2, seed=0, paths=paths, spec=_spec())


def test_netting_report_rejects_bad_args(tmp_path: Path) -> None:
    from src.oibc.train.runner import netting_report

    _, paths = _netting_run(tmp_path, "netbad")
    with pytest.raises(RunnerError):
        netting_report("netbad", ["current_group"], n_random=0, paths=paths, spec=_spec())
    with pytest.raises(RunnerError):
        netting_report("netbad", [], n_random=2, paths=paths, spec=_spec())
    with pytest.raises(RunnerError):
        netting_report("netbad", ["current_group", "current_group"], n_random=2, paths=paths, spec=_spec())


def test_netting_report_fail_closed_branches(tmp_path: Path) -> None:
    from src.oibc.train.runner import netting_report

    _, paths = _netting_run(tmp_path, "netfail")
    with pytest.raises(RunnerError):
        netting_report("missing-run", ["current_group"], n_random=1, paths=paths, spec=_spec())
    with pytest.raises(RunnerError):
        netting_report("netfail", ["../evil"], n_random=1, paths=paths, spec=_spec())
    with pytest.raises(RunnerError):
        netting_report("netfail", ["no-such-comp"], n_random=1, paths=paths, spec=_spec())
    with pytest.raises(RunnerError):
        netting_report("netfail", ["current_group"], n_random=1, paths=paths, spec=_spec(), pred_col="nope")


def test_netting_report_translates_baseline_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.oibc.compose.baselines as baselines_mod
    from src.oibc.compose.constraints import CompositionError
    from src.oibc.train.runner import netting_report

    _, paths = _netting_run(tmp_path, "nettr")
    monkeypatch.setattr(
        baselines_mod, "random_assignment",
        lambda *a, **k: (_ for _ in ()).throw(CompositionError("boom")),
    )
    with pytest.raises(RunnerError):
        netting_report("nettr", ["current_group"], n_random=1, paths=paths, spec=_spec())


def test_netting_report_translates_evaluate_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.oibc.evaluate.netting as netting_mod
    from src.oibc.evaluate.netting import NettingError
    from src.oibc.train.runner import netting_report

    _, paths = _netting_run(tmp_path, "nettre")
    monkeypatch.setattr(
        netting_mod.NettingEvaluator, "evaluate",
        lambda self, assignment: (_ for _ in ()).throw(NettingError("boom")),
    )
    with pytest.raises(RunnerError, match="boom"):
        netting_report("nettre", ["current_group"], n_random=1, paths=paths, spec=_spec())


def test_netting_report_translates_penalties_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.oibc.evaluate.netting as netting_mod
    from src.oibc.evaluate.netting import NettingError
    from src.oibc.train.runner import netting_report

    _, paths = _netting_run(tmp_path, "nettrp")
    original = netting_mod.NettingEvaluator.penalties

    def fail_batch(self, assignments):
        if len(assignments) > 1:
            raise NettingError("boom")
        return original(self, assignments)

    monkeypatch.setattr(
        netting_mod.NettingEvaluator, "penalties",
        fail_batch,
    )
    with pytest.raises(RunnerError, match="boom"):
        netting_report("nettrp", ["current_group"], n_random=2, paths=paths, spec=_spec())


@pytest.mark.parametrize("defect", ["missing_spec", "different_spec", "future_oof", "fractional", "seed"])
def test_netting_report_reproducibility_guards(tmp_path: Path, defect: str) -> None:
    from src.oibc.evaluate.registry import read_records

    _, paths = _netting_run(tmp_path, "guard")
    if defect == "missing_spec":
        record = read_records(paths.registry)[0]
        config = {k: v for k, v in record.config.items() if k != "competition"}
        paths.registry.write_text(record.model_copy(update={"config": config}).model_dump_json() + "\n")
    elif defect == "future_oof":
        target = paths.runs / "guard" / "A_oof.parquet"
        frame = pd.read_parquet(target)
        frame["ts"] += pd.Timedelta(days=1000)
        frame["op_date"] = op_date(frame["ts"])
        frame.to_parquet(target, index=False)
    kwargs = {"paths": paths, "n_random": 2}
    if defect == "different_spec":
        kwargs["spec"] = _spec().model_copy(update={"penalty_rate": 0.2})
    elif defect == "fractional":
        kwargs["n_random"] = 1.5
    elif defect == "seed":
        kwargs["seed"] = -1
    with pytest.raises(RunnerError):
        runner_mod.netting_report("guard", ["current_group"], **kwargs)


def test_netting_report_uses_recorded_spec_and_preserves_artifacts(tmp_path: Path) -> None:
    import json

    _, paths = _netting_run(tmp_path, "recorded")
    before = {p: p.read_bytes() for p in (paths.runs / "recorded").iterdir()}
    registry = paths.registry.read_bytes()
    report = runner_mod.netting_report("recorded", ["current_group"], n_random=1, paths=paths)
    assert report == json.loads(json.dumps(report, allow_nan=False))
    assert report == runner_mod.netting_report("recorded", ["current_group"], n_random=1, paths=paths, spec=_spec())
    assert before == {p: p.read_bytes() for p in (paths.runs / "recorded").iterdir()}
    assert paths.registry.read_bytes() == registry


def test_pipeline_netting_synthetic_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json

    from src.oibc import config, pipeline

    _, paths = _netting_run(tmp_path, "cli-netting")
    monkeypatch.setattr(config, "ArtifactPaths", lambda: paths)
    monkeypatch.setattr(runner_mod, "ArtifactPaths", lambda: paths)
    args = ["report-netting", "--run-id", "cli-netting", "--composition", "current_group",
            "--composition", "t1", "--n-random", "3", "--seed", "5"]
    assert pipeline.main(args) == 0
    output = paths.runs / "cli-netting" / "netting_n3_seed5.json"
    payload = output.read_bytes()
    report = json.loads(payload)
    for fold in ("A", "B"):
        saved = pd.read_parquet(paths.runs / "cli-netting" / f"{fold}_bag_all.parquet")
        assert report["folds"][fold]["compositions"]["current_group"]["penalty"] == pytest.approx(
            saved["penalty"].sum(), rel=1e-9,
        )
    with pytest.raises(FileExistsError):
        pipeline.main(args)
    assert output.read_bytes() == payload


def _regional_cfg(run_id: str, weight: float = 0.6) -> RunConfig:
    from src.oibc.config import RegionalConfig

    return _cfg(
        run_id,
        model=ModelConfig(
            n_rounds=3, num_leaves=4, n_jobs=1, min_child_samples=5, seeds=(0, 1),
            regional=RegionalConfig(
                weight=weight, units=((1, 0), (2, 0)), kmeans_n_init=1,
                num_leaves=4, min_child_samples=5, n_rounds=3,
            ),
        ),
    )


def test_run_validation_zero_weight_bit_identical(tmp_path: Path) -> None:
    panel = _make_panel(tmp_path)
    paths = _paths(tmp_path)
    vcfg = _vcfg()
    run_validation(_regional_cfg("w0", weight=0.0), panel_path=panel, vcfg=vcfg, paths=paths, save_oof=True)
    run_validation(_cfg("plain"), panel_path=panel, vcfg=vcfg, paths=paths, save_oof=True)
    for fold in ("A", "B"):
        left = load_oof("w0", fold, paths=paths)
        right = load_oof("plain", fold, paths=paths)
        for col in ("pred_s0", "pred_s1", "pred_bag"):
            np.testing.assert_array_equal(left[col].to_numpy(), right[col].to_numpy())
    from src.oibc.evaluate.registry import read_records

    rec = next(r for r in read_records(paths.registry) if r.run_id == "w0")
    assert all("regional_sec" not in m for m in rec.metrics.values())


def test_run_validation_blend_applied_to_oof(tmp_path: Path) -> None:
    from src.oibc.models.regional import site_units

    panel = _make_panel(tmp_path)
    paths = _paths(tmp_path)
    vcfg = _vcfg()
    cfg_w = _regional_cfg("blend", weight=0.6)
    run_validation(_regional_cfg("base0", weight=0.0), panel_path=panel, vcfg=vcfg, paths=paths, save_oof=True)
    run_validation(cfg_w, panel_path=panel, vcfg=vcfg, paths=paths, save_oof=True)
    oof0 = load_oof("base0", "A", paths=paths)
    oofw = load_oof("blend", "A", paths=paths)
    merged = oof0[["site_id", "ts", "pred_bag"]].merge(
        oofw[["site_id", "ts", "pred_bag"]], on=["site_id", "ts"], suffixes=("_0", "_w")
    )
    night = merged.loc[~merged["ts"].dt.hour.isin([11, 12, 13])]
    assert len(night) > 0
    assert bool((merged["pred_bag_w"] == 0.0).loc[merged.index.isin(night.index)].all())
    day = merged.loc[merged["ts"].dt.hour.isin([11, 12, 13])].copy()
    assert len(day) > 0
    assert bool(((day["pred_bag_w"] - 0.4 * day["pred_bag_0"]) >= -1e-9).all())
    assert bool((day["pred_bag_w"] != day["pred_bag_0"]).any())
    day["reg"] = (day["pred_bag_w"] - 0.4 * day["pred_bag_0"]) / 0.6
    geo = runner_mod._site_geo(pd.read_parquet(panel))
    units = site_units(geo, 2, 0, cfg_w.model.regional)
    day["unit"] = day["site_id"].map({k: int(v) for k, v in units.items()})
    day["ratio"] = day["reg"] / 100.0
    for (_, _), group in day.groupby(["unit", "ts"]):
        vals = group["ratio"].to_numpy()
        assert vals.max() - vals.min() <= 1e-9 * max(1.0, abs(vals).max())


def test_run_validation_records_regional_config(tmp_path: Path) -> None:
    panel = _make_panel(tmp_path)
    paths = _paths(tmp_path)
    run_validation(_regional_cfg("reg"), panel_path=panel, vcfg=_vcfg(), paths=paths)
    from src.oibc.evaluate.registry import read_records

    rec = next(r for r in read_records(paths.registry) if r.run_id == "reg")
    assert rec.config["model"]["regional"]["weight"] == 0.6
    assert rec.metrics and all(
        np.isfinite(m["regional_sec"]) and m["regional_sec"] >= 0 for m in rec.metrics.values()
    )


def test_config_diff_reports_regional(tmp_path: Path) -> None:
    import copy

    panel = _make_panel(tmp_path)
    paths = _paths(tmp_path)
    run_validation(_cfg("base"), panel_path=panel, vcfg=_vcfg(), paths=paths)
    from src.oibc.evaluate.registry import read_records

    record = next(r for r in read_records(paths.registry) if r.run_id == "base")
    config = copy.deepcopy(record.config)
    config["model"].pop("regional")
    legacy = record.model_copy(update={"config": config})
    assert "model.regional" in runner_mod._config_diff_keys(legacy, _regional_cfg("cand"))


def test_forecast_eval_regional_contract(tmp_path: Path) -> None:
    panel = _make_panel(tmp_path)
    spec = _spec()
    out = forecast_eval(_regional_cfg("r1"), panel_path=panel, spec=spec)
    assert len(out) == 2 * spec.n_eval_hours
    assert np.isfinite(out["forecast_kwh"].to_numpy()).all() and (out["forecast_kwh"] >= 0).all()
    eval_ts = pd.date_range(spec.eval_start, spec.eval_end, freq="h").as_unit("us")
    for group in ("G1", "G2"):
        got = pd.DatetimeIndex(out.loc[out["group"] == group, "ts"]).sort_values()
        assert got.equals(eval_ts)
    base = forecast_eval(_cfg("r0"), panel_path=panel, spec=spec)
    merged = out.merge(base, on=["group", "ts"], suffixes=("_w", "_0"))
    assert bool((merged["forecast_kwh_w"] != merged["forecast_kwh_0"]).any())


def test_forecast_eval_no_future_leak_regional(tmp_path: Path) -> None:
    panel = _make_panel(tmp_path)
    cfg = _regional_cfg("r1")
    before = forecast_eval(cfg, panel_path=panel, spec=_spec())
    gen_path = tmp_path / "canonical" / "generation.parquet"
    gen = pd.read_parquet(gen_path)
    gen["energy"] = gen["energy"] * 100.0
    gen.to_parquet(gen_path, index=False)
    after = forecast_eval(cfg, panel_path=panel, spec=_spec())
    pd.testing.assert_frame_equal(before, after)


def test_site_geo_inconsistent_rejected() -> None:
    frame = pd.DataFrame(
        {
            "site_id": ["a", "a", "b"],
            "lat": [1.0, 2.0, 1.0],
            "lon": [0.0, 0.0, 0.0],
            "capacity_kw": [1.0, 1.0, 1.0],
        }
    )
    with pytest.raises(RunnerError):
        runner_mod._site_geo(frame)


@pytest.mark.parametrize("defect", ["missing_col", "empty", "null_id", "empty_id", "nonfinite_lat", "nonpositive_cap"])
def test_site_geo_guards(defect: str) -> None:
    base = pd.DataFrame(
        {
            "site_id": ["a", "a", "b"],
            "lat": [1.0, 1.0, 2.0],
            "lon": [0.0, 0.0, 0.0],
            "capacity_kw": [1.0, 1.0, 1.0],
        }
    )
    if defect == "missing_col":
        frame = base.drop(columns=["lat"])
    elif defect == "empty":
        frame = base.iloc[:0]
    elif defect == "null_id":
        frame = base.copy()
        frame.loc[0, "site_id"] = None
    elif defect == "empty_id":
        frame = base.copy()
        frame.loc[0, "site_id"] = ""
    elif defect == "nonfinite_lat":
        frame = base.copy()
        frame.loc[0, "lat"] = float("inf")
    else:
        frame = base.copy()
        frame.loc[0, "capacity_kw"] = 0.0
    with pytest.raises(RunnerError):
        runner_mod._site_geo(frame)


def _fleet_panel(path: Path, constant: float | None = None) -> Path:
    panel = pd.read_parquet(path)
    ts = pd.DatetimeIndex(panel["ts"])
    if constant is None:
        vals = ((ts.dayofyear * 7 + ts.hour * 13) % 80 + 10).astype(float).to_numpy()
    else:
        vals = np.full(len(panel), float(constant))
    panel["wa_cloud_fleet"] = vals
    if constant is None:
        panel["wb_cloud_fleet"] = np.clip(vals + 5.0, 0.0, 100.0)
    else:
        panel["wb_cloud_fleet"] = vals
    panel.to_parquet(path, index=False)
    return path


_CAL_PANEL_BYTES: bytes | None = None
_CAL_REGISTRY_TEXT: str = ""
_CAL_TPL_PANEL_PATH: str = ""
_CAL_RUN_FILES: dict[str, bytes] = {}


def _cal_artifacts(tmp_path: Path, run_id: str, *, constant: float | None = None):
    global _CAL_PANEL_BYTES, _CAL_REGISTRY_TEXT, _CAL_TPL_PANEL_PATH
    from src.oibc.train.runner import run_validation

    panel = _make_panel(tmp_path)
    paths = _paths(tmp_path)
    vcfg = _vcfg()
    if constant is not None:
        _fleet_panel(panel, constant)
        run_validation(_cfg(run_id), panel_path=panel, vcfg=vcfg, paths=paths, save_oof=True)
        return panel, paths, vcfg
    if _CAL_PANEL_BYTES is None:
        _fleet_panel(panel, None)
        run_validation(_cfg("__cal_tpl__"), panel_path=panel, vcfg=vcfg, paths=paths, save_oof=True)
        _CAL_PANEL_BYTES = panel.read_bytes()
        _CAL_REGISTRY_TEXT = paths.registry.read_text(encoding="utf-8")
        _CAL_TPL_PANEL_PATH = str(panel.resolve())
        tpl_dir = paths.runs / "__cal_tpl__"
        for item in tpl_dir.iterdir():
            _CAL_RUN_FILES[item.name] = item.read_bytes()
            item.unlink()
        tpl_dir.rmdir()
    else:
        panel.write_bytes(_CAL_PANEL_BYTES)
    run_dir = paths.runs / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    for name, data in _CAL_RUN_FILES.items():
        (run_dir / name).write_bytes(data)
    reg_text = _CAL_REGISTRY_TEXT.replace("__cal_tpl__", run_id).replace(_CAL_TPL_PANEL_PATH, str(panel.resolve()))
    paths.registry.write_text(reg_text, encoding="utf-8")
    return panel, paths, vcfg


def _cal_run(tmp_path: Path, run_id: str = "cal", *, constant: float | None = None):
    from src.oibc.train.runner import calibration_report

    panel, paths, vcfg = _cal_artifacts(tmp_path, run_id, constant=constant)
    return panel, paths, vcfg, calibration_report(run_id, paths=paths, vcfg=vcfg)


def test_calibration_report_schema_and_pairs(tmp_path: Path) -> None:
    import json

    _, _, _, report = _cal_run(tmp_path)
    assert set(report) >= {"run_id", "panel_sha256", "composition_id", "composition_sha256", "config",
                           "cross_fit", "pooled", "production", "decision", "reasons"}
    pairs = report["cross_fit"]
    assert isinstance(pairs, list) and len(pairs) == 2
    by_pair = {(p["fit"], p["apply"]): p for p in pairs}
    assert by_pair[("A", "B")]["causal"] is True
    assert by_pair[("B", "A")]["causal"] is False
    for p in pairs:
        for key in ("W_base", "W_cal", "rel_delta", "ci_low", "ci_high", "a", "b"):
            assert np.isfinite(p[key])
        assert isinstance(p["quarter_rel"], dict)
    json.dumps(report, allow_nan=False)
    assert report["decision"] in {"adopt", "reject"}


def test_calibration_report_cross_fit_uses_other_fold(tmp_path: Path) -> None:
    from src.oibc.train import runner as cal_mod

    panel, paths, vcfg, before = _cal_run(tmp_path, run_id="calx")
    import pandas as pd

    oof_b = pd.read_parquet(paths.runs / "calx" / "B_oof.parquet")
    oof_b["pred_bag"] = oof_b["pred_bag"] * 1.5 + 1.0
    oof_b.to_parquet(paths.runs / "calx" / "B_oof.parquet", index=False)
    after = cal_mod.calibration_report("calx", paths=paths, vcfg=vcfg)
    before_ab = next(p for p in before["cross_fit"] if (p["fit"], p["apply"]) == ("A", "B"))
    after_ab = next(p for p in after["cross_fit"] if (p["fit"], p["apply"]) == ("A", "B"))
    assert (before_ab["a"], before_ab["b"]) == (after_ab["a"], after_ab["b"])
    assert (after_ab["W_base"], after_ab["W_cal"]) != (before_ab["W_base"], before_ab["W_cal"])


def test_calibration_report_identity_on_unbiased(tmp_path: Path) -> None:
    import pandas as pd

    from src.oibc.train import runner as cal_mod

    _, paths, vcfg = _cal_artifacts(tmp_path, "cali")
    for fold in ("A", "B"):
        path = paths.runs / "cali" / f"{fold}_oof.parquet"
        oof = pd.read_parquet(path)
        oof["pred_bag"] = oof["actual_kwh"]
        oof.to_parquet(path, index=False)
    report = cal_mod.calibration_report("cali", paths=paths, vcfg=vcfg)
    for p in report["cross_fit"]:
        assert (p["a"], p["b"]) == (1.0, 0.0)
        assert p["rel_delta"] == 0.0
    assert report["decision"] == "reject"
    assert report["reasons"]


def test_calibration_report_requires_oof_and_panel_hash(tmp_path: Path) -> None:
    from src.oibc.train import runner as cal_mod

    panel, paths, vcfg = _cal_artifacts(tmp_path, "hashok")
    for f in (paths.runs / "hashok").glob("*_oof.parquet"):
        f.unlink()
    with pytest.raises(RunnerError):
        cal_mod.calibration_report("hashok", paths=paths, vcfg=vcfg)
    panel, paths, vcfg = _cal_artifacts(tmp_path, "hashbad")
    frame = pd.read_parquet(panel)
    frame.loc[0, "doy_sin"] = 0.9
    frame.to_parquet(panel, index=False)
    with pytest.raises(RunnerError):
        cal_mod.calibration_report("hashbad", paths=paths, vcfg=vcfg)


def test_calibration_report_decision_rule(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.oibc.models.calibrate import CloudCalibration
    from src.oibc.train import runner as cal_mod

    _, paths, vcfg = _cal_artifacts(tmp_path, "cald")
    monkeypatch.setattr(
        cal_mod, "fit_cloud_calibration",
        lambda hourly, index, cfg, spec: CloudCalibration(a=1.0, b=0.0, at_boundary=False, n_fit_hours=1),
    )
    report = cal_mod.calibration_report("cald", paths=paths, vcfg=vcfg)
    assert report["decision"] == "reject"
    assert any("pair_not_improved" in r for r in report["reasons"])


def test_forecast_eval_calibration_scales_daylight(tmp_path: Path) -> None:
    from src.oibc.config import CalibrationConfig
    from src.oibc.models.calibrate import CloudCalibration

    panel = _fleet_panel(_make_panel(tmp_path), 50.0)
    cfg = _cfg("calscale")
    base = forecast_eval(cfg, panel_path=panel, spec=_spec())
    cal = forecast_eval(
        cfg, panel_path=panel, spec=_spec(),
        calibration=CloudCalibration(a=0.98, b=0.08, at_boundary=False, n_fit_hours=1),
        calibration_cfg=CalibrationConfig(),
    )
    np.testing.assert_allclose(
        cal["forecast_kwh"].to_numpy(), base["forecast_kwh"].to_numpy() * 1.02, rtol=1e-12
    )
    assert (cal["forecast_kwh"] >= 0).all() and np.isfinite(cal["forecast_kwh"].to_numpy()).all()


def test_forecast_eval_calibration_none_identical(tmp_path: Path) -> None:
    panel = _fleet_panel(_make_panel(tmp_path), 50.0)
    cfg = _cfg("calnone")
    first = forecast_eval(cfg, panel_path=panel, spec=_spec())
    second = forecast_eval(cfg, panel_path=panel, spec=_spec())
    pd.testing.assert_frame_equal(first, second, check_exact=True)


def test_forecast_eval_calibration_args_paired(tmp_path: Path) -> None:
    from src.oibc.config import CalibrationConfig
    from src.oibc.models.calibrate import CloudCalibration

    panel = _fleet_panel(_make_panel(tmp_path), 50.0)
    cfg = _cfg("calpair")
    with pytest.raises(RunnerError):
        forecast_eval(
            cfg, panel_path=panel, spec=_spec(),
            calibration=CloudCalibration(a=1.0, b=0.0, at_boundary=False, n_fit_hours=1),
        )
    with pytest.raises(RunnerError):
        forecast_eval(cfg, panel_path=panel, spec=_spec(), calibration_cfg=CalibrationConfig())


def test_forecast_eval_missing_cloud_rejected(tmp_path: Path) -> None:
    from src.oibc.config import CalibrationConfig
    from src.oibc.models.calibrate import CloudCalibration

    panel = _fleet_panel(_make_panel(tmp_path), 50.0)
    frame = pd.read_parquet(panel)
    target = frame.loc[frame["period"] == "eval", "ts"].iloc[0]
    frame.loc[frame["ts"] == target, ["wa_cloud_fleet", "wb_cloud_fleet"]] = float("nan")
    frame.to_parquet(panel, index=False)
    with pytest.raises(RunnerError):
        forecast_eval(
            _cfg("calmiss"), panel_path=panel, spec=_spec(),
            calibration=CloudCalibration(a=0.98, b=0.08, at_boundary=False, n_fit_hours=1),
            calibration_cfg=CalibrationConfig(),
        )


def test_forecast_eval_missing_calibration_columns(tmp_path: Path) -> None:
    from src.oibc.config import CalibrationConfig
    from src.oibc.models.calibrate import CloudCalibration

    panel = _make_panel(tmp_path)
    with pytest.raises(RunnerError, match="calibration columns"):
        forecast_eval(
            _cfg("cc"), panel_path=panel, spec=_spec(),
            calibration=CloudCalibration(a=1.0, b=0.0, at_boundary=False, n_fit_hours=1),
            calibration_cfg=CalibrationConfig(),
        )


def test_forecast_eval_bad_cloud_rejected(tmp_path: Path) -> None:
    from src.oibc.config import CalibrationConfig
    from src.oibc.models.calibrate import CloudCalibration

    panel = _fleet_panel(_make_panel(tmp_path), 150.0)
    with pytest.raises(RunnerError):
        forecast_eval(
            _cfg("cb"), panel_path=panel, spec=_spec(),
            calibration=CloudCalibration(a=1.0, b=0.0, at_boundary=False, n_fit_hours=1),
            calibration_cfg=CalibrationConfig(),
        )


def test_forecast_eval_bad_calibration_rejected(tmp_path: Path) -> None:
    from src.oibc.config import CalibrationConfig
    from src.oibc.models.calibrate import CloudCalibration

    panel = _fleet_panel(_make_panel(tmp_path), 50.0)
    with pytest.raises(RunnerError):
        forecast_eval(
            _cfg("cn"), panel_path=panel, spec=_spec(),
            calibration=CloudCalibration(a=float("nan"), b=0.0, at_boundary=False, n_fit_hours=1),
            calibration_cfg=CalibrationConfig(),
        )


def _hourly_pair(base_pen: list[float], cal_pen: list[float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    ts = pd.DatetimeIndex(["2023-01-10 11:00", "2023-01-11 11:00"], tz=KST).as_unit("us")
    opd = pd.DatetimeIndex(["2023-01-10", "2023-01-11"], tz=KST).as_unit("us")
    kw = {"group": pd.Series(["G1", "G1"], dtype="string"),
          "ts": pd.Series(ts, dtype=_DT), "op_date": pd.Series(opd, dtype=_DT)}
    base = pd.DataFrame({**kw, "penalty": pd.Series(base_pen, dtype="float64")})
    cal = pd.DataFrame({**kw, "penalty": pd.Series(cal_pen, dtype="float64")})
    return base, cal


def test_calibration_helper_guards(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.oibc.train import runner as cal_mod

    base, cal = _hourly_pair([1.0, 3.0], [0.5, 1.5])
    assert cal_mod._quarter_rel_calibration(base, cal) == {"2023Q1": pytest.approx(-0.5)}
    shifted = cal.copy()
    shifted["ts"] = shifted["ts"] + pd.Timedelta(hours=1)
    with pytest.raises(RunnerError, match="alignment"):
        cal_mod._quarter_rel_calibration(base, shifted)
    _, inf_cal = _hourly_pair([1.0, 3.0], [float("inf"), 1.0])
    with pytest.raises(RunnerError, match="quarter"):
        cal_mod._quarter_rel_calibration(base, inf_cal)
    assert cal_mod._rel_delta(0.0, 0.0) == 0.0
    with pytest.raises(RunnerError, match="baseline"):
        cal_mod._rel_delta(1.0, 0.0)
    with pytest.raises(RunnerError, match="rel_delta"):
        cal_mod._rel_delta(float("inf"), 1.0)
    with pytest.raises(RunnerError):
        cal_mod._pair_ci(np.array([1.0]), np.array([1.0, 2.0]), 1.0, 2.0, _vcfg())
    monkeypatch.setattr(cal_mod, "block_bootstrap_ci", lambda *a, **k: (float("nan"), 0.0))
    with pytest.raises(RunnerError, match="CI"):
        cal_mod._pair_ci(np.array([1.0, 2.0]), np.array([1.0, 2.0]), 1.0, 2.0, _vcfg())


def test_calibration_report_invalid_spec_rejected(tmp_path: Path) -> None:
    from datetime import UTC, datetime

    from src.oibc.evaluate.registry import ExperimentRecord, append_record
    from src.oibc.train import runner as cal_mod

    _make_panel(tmp_path)
    paths = _paths(tmp_path)
    rec = ExperimentRecord(
        run_id="ghost", created_at=datetime(2026, 1, 1, tzinfo=UTC), hypothesis="h",
        config={"competition": {"broken": True}}, git_commit=None, seeds=(0, 1),
        folds=("A", "B"), metrics={}, comparison={},
    )
    append_record(rec, paths.registry)
    with pytest.raises(RunnerError, match="competition spec"):
        cal_mod.calibration_report("ghost", paths=paths, vcfg=_vcfg())


def test_calibration_report_missing_provenance_rejected(tmp_path: Path) -> None:
    from datetime import UTC, datetime

    from src.oibc.evaluate.registry import ExperimentRecord, append_record
    from src.oibc.train import runner as cal_mod

    _make_panel(tmp_path)
    paths = _paths(tmp_path)
    rec = ExperimentRecord(
        run_id="ghost", created_at=datetime(2026, 1, 1, tzinfo=UTC), hypothesis="h",
        config={"competition": _spec().model_dump(mode="json")},
        git_commit=None, seeds=(0, 1), folds=("A", "B"), metrics={}, comparison={},
    )
    append_record(rec, paths.registry)
    with pytest.raises(RunnerError, match="provenance"):
        cal_mod.calibration_report("ghost", paths=paths, vcfg=_vcfg())


def test_calibration_report_single_fold_rejected(tmp_path: Path) -> None:
    from src.oibc.train import runner as cal_mod

    panel = _fleet_panel(_make_panel(tmp_path))
    paths = _paths(tmp_path)
    solo = ValidationConfig(
        folds=_vcfg().folds[:1], embargo_days=1, min_warm_days=1, seeds=(0, 1), bootstrap_n=100
    )
    cal_mod.run_validation(_cfg("solo"), panel_path=panel, vcfg=solo, paths=paths, save_oof=True)
    with pytest.raises(RunnerError, match=">= 2 folds"):
        cal_mod.calibration_report("solo", paths=paths, vcfg=solo)


def test_calibration_report_missing_panel_file(tmp_path: Path) -> None:
    from src.oibc.train import runner as cal_mod

    panel, paths, vcfg = _cal_artifacts(tmp_path, "calgone")
    panel.unlink()
    with pytest.raises(RunnerError, match="unreadable"):
        cal_mod.calibration_report("calgone", paths=paths, vcfg=vcfg)


def test_calibration_report_panel_read_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.oibc.train import runner as cal_mod

    panel, paths, vcfg = _cal_artifacts(tmp_path, "calread")
    real_read = pd.read_parquet
    monkeypatch.setattr(
        pd, "read_parquet",
        lambda path, *a, **k: (_ for _ in ()).throw(OSError("disk gone"))
        if str(path) == str(panel) else real_read(path, *a, **k),
    )
    with pytest.raises(RunnerError, match="unreadable"):
        cal_mod.calibration_report("calread", paths=paths, vcfg=vcfg)


def test_calibration_report_panel_contracts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.oibc.train import runner as cal_mod

    panel, paths, vcfg = _cal_artifacts(tmp_path, "calpanel")
    full = pd.read_parquet(panel)

    def _frame(kind: str) -> pd.DataFrame:
        if kind == "nocol":
            return full.drop(columns=["wa_cloud_fleet"])
        if kind == "empty":
            out = full.copy()
            out["period"] = "eval"
            return out
        out = full.copy()
        out["wa_cloud_fleet"] = 150.0
        return out

    real_read = pd.read_parquet
    state = {"kind": "nocol"}

    def _selective(path: object, *a: object, **k: object) -> pd.DataFrame:
        if str(path) == str(panel):
            return _frame(state["kind"])
        return real_read(path, *a, **k)  # type: ignore[arg-type]

    monkeypatch.setattr(pd, "read_parquet", _selective)
    with pytest.raises(RunnerError, match="columns"):
        cal_mod.calibration_report("calpanel", paths=paths, vcfg=vcfg)
    state["kind"] = "empty"
    with pytest.raises(RunnerError, match="train period"):
        cal_mod.calibration_report("calpanel", paths=paths, vcfg=vcfg)
    state["kind"] = "range"
    with pytest.raises(RunnerError, match="within"):
        cal_mod.calibration_report("calpanel", paths=paths, vcfg=vcfg)


def test_calibration_report_base_settle_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.oibc.evaluate.settlement import SettlementError
    from src.oibc.train import runner as cal_mod

    _, paths, vcfg = _cal_artifacts(tmp_path, "calbase")

    def _boom(*a: object, **k: object) -> object:
        raise SettlementError("boom")

    monkeypatch.setattr(cal_mod, "aggregate_groups", _boom)
    with pytest.raises(RunnerError, match="fold A"):
        cal_mod.calibration_report("calbase", paths=paths, vcfg=vcfg)


def test_calibration_report_fit_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.oibc.models.calibrate import CalibrationError
    from src.oibc.train import runner as cal_mod

    _, paths, vcfg = _cal_artifacts(tmp_path, "calfit")

    def _boom(*a: object, **k: object) -> object:
        raise CalibrationError("boom")

    monkeypatch.setattr(cal_mod, "fit_cloud_calibration", _boom)
    with pytest.raises(RunnerError, match="boom"):
        cal_mod.calibration_report("calfit", paths=paths, vcfg=vcfg)


def test_calibration_report_nonfinite_scales(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.oibc.train import runner as cal_mod

    _, paths, vcfg = _cal_artifacts(tmp_path, "calscale")
    monkeypatch.setattr(cal_mod, "scale_factors", lambda ts, index, cal: np.full(len(ts), np.inf))
    with pytest.raises(RunnerError, match="scale"):
        cal_mod.calibration_report("calscale", paths=paths, vcfg=vcfg)


def test_calibration_report_apply_settle_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.oibc.evaluate.settlement import SettlementError
    from src.oibc.train import runner as cal_mod

    _, paths, vcfg = _cal_artifacts(tmp_path, "calapply")
    real_settle = cal_mod.settle
    calls = {"n": 0}

    def _counting(frame: pd.DataFrame, smp: pd.DataFrame, **k: object) -> object:
        calls["n"] += 1
        if calls["n"] == 3:
            raise SettlementError("boom")
        return real_settle(frame, smp, **k)

    monkeypatch.setattr(cal_mod, "settle", _counting)
    with pytest.raises(RunnerError, match="fold A"):
        cal_mod.calibration_report("calapply", paths=paths, vcfg=vcfg)


def test_calibration_report_daily_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.oibc.train import runner as cal_mod

    _, paths, vcfg = _cal_artifacts(tmp_path, "caldaily")
    real_daily = cal_mod.daily_totals
    calls = {"n": 0}

    def _truncating(result: object, **k: object) -> pd.DataFrame:
        calls["n"] += 1
        out = real_daily(result, **k)  # type: ignore[arg-type]
        return out.iloc[:1] if calls["n"] == 3 else out

    monkeypatch.setattr(cal_mod, "daily_totals", _truncating)
    with pytest.raises(RunnerError, match="daily length"):
        cal_mod.calibration_report("caldaily", paths=paths, vcfg=vcfg)


def test_calibration_report_production_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.oibc.evaluate.settlement import SettlementError
    from src.oibc.models.calibrate import CalibrationError
    from src.oibc.train import runner as cal_mod

    def _run_with(patch: dict[str, object], run_id: str, match: str) -> None:
        _, paths, vcfg = _cal_artifacts(tmp_path, run_id)
        with monkeypatch.context() as sub:
            for name, func in patch.items():
                sub.setattr(cal_mod, name, func)
            with pytest.raises(RunnerError, match=match):
                cal_mod.calibration_report(run_id, paths=paths, vcfg=vcfg)

    real_fit, real_scale, real_settle = cal_mod.fit_cloud_calibration, cal_mod.scale_factors, cal_mod.settle
    for run_id, name, fail_at, match in [
        ("calp1", "fit_cloud_calibration", 3, "boom-fit"),
        ("calp2", "scale_factors", 3, "boom-scale"),
        ("calp3", "settle", 5, "boom-settle"),
    ]:
        calls = {"n": 0}
        if name == "fit_cloud_calibration":
            def _fit(hourly, index, cfg, spec, _r=real_fit, _c=calls, _n=fail_at):
                _c["n"] += 1
                if _c["n"] == _n:
                    raise CalibrationError("boom-fit")
                return _r(hourly, index, cfg, spec)
            _run_with({"fit_cloud_calibration": _fit}, run_id, match)
        elif name == "scale_factors":
            def _scale(ts, index, cal, _r=real_scale, _c=calls, _n=fail_at):
                _c["n"] += 1
                if _c["n"] == _n:
                    raise CalibrationError("boom-scale")
                return _r(ts, index, cal)
            _run_with({"scale_factors": _scale}, run_id, match)
        else:
            def _settle(frame, smp, _r=real_settle, _c=calls, _n=fail_at, **k):
                _c["n"] += 1
                if _c["n"] == _n:
                    raise SettlementError("boom-settle")
                return _r(frame, smp, **k)
            _run_with({"settle": _settle}, run_id, match)


def test_calibration_report_serialization_guard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json as _json

    from src.oibc.train import runner as cal_mod

    _, paths, vcfg = _cal_artifacts(tmp_path, "caljson")

    def _boom(*a: object, **k: object) -> str:
        raise ValueError("boom")

    monkeypatch.setattr(cal_mod.json, "dumps", _boom)
    with pytest.raises(RunnerError, match="serializable"):
        cal_mod.calibration_report("caljson", paths=paths, vcfg=vcfg)
    monkeypatch.setattr(cal_mod.json, "dumps", _json.dumps)
