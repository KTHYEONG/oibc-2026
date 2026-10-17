import numpy as np
import pandas as pd
import pytest

from src.oibc.config import ModelConfig, RegionalConfig
from src.oibc.models.gbm import bag
from src.oibc.models.regional import (
    RegionalError,
    _panel_matrix,
    blend_seed_preds,
    fit_predict_regional,
    site_units,
    unit_centroids,
)

FEATURES = ["f0", "f1", "f2", "f3"]

TS = pd.date_range("2024-01-01 01:00", periods=3, freq="h", tz="Asia/Seoul")


def _regional(**kwargs: object) -> RegionalConfig:
    base: dict[str, object] = {
        "weight": 0.6,
        "units": ((1, 0), (2, 0)),
        "sd_vars": ("f0",),
        "excluded_vars": (),
        "n_rounds": 10,
        "min_child_samples": 2,
    }
    base.update(kwargs)
    return RegionalConfig(**base)  # type: ignore[arg-type]


def _cfg(**kwargs: object) -> ModelConfig:
    base: dict[str, object] = {"n_jobs": 1, "seed_workers": 1, "seeds": (0, 1, 2)}
    base.update(kwargs)
    if "regional" not in base:
        base["regional"] = _regional()
    return ModelConfig(**base)  # type: ignore[arg-type]


def _geo(n: int = 6) -> pd.DataFrame:
    half = n // 2
    lat = [36.0 + 0.01 * i for i in range(half)] + [37.5 + 0.01 * i for i in range(n - half)]
    lon = [127.0 + 0.01 * i for i in range(half)] + [129.0 + 0.01 * i for i in range(n - half)]
    return pd.DataFrame({
        "site_id": [f"s{i}" for i in range(n)],
        "lat": lat,
        "lon": lon,
        "capacity_kw": [100.0 + 50.0 * (i % 3) for i in range(n)],
    })


def _frame(geo: pd.DataFrame, ts: pd.DatetimeIndex = TS, seed: int = 0, labeled: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = [(s, t) for s in geo["site_id"] for t in ts]
    n = len(rows)
    cap_of = dict(zip(geo["site_id"], geo["capacity_kw"], strict=True))
    data: dict[str, object] = {
        "site_id": [s for s, _ in rows],
        "ts": [t for _, t in rows],
        "capacity_kw": [cap_of[s] for s, _ in rows],
        "f0": rng.normal(loc=20.0, scale=5.0, size=n),
        "f1": rng.normal(size=n),
        "f2": rng.normal(size=n),
        "f3": rng.normal(size=n),
        "smp": np.full(n, 120.0),
        "k": np.ones(n),
    }
    if labeled:
        data["energy"] = np.abs(rng.normal(loc=60.0, scale=20.0, size=n))
    return pd.DataFrame(data)


def test_site_units_single_unit() -> None:
    geo = _geo()
    out = site_units(geo, 1, 0, _regional())
    assert out.dtype == np.int64
    assert (out.to_numpy() == 0).all()
    assert out.index.tolist() == geo["site_id"].tolist()


def test_site_units_deterministic_and_complete() -> None:
    geo = _geo()
    cfg = _regional()
    first = site_units(geo, 2, 0, cfg)
    second = site_units(geo, 2, 0, cfg)
    assert first.equals(second)
    assert set(first.to_numpy().tolist()) == {0, 1}
    assert first.iloc[0] == first.iloc[1] == first.iloc[2]
    assert first.iloc[3] == first.iloc[4] == first.iloc[5]
    assert first.iloc[0] != first.iloc[3]


def test_site_units_too_many_units_rejected() -> None:
    with pytest.raises(RegionalError):
        site_units(_geo(6), 7, 0, _regional())


@pytest.mark.parametrize("kind", ["duplicate", "capacity", "lat"])
def test_site_geo_contract_rejected(kind: str) -> None:
    geo = _geo()
    if kind == "duplicate":
        geo.loc[1, "site_id"] = geo.loc[0, "site_id"]
    elif kind == "capacity":
        geo.loc[0, "capacity_kw"] = 0.0
    else:
        geo.loc[0, "lat"] = np.nan
    with pytest.raises(RegionalError):
        site_units(geo, 1, 0, _regional())


def test_site_geo_columns_contract_rejected() -> None:
    geo = _geo()
    with pytest.raises(RegionalError):
        site_units(pd.concat([geo, geo[["lat"]]], axis=1), 1, 0, _regional())
    with pytest.raises(RegionalError):
        site_units(geo.drop(columns=["lon"]), 1, 0, _regional())
    bad = geo.copy()
    bad["lat"] = ["x"] * len(geo)
    with pytest.raises(RegionalError):
        site_units(bad, 1, 0, _regional())


@pytest.mark.parametrize("kwargs", [{"n_units": 0}, {"n_units": -2}, {"kmeans_seed": -1}])
def test_site_units_bad_counts_rejected(kwargs: dict) -> None:
    args: dict[str, int] = {"n_units": 2, "kmeans_seed": 0}
    args.update(kwargs)  # type: ignore[typeddict-item]
    with pytest.raises(RegionalError):
        site_units(_geo(), args["n_units"], args["kmeans_seed"], _regional())


def test_unit_centroids_contract_rejected() -> None:
    geo = _geo()
    units = site_units(geo, 2, 0, _regional())
    with pytest.raises(RegionalError):
        unit_centroids(geo, pd.Series([0, 1], index=pd.Index(["s0", "s0"])))
    with pytest.raises(RegionalError):
        unit_centroids(geo, units.drop("s0"))
    with pytest.raises(RegionalError):
        unit_centroids(geo, units.astype("float64"))


def test_aggregate_weighted_mean_ignores_nan() -> None:
    geo = pd.DataFrame({
        "site_id": ["a", "b"],
        "lat": [36.0, 36.0],
        "lon": [127.0, 127.0],
        "capacity_kw": [100.0, 300.0],
    })
    ts = pd.DatetimeIndex([TS[0]])
    frame = pd.DataFrame({
        "site_id": ["a", "b"],
        "ts": [ts[0], ts[0]],
        "capacity_kw": [100.0, 300.0],
        "f0": [1.0, np.nan],
        "f1": [1.0, 3.0],
    })
    units = site_units(geo, 1, 0, _regional())
    centroids = unit_centroids(geo, units)
    mat, cols, _, _, _ = _panel_matrix(frame, ["f0", "f1"], ("f0",), {"a": 0, "b": 0}, centroids)
    assert cols == ["f0", "f1", "f0_sd", "unit_lat", "unit_lon"]
    assert mat.shape == (1, 5)
    assert mat[0, 0] == pytest.approx(1.0)
    assert mat[0, 1] == pytest.approx(2.5)
    assert mat[0, 2] == pytest.approx(0.0)


def test_predict_row_scaling_conservation() -> None:
    geo = _geo()
    cfg = _cfg(regional=_regional(units=((2, 0),)))
    train = _frame(geo, labeled=True)
    test = _frame(geo, seed=7, labeled=False)
    out = fit_predict_regional(train, test, FEATURES, geo, cfg)
    cf = out[0] / test["capacity_kw"].to_numpy()
    units = site_units(geo, 2, 0, cfg.regional)
    key = list(zip(test["site_id"].map(units).tolist(), test["ts"].tolist(), strict=True))
    by_key: dict[object, list[float]] = {}
    for k, v in zip(key, cf.tolist(), strict=True):
        by_key.setdefault(k, []).append(v)
    for vals in by_key.values():
        assert np.allclose(vals, vals[0], rtol=1e-12)
    cap = test["capacity_kw"].to_numpy()
    for k, vals in by_key.items():
        mask = np.array([kk == k for kk in key])
        np.testing.assert_allclose(out[0][mask].sum(), vals[0] * cap[mask].sum(), rtol=1e-12)


def test_test_row_permutation_equivariant() -> None:
    geo = _geo()
    cfg = _cfg()
    train = _frame(geo, labeled=True)
    test = _frame(geo, seed=7, labeled=False)
    base = fit_predict_regional(train, test, FEATURES, geo, cfg)
    perm = np.random.default_rng(3).permutation(len(test))
    shuffled = test.iloc[perm].reset_index(drop=True)
    out = fit_predict_regional(train, shuffled, FEATURES, geo, cfg)
    assert np.array_equal(out, base[:, perm])


def test_train_row_order_invariant() -> None:
    geo = _geo()
    cfg = _cfg()
    train = _frame(geo, labeled=True)
    test = _frame(geo, seed=7, labeled=False)
    base = fit_predict_regional(train, test, FEATURES, geo, cfg)
    perm = np.random.default_rng(5).permutation(len(train))
    shuffled = train.iloc[perm].reset_index(drop=True)
    assert np.array_equal(fit_predict_regional(shuffled, test, FEATURES, geo, cfg), base)


def test_cross_section_causal_invariance() -> None:
    geo = _geo()
    cfg = _cfg()
    train = _frame(geo, labeled=True)
    test = _frame(geo, seed=7, labeled=False)
    base = fit_predict_regional(train, test, FEATURES, geo, cfg)
    broken = test.copy()
    broken.loc[broken["ts"] == TS[2], FEATURES] = 1e6
    out = fit_predict_regional(train, broken, FEATURES, geo, cfg)
    keep = (test["ts"] != TS[2]).to_numpy()
    assert np.array_equal(out[:, keep], base[:, keep])


def test_cold_start_site_predicted() -> None:
    geo = _geo(8)
    cfg = _cfg()
    train_geo = geo.iloc[:6]
    train = _frame(train_geo, labeled=True)
    test = _frame(geo, seed=7, labeled=False)
    out = fit_predict_regional(train, test, FEATURES, geo, cfg)
    assert out.shape == (3, len(test))
    assert np.isfinite(out).all() and (out >= 0).all()


def test_test_does_not_require_labels() -> None:
    geo = _geo()
    cfg = _cfg()
    train = _frame(geo, labeled=True)
    test = _frame(geo, seed=7, labeled=False).drop(columns=["smp", "k"])
    assert "energy" not in test.columns
    out = fit_predict_regional(train, test, FEATURES, geo, cfg)
    assert np.isfinite(out).all()


@pytest.mark.parametrize("kind", ["energy_nan", "empty_train", "smp_conflict", "inf_feature"])
def test_train_contract_fail_closed(kind: str) -> None:
    geo = _geo()
    cfg = _cfg()
    train = _frame(geo, labeled=True)
    test = _frame(geo, seed=7, labeled=False)
    if kind == "energy_nan":
        train.loc[0, "energy"] = np.nan
    elif kind == "empty_train":
        train = train.iloc[0:0]
    elif kind == "smp_conflict":
        cfg = _cfg(regional=_regional(units=((1, 0),)))
        mask = train["ts"] == TS[0]
        train.loc[mask, "smp"] = [100.0 + (i % 2) for i in range(int(mask.sum()))]
    else:
        train.loc[0, "f0"] = np.inf
    with pytest.raises(RegionalError):
        fit_predict_regional(train, test, FEATURES, geo, cfg)


def test_deterministic_output() -> None:
    geo = _geo()
    cfg = _cfg()
    train = _frame(geo, labeled=True)
    test = _frame(geo, seed=7, labeled=False)
    first = fit_predict_regional(train, test, FEATURES, geo, cfg)
    second = fit_predict_regional(train, test, FEATURES, geo, cfg)
    assert np.array_equal(first, second)
    assert first.shape == (len(cfg.seeds), len(test))


def test_blend_zero_weight_identity() -> None:
    rng = np.random.default_rng(0)
    plant = np.abs(rng.normal(size=(3, 10)))
    regional = np.abs(rng.normal(size=(3, 10)))
    out = blend_seed_preds(plant, regional, 0.0)
    assert np.array_equal(out, plant)
    assert out is not plant


def test_blend_bag_linearity() -> None:
    rng = np.random.default_rng(1)
    plant = np.abs(rng.normal(size=(3, 12)))
    regional = np.abs(rng.normal(size=(3, 12)))
    out = blend_seed_preds(plant, regional, 0.6)
    np.testing.assert_allclose(bag(out), 0.4 * bag(plant) + 0.6 * bag(regional), rtol=1e-12)


@pytest.mark.parametrize("kind", ["shape", "negative", "regional_negative", "weight"])
def test_blend_contract_rejected(kind: str) -> None:
    plant = np.ones((2, 4))
    regional = np.ones((2, 4))
    weight = 0.5
    if kind == "shape":
        regional = np.ones((2, 5))
    elif kind == "negative":
        plant = plant * -1.0
    elif kind == "regional_negative":
        regional = regional * -1.0
    else:
        weight = 1.5
    with pytest.raises(RegionalError):
        blend_seed_preds(plant, regional, weight)


def test_blend_non_real_weight_rejected() -> None:
    with pytest.raises(RegionalError):
        blend_seed_preds(np.ones((2, 2)), np.ones((2, 2)), "bad")  # type: ignore[arg-type]


def _panel_parts() -> tuple[pd.DataFrame, dict[str, int], pd.DataFrame]:
    geo = _geo()
    units = site_units(geo, 1, 0, _regional())
    return geo, {k: int(v) for k, v in units.items()}, unit_centroids(geo, units)


def test_panel_frame_contract_rejected() -> None:
    geo, unit_of, centroids = _panel_parts()
    frame = _frame(geo, labeled=False)
    with pytest.raises(RegionalError):
        _panel_matrix(pd.concat([frame, frame[["f0"]]], axis=1), FEATURES, ("f0",), unit_of, centroids)
    with pytest.raises(RegionalError):
        _panel_matrix(frame.drop(columns=["f1"]), FEATURES, ("f0",), unit_of, centroids)
    bad = frame.copy()
    bad["capacity_kw"] = ["x"] * len(frame)
    with pytest.raises(RegionalError):
        _panel_matrix(bad, FEATURES, ("f0",), unit_of, centroids)
    bad = frame.copy()
    bad.loc[0, "capacity_kw"] = 0.0
    with pytest.raises(RegionalError):
        _panel_matrix(bad, FEATURES, ("f0",), unit_of, centroids)
    ghost = frame.copy()
    ghost.loc[0, "site_id"] = "ghost"
    with pytest.raises(RegionalError):
        _panel_matrix(ghost, FEATURES, ("f0",), unit_of, centroids)
    bad = frame.copy()
    bad["f1"] = ["x"] * len(frame)
    with pytest.raises(RegionalError):
        _panel_matrix(bad, FEATURES, ("f0",), unit_of, centroids)
    with pytest.raises(RegionalError):
        _panel_matrix(frame, FEATURES, ("f0",), unit_of, centroids.iloc[0:0])
    bad = frame.copy()
    bad.loc[0, "ts"] = pd.NaT
    with pytest.raises(RegionalError):
        _panel_matrix(bad, FEATURES, ("f0",), unit_of, centroids)
    bad = frame.copy()
    bad["ts"] = [1 if i == 0 else "a" for i in range(len(frame))]
    with pytest.raises(RegionalError):
        _panel_matrix(bad, FEATURES, ("f0",), unit_of, centroids)


def test_panel_targets_contract_rejected() -> None:
    geo = _geo()
    cfg = _cfg()
    train = _frame(geo, labeled=True)
    test = _frame(geo, seed=7, labeled=False)
    with pytest.raises(RegionalError):
        fit_predict_regional(train.drop(columns=["smp"]), test, FEATURES, geo, cfg)
    bad = train.copy()
    bad["energy"] = ["x"] * len(train)
    with pytest.raises(RegionalError):
        fit_predict_regional(bad, test, FEATURES, geo, cfg)
    bad = train.copy()
    bad.loc[0, "smp"] = np.nan
    with pytest.raises(RegionalError):
        fit_predict_regional(bad, test, FEATURES, geo, cfg)


@pytest.mark.parametrize("wmode", ["k_cap", "none"])
def test_regional_weight_modes_run(wmode: str) -> None:
    geo = _geo()
    cfg = _cfg(weight=wmode)  # type: ignore[arg-type]
    out = fit_predict_regional(_frame(geo, labeled=True), _frame(geo, seed=7, labeled=False), FEATURES, geo, cfg)
    assert np.isfinite(out).all() and (out >= 0).all()


def test_regional_weight_overflow_rejected() -> None:
    geo = _geo()
    cfg = _cfg()
    train = _frame(geo, labeled=True)
    train["smp"] = 1e308
    train["k"] = 10.0
    with pytest.raises(RegionalError, match="weights must be finite"):
        fit_predict_regional(train, _frame(geo, seed=7, labeled=False), FEATURES, geo, cfg)


@pytest.mark.parametrize("features", [[], ["f0", "f0"]])
def test_regional_invalid_feature_list(features: list) -> None:
    geo = _geo()
    with pytest.raises(RegionalError):
        fit_predict_regional(_frame(geo, labeled=True), _frame(geo, seed=7, labeled=False), features, geo, _cfg())


def test_regional_sd_var_must_be_mean_feature() -> None:
    geo = _geo()
    cfg = _cfg(regional=_regional(sd_vars=("zz",)))
    with pytest.raises(RegionalError):
        fit_predict_regional(_frame(geo, labeled=True), _frame(geo, seed=7, labeled=False), FEATURES, geo, cfg)


def test_regional_empty_units_rejected() -> None:
    geo = _geo()
    cfg = _cfg(regional=_regional(weight=0.0, units=()))
    with pytest.raises(RegionalError, match="units must not be empty"):
        fit_predict_regional(_frame(geo, labeled=True), _frame(geo, seed=7, labeled=False), FEATURES, geo, cfg)


def test_regional_test_capacity_contract_rejected() -> None:
    geo = _geo()
    cfg = _cfg()
    train = _frame(geo, labeled=True)
    bad = _frame(geo, seed=7, labeled=False)
    bad["capacity_kw"] = ["x"] * len(bad)
    with pytest.raises(RegionalError):
        fit_predict_regional(train, bad, FEATURES, geo, cfg)
    bad = _frame(geo, seed=7, labeled=False)
    bad.loc[0, "capacity_kw"] = 0.0
    with pytest.raises(RegionalError):
        fit_predict_regional(train, bad, FEATURES, geo, cfg)


def test_regional_parallel_matches_serial() -> None:
    geo = _geo()
    train = _frame(geo, labeled=True)
    test = _frame(geo, seed=7, labeled=False)
    seq = fit_predict_regional(train, test, FEATURES, geo, _cfg())
    par = fit_predict_regional(train, test, FEATURES, geo, _cfg(n_jobs=2, seed_workers=2))
    assert np.array_equal(par, seq)


def test_regional_prediction_overflow_rejected() -> None:
    geo = _geo()
    cfg = _cfg(weight="none", regional=_regional(units=((6, 0), (6, 1), (6, 2))))  # type: ignore[arg-type]
    train = _frame(geo, labeled=True)
    train["capacity_kw"] = 1e308
    train["energy"] = 1e308
    test = _frame(geo, seed=7, labeled=False)
    test["capacity_kw"] = 1e308
    with pytest.raises(RegionalError, match="finite and non-negative"):
        fit_predict_regional(train, test, FEATURES, geo, cfg)
