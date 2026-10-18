import numpy as np
import pandas as pd
import pytest
import warnings
from sklearn.exceptions import ConvergenceWarning

from src.oibc.config import FamilyConfig, ModelConfig, RegionalConfig
from src.oibc.models.families import FamilyError, blend_families, fit_predict_catboost, fit_predict_mlp

FEATURES = ["cs_ghi", "f0", "f1"]


@pytest.mark.parametrize("member", [False, True])
def test_blend_non_numeric_inputs_raise_family_error(member: bool) -> None:
    base = np.array([[1.0]]) if member else np.array([["invalid"]])
    cat = np.array(["invalid"]) if member else np.array([1.0])
    with pytest.raises(FamilyError, match="numeric"):
        blend_families(base, cat, None, FamilyConfig(cat_weight=0.2))


def test_blend_rejects_floating_point_overflow() -> None:
    maximum = np.finfo("float64").max
    cfg = FamilyConfig(cat_weight=0.45657850369813724, mlp_weight=0.11055200258577665)
    with pytest.raises(FamilyError, match="finite"):
        blend_families(np.array([[maximum]]), np.array([maximum]), np.array([maximum]), cfg)


def test_mlp_missing_features_use_train_medians_and_zero() -> None:
    from src.oibc.models.families import _mlp_design
    from sklearn.preprocessing import StandardScaler

    train = np.array([[1.0, np.nan, np.nan], [3.0, 4.0, np.nan], [5.0, 8.0, np.nan]])
    test = np.array([[2.0, np.nan, 7.0]])
    filled = np.array([[1.0, 6.0, 0.0], [3.0, 4.0, 0.0], [5.0, 8.0, 0.0]])
    filled_test = np.array([[2.0, 6.0, 7.0]])
    scaler = StandardScaler()
    expected_train = scaler.fit_transform(np.hstack((filled, filled * filled[:, :1])))
    expected_test = scaler.transform(np.hstack((filled_test, filled_test * filled_test[:, :1])))
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        actual_train, actual_test = _mlp_design(train, test, ["cs_ghi", "f0", "f1"])
    np.testing.assert_array_equal(actual_train, expected_train)
    np.testing.assert_array_equal(actual_test, expected_test)


def test_mlp_warning_filters_restored_and_worker_counts_match() -> None:
    geo = _geo()
    train = _frame(geo)
    test = _frame(geo, seed=1, labeled=False)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        before = list(warnings.filters)
        serial = fit_predict_mlp(train, test, FEATURES, geo, _cfg(seed_workers=1))
        parallel_cfg = fit_predict_mlp(train, test, FEATURES, geo, _cfg(seed_workers=2))
        assert warnings.filters == before
    assert all(not issubclass(w.category, ConvergenceWarning) for w in captured)
    np.testing.assert_array_equal(serial, parallel_cfg)

TS = pd.date_range("2024-01-01 09:00", periods=6, freq="h", tz="Asia/Seoul")


def _regional(**kwargs: object) -> RegionalConfig:
    base: dict[str, object] = {
        "weight": 0.6,
        "units": ((2, 0),),
        "sd_vars": ("cs_ghi",),
        "excluded_vars": (),
        "n_rounds": 10,
        "min_child_samples": 2,
    }
    base.update(kwargs)
    return RegionalConfig(**base)  # type: ignore[arg-type]


def _cfg(**kwargs: object) -> ModelConfig:
    base: dict[str, object] = {
        "n_jobs": 1,
        "seed_workers": 1,
        "seeds": (0, 1),
        "regional": _regional(),
    }
    fam: dict[str, object] = {
        "cat_weight": 0.2,
        "mlp_weight": 0.2,
        "cat_units": ((2, 0),),
        "cat_seeds": (0, 1),
        "cat_iterations": 20,
        "mlp_units": ((2, 0),),
        "mlp_seeds": (0, 1),
        "mlp_hidden": (8,),
        "mlp_max_iter": 5,
    }
    fam.update(kwargs.pop("families", {}))  # type: ignore[arg-type]
    base.update(kwargs)
    base["families"] = FamilyConfig(**fam)  # type: ignore[arg-type]
    return ModelConfig(**base)  # type: ignore[arg-type]


def _geo(n: int = 6) -> pd.DataFrame:
    half = n // 2
    lat = [36.0 + 0.01 * i for i in range(half)] + [37.5 + 0.01 * i for i in range(n - half)]
    lon = [127.0 + 0.01 * i for i in range(half)] + [129.0 + 0.01 * i for i in range(n - half)]
    return pd.DataFrame(
        {
            "site_id": [f"s{i}" for i in range(n)],
            "lat": lat,
            "lon": lon,
            "capacity_kw": [100.0 + 50.0 * (i % 3) for i in range(n)],
        }
    )


def _frame(geo: pd.DataFrame, seed: int = 0, labeled: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = [(s, t) for s in geo["site_id"] for t in TS]
    n = len(rows)
    cap_of = dict(zip(geo["site_id"], geo["capacity_kw"], strict=True))
    data: dict[str, object] = {
        "site_id": [s for s, _ in rows],
        "ts": [t for _, t in rows],
        "capacity_kw": [cap_of[s] for s, _ in rows],
        "cs_ghi": np.abs(rng.normal(loc=500.0, scale=100.0, size=n)),
        "f0": rng.normal(size=n),
        "f1": rng.normal(size=n),
        "smp": np.full(n, 120.0),
        "k": np.ones(n),
    }
    if labeled:
        data["energy"] = np.abs(rng.normal(loc=60.0, scale=20.0, size=n))
    return pd.DataFrame(data)


def test_catboost_prediction_shape_and_non_negativity() -> None:
    geo = _geo()
    train = _frame(geo, seed=0)
    test = _frame(geo, seed=1, labeled=False)
    out = fit_predict_catboost(train, test, FEATURES, geo, _cfg())
    assert out.shape == (len(test),)
    assert np.isfinite(out).all()
    assert (out >= 0).all()


def test_catboost_deterministic_across_calls() -> None:
    geo = _geo()
    train = _frame(geo, seed=0)
    test = _frame(geo, seed=1, labeled=False)
    cfg = _cfg()
    assert np.array_equal(
        fit_predict_catboost(train, test, FEATURES, geo, cfg),
        fit_predict_catboost(train, test, FEATURES, geo, cfg),
    )


def test_catboost_writes_no_training_files(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    geo = _geo()
    train = _frame(geo, seed=0)
    test = _frame(geo, seed=1, labeled=False)
    monkeypatch.chdir(tmp_path)
    fit_predict_catboost(train, test, FEATURES, geo, _cfg())
    assert not (tmp_path / "catboost_info").exists()


def test_mlp_prediction_shape_and_non_negativity() -> None:
    geo = _geo()
    train = _frame(geo, seed=0)
    test = _frame(geo, seed=1, labeled=False)
    out = fit_predict_mlp(train, test, FEATURES, geo, _cfg())
    assert out.shape == (len(test),)
    assert np.isfinite(out).all()
    assert (out >= 0).all()


def test_mlp_deterministic_across_calls() -> None:
    geo = _geo()
    train = _frame(geo, seed=0)
    test = _frame(geo, seed=1, labeled=False)
    cfg = _cfg()
    cfg = ModelConfig(**{**cfg.model_dump(), "seed_workers": 2})
    assert np.array_equal(
        fit_predict_mlp(train, test, FEATURES, geo, cfg),
        fit_predict_mlp(train, test, FEATURES, geo, cfg),
    )


def test_mlp_uses_train_statistics_only() -> None:
    geo = _geo()
    train = _frame(geo, seed=0)
    test = _frame(geo, seed=1, labeled=False)
    cfg = _cfg()
    units = sorted(test["site_id"].unique().tolist())
    small = test[test["site_id"].isin(units[:2])].copy()
    assert len(small) >= 2
    changed = small.copy()
    idx = changed.index[0]
    changed.loc[idx, "cs_ghi"] = changed["cs_ghi"].max() * 100.0 + 1e6
    base_pred = fit_predict_mlp(train, small, FEATURES, geo, cfg)
    shifted_pred = fit_predict_mlp(train, changed, FEATURES, geo, cfg)
    same_ts = small["ts"] == small.loc[idx, "ts"]
    assert (~same_ts).any()
    assert np.array_equal(base_pred[~same_ts.to_numpy()], shifted_pred[~same_ts.to_numpy()])


def test_mlp_requires_cs_ghi() -> None:
    geo = _geo()
    train = _frame(geo, seed=0)
    test = _frame(geo, seed=1, labeled=False)
    with pytest.raises(FamilyError):
        fit_predict_mlp(train, test, ["f0", "f1"], geo, _cfg())


@pytest.mark.parametrize("fn", [fit_predict_catboost, fit_predict_mlp])
def test_family_members_redistribute_by_capacity(fn: object) -> None:  # type: ignore[no-untyped-def]
    geo = pd.DataFrame(
        {
            "site_id": ["a", "b"],
            "lat": [36.0, 36.01],
            "lon": [127.0, 127.01],
            "capacity_kw": [100.0, 300.0],
        }
    )
    ts = pd.date_range("2024-01-01 09:00", periods=12, freq="h", tz="Asia/Seoul")
    rng = np.random.default_rng(0)
    rows = [(s, t) for t in ts for s in ["a", "b"]]
    n = len(rows)
    cap_of = {"a": 100.0, "b": 300.0}
    train = pd.DataFrame(
        {
            "site_id": [s for s, _ in rows],
            "ts": [t for _, t in rows],
            "capacity_kw": [cap_of[s] for s, _ in rows],
            "cs_ghi": np.abs(rng.normal(500, 50, n)),
            "f0": rng.normal(size=n),
            "f1": rng.normal(size=n),
            "smp": np.full(n, 120.0),
            "k": np.ones(n),
            "energy": np.abs(rng.normal(60, 10, n)),
        }
    )
    test = train.drop(columns=["energy"]).copy()
    cfg = _cfg(**{"families": {"cat_units": ((1, 0),), "mlp_units": ((1, 0),)}})
    out = fn(train, test, FEATURES, geo, cfg)  # type: ignore[operator]
    a = out[test["site_id"].to_numpy() == "a"]
    b = out[test["site_id"].to_numpy() == "b"]
    np.testing.assert_allclose(b / np.maximum(a, 1e-12), 3.0, rtol=1e-9)


@pytest.mark.parametrize("fn", [fit_predict_catboost, fit_predict_mlp])
def test_family_rejects_missing_sd_var(fn: object) -> None:  # type: ignore[no-untyped-def]
    geo = _geo()
    train = _frame(geo, seed=0)
    test = _frame(geo, seed=1, labeled=False)
    cfg = _cfg()
    cfg = ModelConfig(**{**cfg.model_dump(), "regional": {**cfg.regional.model_dump(), "sd_vars": ("nope",)}})
    with pytest.raises(FamilyError):
        fn(train, test, FEATURES, geo, cfg)  # type: ignore[operator]


@pytest.mark.parametrize("fn", [fit_predict_catboost, fit_predict_mlp])
def test_family_rejects_empty_frames(fn: object) -> None:  # type: ignore[no-untyped-def]
    geo = _geo()
    train = _frame(geo, seed=0)
    test = _frame(geo, seed=1, labeled=False)
    cfg = _cfg()
    with pytest.raises(FamilyError):
        fn(train.iloc[0:0], test, FEATURES, geo, cfg)  # type: ignore[operator]
    with pytest.raises(FamilyError):
        fn(train, test.iloc[0:0], FEATURES, geo, cfg)  # type: ignore[operator]


def test_blend_families_weighted_sum() -> None:
    base = np.array([[10.0, 20.0], [30.0, 40.0]])
    cat = np.array([0.0, 100.0])
    mlp = np.array([50.0, 50.0])
    cfg = FamilyConfig(cat_weight=0.2, mlp_weight=0.2)
    np.testing.assert_allclose(blend_families(base, cat, mlp, cfg), np.array([[16.0, 42.0], [28.0, 54.0]]))


def test_blend_families_zero_weights_returns_copy() -> None:
    base = np.array([[10.0, 20.0], [30.0, 40.0]])
    out = blend_families(base, None, None, FamilyConfig())
    assert out is not base
    np.testing.assert_array_equal(out, base)


def test_blend_families_requires_active_member() -> None:
    base = np.array([[10.0, 20.0]])
    with pytest.raises(FamilyError):
        blend_families(base, None, np.array([1.0, 1.0]), FamilyConfig(cat_weight=0.2))
    with pytest.raises(FamilyError):
        blend_families(base, np.array([1.0, 1.0]), None, FamilyConfig(mlp_weight=0.2))


def test_blend_families_rejects_bad_shapes_and_values() -> None:
    cfg = FamilyConfig(cat_weight=0.2, mlp_weight=0.2)
    good = np.array([[10.0, 20.0]])
    with pytest.raises(FamilyError):
        blend_families(good, np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.0]), cfg)
    with pytest.raises(FamilyError):
        blend_families(np.array([10.0, 20.0]), np.array([1.0, 1.0]), np.array([1.0, 1.0]), cfg)
    with pytest.raises(FamilyError):
        blend_families(np.array([[-1.0, 2.0]]), np.array([1.0, 1.0]), np.array([1.0, 1.0]), cfg)
    with pytest.raises(FamilyError):
        blend_families(np.array([[np.nan, 2.0]]), np.array([1.0, 1.0]), np.array([1.0, 1.0]), cfg)


@pytest.mark.parametrize("features", [(), ("cs_ghi", "cs_ghi")])
def test_family_rejects_bad_features(features: tuple) -> None:
    geo = _geo()
    train = _frame(geo, seed=0)
    test = _frame(geo, seed=1, labeled=False)
    with pytest.raises(FamilyError):
        fit_predict_catboost(train, test, list(features), geo, _cfg())


def test_family_rejects_non_numeric_capacity() -> None:
    geo = _geo()
    train = _frame(geo, seed=0)
    test = _frame(geo, seed=1, labeled=False).astype({"capacity_kw": "object"})
    test["capacity_kw"] = ["x"] * len(test)
    with pytest.raises(FamilyError):
        fit_predict_catboost(train, test, FEATURES, geo, _cfg())


def test_family_rejects_non_positive_capacity() -> None:
    geo = _geo()
    train = _frame(geo, seed=0)
    test = _frame(geo, seed=1, labeled=False)
    test.loc[test.index[0], "capacity_kw"] = 0.0
    with pytest.raises(FamilyError):
        fit_predict_catboost(train, test, FEATURES, geo, _cfg())


def test_family_wraps_regional_contract_errors() -> None:
    geo = _geo()
    train = _frame(geo, seed=0).drop(columns=["energy"])
    test = _frame(geo, seed=1, labeled=False)
    with pytest.raises(FamilyError):
        fit_predict_catboost(train, test, FEATURES, geo, _cfg())


def test_family_rejects_zero_mean_weight(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import src.oibc.models.regional as regional

    orig = regional._panel_targets

    def _zero(*args: object, **kwargs: object) -> tuple:
        target, weight = orig(*args, **kwargs)  # type: ignore[arg-type]
        return target, np.zeros_like(weight)

    monkeypatch.setattr(regional, "_panel_targets", _zero)
    geo = _geo()
    train = _frame(geo, seed=0)
    test = _frame(geo, seed=1, labeled=False)
    with pytest.raises(FamilyError):
        fit_predict_catboost(train, test, FEATURES, geo, _cfg())


def test_family_empty_units_rejected_when_idle() -> None:
    geo = _geo()
    train = _frame(geo, seed=0)
    test = _frame(geo, seed=1, labeled=False)
    idle_cat = ModelConfig(
        n_jobs=1,
        seed_workers=1,
        seeds=(0, 1),
        regional=_regional(),
        families=FamilyConfig(cat_units=(), mlp_units=()),
    )
    with pytest.raises(FamilyError):
        fit_predict_catboost(train, test, FEATURES, geo, idle_cat)
    with pytest.raises(FamilyError):
        fit_predict_mlp(train, test, FEATURES, geo, idle_cat)


def test_family_rejects_non_finite_member_output(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    class _InfBoost:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def fit(self, *args: object, **kwargs: object) -> "_InfBoost":
            return self

        def predict(self, data: np.ndarray) -> np.ndarray:
            return np.full(len(data), np.inf)

    class _NanMLP:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def fit(self, *args: object, **kwargs: object) -> "_NanMLP":
            return self

        def predict(self, data: np.ndarray) -> np.ndarray:
            return np.full(len(data), np.nan)

    monkeypatch.setattr("catboost.CatBoostRegressor", _InfBoost)
    monkeypatch.setattr("src.oibc.models.families.MLPRegressor", _NanMLP)
    geo = _geo()
    train = _frame(geo, seed=0)
    test = _frame(geo, seed=1, labeled=False)
    with pytest.raises(FamilyError):
        fit_predict_catboost(train, test, FEATURES, geo, _cfg())
    with pytest.raises(FamilyError):
        fit_predict_mlp(train, test, FEATURES, geo, _cfg())


def test_mlp_design_requires_cs_ghi_without_sd() -> None:
    geo = _geo()
    train = _frame(geo, seed=0)
    test = _frame(geo, seed=1, labeled=False)
    cfg = ModelConfig(
        n_jobs=1,
        seed_workers=1,
        seeds=(0, 1),
        regional=_regional(sd_vars=(), excluded_vars=()),
        families=FamilyConfig(
            cat_weight=0.0,
            mlp_weight=0.0,
            mlp_units=((2, 0),),
            mlp_seeds=(0,),
            mlp_max_iter=5,
            mlp_hidden=(8,),
        ),
    )
    with pytest.raises(FamilyError):
        fit_predict_mlp(train, test, ["f0", "f1"], geo, cfg)


def test_blend_families_rejects_non_finite_member() -> None:
    cfg = FamilyConfig(cat_weight=0.2, mlp_weight=0.2)
    base = np.array([[10.0, 20.0]])
    with pytest.raises(FamilyError):
        blend_families(base, np.array([np.inf, 1.0]), np.array([1.0, 1.0]), cfg)
    with pytest.raises(FamilyError):
        blend_families(base, np.array([-1.0, 1.0]), np.array([1.0, 1.0]), cfg)


@pytest.mark.parametrize("fn", [fit_predict_catboost, fit_predict_mlp])
def test_family_does_not_mutate_inputs(fn: object) -> None:  # type: ignore[no-untyped-def]
    geo = _geo()
    train = _frame(geo, seed=0)
    test = _frame(geo, seed=1, labeled=False)
    before_train = train.copy(deep=True)
    before_test = test.copy(deep=True)
    fn(train, test, FEATURES, geo, _cfg())  # type: ignore[operator]
    pd.testing.assert_frame_equal(train, before_train)
    pd.testing.assert_frame_equal(test, before_test)
