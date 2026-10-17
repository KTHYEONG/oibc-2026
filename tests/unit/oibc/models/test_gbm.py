import numpy as np
import pandas as pd
import pytest

from src.oibc.config import ModelConfig
from src.oibc.models.gbm import ModelError, _features, bag, fit_seed_models, predict_seed_kwh, sample_weight, training_target

FEATURES = ["f0", "f1", "f2"]


def _cfg(**kwargs: object) -> ModelConfig:
    base: dict[str, object] = {"n_rounds": 5, "num_leaves": 4, "n_jobs": 1, "min_child_samples": 10}
    base.update(kwargs)
    return ModelConfig(**base)  # type: ignore[arg-type]


def _panel(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "f0": rng.normal(size=n),
        "f1": rng.normal(size=n),
        "f2": rng.normal(size=n),
        "energy": np.abs(rng.normal(loc=50.0, scale=20.0, size=n)),
        "capacity_kw": np.full(n, 100.0),
        "smp": np.full(n, 120.0),
        "k": np.ones(n),
    })


def test_target_clipping() -> None:
    panel = pd.DataFrame({"energy": [-5.0, 50.0, 200.0], "capacity_kw": [100.0, 100.0, 100.0]})
    out = training_target(panel, _cfg())
    assert out.dtype == np.float32
    assert out.tolist() == pytest.approx([0.0, 0.5, 1.2])


def test_target_rejects_eval_rows() -> None:
    panel = pd.DataFrame({"energy": [10.0, np.nan], "capacity_kw": [100.0, 100.0]})
    with pytest.raises(ModelError):
        training_target(panel, _cfg())


def test_weight_modes() -> None:
    panel = pd.DataFrame({"smp": [0.0], "k": [3], "capacity_kw": [10.0]})
    cfg = _cfg()
    assert sample_weight(panel, cfg).tolist() == pytest.approx([cfg.min_weight])
    assert sample_weight(panel, _cfg(weight="k_cap")).tolist() == pytest.approx([30.0])
    assert sample_weight(panel, _cfg(weight="none")).tolist() == pytest.approx([1.0])


def test_predict_nonneg_finite_scales_with_capacity() -> None:
    train = _panel()
    cfg = _cfg()
    models = fit_seed_models(train, FEATURES, cfg)
    frame = train.copy()
    base = predict_seed_kwh(models, frame, FEATURES)
    assert (base >= 0.0).all() and np.isfinite(base).all()
    doubled = frame.copy()
    doubled["capacity_kw"] = doubled["capacity_kw"] * 2.0
    np.testing.assert_allclose(predict_seed_kwh(models, doubled, FEATURES), base * 2.0, rtol=1e-9)
    pd.testing.assert_frame_equal(frame, train)


def test_seed_determinism() -> None:
    train = _panel()
    cfg = _cfg()
    first = predict_seed_kwh(fit_seed_models(train, FEATURES, cfg), train, FEATURES)
    second = predict_seed_kwh(fit_seed_models(train, FEATURES, cfg), train, FEATURES)
    assert np.array_equal(first, second)


def test_seed_bag_mean() -> None:
    train = _panel()
    cfg = _cfg(seeds=(0, 1))
    preds = predict_seed_kwh(fit_seed_models(train, FEATURES, cfg), train, FEATURES)
    assert preds.shape == (2, len(train))
    assert not np.array_equal(preds[0], preds[1])
    np.testing.assert_allclose(bag(preds), preds.mean(axis=0), rtol=1e-12)


def test_missing_feature_rejected() -> None:
    with pytest.raises(ModelError):
        fit_seed_models(_panel(n=50), ["no_such_col"], _cfg())


def test_target_rejects_bad_capacity() -> None:
    bad = pd.DataFrame({"energy": [10.0], "capacity_kw": [0.0]})
    with pytest.raises(ModelError):
        training_target(bad, _cfg())


def test_weight_rejects_missing_values() -> None:
    bad = pd.DataFrame({"smp": [np.nan], "k": [1], "capacity_kw": [10.0]})
    with pytest.raises(ModelError):
        sample_weight(bad, _cfg())
    bad_cap = pd.DataFrame({"smp": [100.0], "k": [1], "capacity_kw": [np.inf]})
    with pytest.raises(ModelError):
        sample_weight(bad_cap, _cfg())


def test_fit_rejects_empty_and_bad_features() -> None:
    with pytest.raises(ModelError):
        fit_seed_models(_panel(n=10).iloc[0:0], FEATURES, _cfg())
    bad = _panel(n=10)
    bad["f0"] = pd.Series(["x"] * 10)
    with pytest.raises(ModelError):
        fit_seed_models(bad, FEATURES, _cfg())
    bad_nan = _panel(n=10)
    bad_nan.loc[0, "f1"] = np.inf
    with pytest.raises(ModelError):
        fit_seed_models(bad_nan, FEATURES, _cfg())


def test_predict_rejects_empty_models_and_bad_frame() -> None:
    train = _panel(n=50)
    models = fit_seed_models(train, FEATURES, _cfg())
    with pytest.raises(ModelError):
        predict_seed_kwh([], train, FEATURES)
    bad = train.copy()
    bad["f0"] = pd.Series(["x"] * len(train))
    with pytest.raises(ModelError):
        predict_seed_kwh(models, bad, FEATURES)
    bad_cap = train.copy()
    bad_cap["capacity_kw"] = np.inf
    with pytest.raises(ModelError):
        predict_seed_kwh(models, bad_cap, FEATURES)


def test_predict_rejects_nonfinite_output() -> None:
    class _InfModel:
        def predict(self, data: np.ndarray) -> np.ndarray:
            return np.full(len(data), np.inf)

    frame = _panel(n=10)
    with pytest.raises(ModelError, match="finite"):
        predict_seed_kwh([_InfModel()], frame, FEATURES)  # type: ignore[list-item]


def test_bag_rejects_bad_shape() -> None:
    with pytest.raises(ModelError):
        bag(np.ones(5))
    with pytest.raises(ModelError):
        bag(np.empty((0, 5)))


def test_nan_features_supported_and_empty_prediction() -> None:
    frame = _panel()
    frame.loc[::3, "f0"] = np.nan
    models = fit_seed_models(frame, FEATURES, _cfg())
    assert np.isfinite(predict_seed_kwh(models, frame, FEATURES)).all()
    assert predict_seed_kwh(models, frame.iloc[:0], FEATURES).shape == (3, 0)


@pytest.mark.parametrize("features", [[], ["f0", "f0"]])
def test_invalid_feature_list(features: list[str]) -> None:
    with pytest.raises(ModelError, match="non-empty and distinct"):
        fit_seed_models(_panel(), features, _cfg())


def test_duplicate_and_nonnumeric_contract_columns() -> None:
    panel = _panel()
    with pytest.raises(ModelError, match="duplicate"):
        training_target(pd.concat([panel, panel[["energy"]]], axis=1), _cfg())
    panel["energy"] = "bad"
    with pytest.raises(ModelError, match="numeric"):
        training_target(panel, _cfg())


def test_weight_overflow_rejected() -> None:
    panel = pd.DataFrame({"smp": [1e300], "k": [3], "capacity_kw": [1e300]})
    with pytest.raises(ModelError, match="weights must be finite"):
        sample_weight(panel, _cfg())


@pytest.mark.parametrize("value", [np.nan, np.inf, -1.0])
def test_bag_invalid_values(value: float) -> None:
    with pytest.raises(ModelError, match="finite and non-negative"):
        bag(np.array([[value]]))


def test_bag_large_finite_values() -> None:
    np.testing.assert_array_equal(bag(np.full((3, 1), 1e308)), [1e308])


def test_prediction_overflow_rejected() -> None:
    class HugeModel:
        def predict(self, data: np.ndarray) -> np.ndarray:
            return np.full(len(data), 1e308)

    with pytest.raises(ModelError, match="prediction must be finite"):
        predict_seed_kwh([HugeModel()], _panel(), FEATURES)  # type: ignore[list-item]


def test_concurrent_training_matches_sequential() -> None:
    train = _panel()
    seq = predict_seed_kwh(
        fit_seed_models(train, FEATURES, _cfg(seeds=(0, 1, 2), seed_workers=1, n_jobs=1)),
        train, FEATURES,
    )
    par = predict_seed_kwh(
        fit_seed_models(train, FEATURES, _cfg(seeds=(0, 1, 2), seed_workers=3, n_jobs=3)),
        train, FEATURES,
    )
    assert np.array_equal(seq, par)


@pytest.mark.parametrize("seed_workers", [1, 2])
def test_seed_dataset_independent(seed_workers: int) -> None:
    train = _panel()
    pair = predict_seed_kwh(
        fit_seed_models(train, FEATURES, _cfg(seeds=(0, 1), seed_workers=seed_workers, n_jobs=2)),
        train, FEATURES,
    )
    solo = predict_seed_kwh(
        fit_seed_models(train, FEATURES, _cfg(seeds=(1,), seed_workers=1, n_jobs=1)),
        train, FEATURES,
    )
    assert np.array_equal(pair[1], solo[0])


def test_seed_order_preserved() -> None:
    train = _panel()
    models = fit_seed_models(train, FEATURES, _cfg(seeds=(2, 0), seed_workers=2, n_jobs=2))
    preds = predict_seed_kwh(models, train, FEATURES)
    for i, seed in enumerate((2, 0)):
        solo = predict_seed_kwh(
            fit_seed_models(train, FEATURES, _cfg(seeds=(seed,), seed_workers=1, n_jobs=1)),
            train, FEATURES,
        )
        assert np.array_equal(preds[i], solo[0])


def test_features_float32_contiguous_nan() -> None:
    frame = _panel(n=50)
    frame.loc[::5, "f0"] = np.nan
    out = _features(frame, FEATURES)
    assert out.dtype == np.float32
    assert out.flags["C_CONTIGUOUS"]
    assert np.isnan(out[::5, 0]).all()


def test_features_float32_overflow_rejected() -> None:
    bad = _panel(n=50)
    bad.loc[0, "f0"] = 1e39
    with pytest.raises(ModelError):
        fit_seed_models(bad, FEATURES, _cfg())


def test_predict_threads_invariant() -> None:
    train = _panel()
    models = fit_seed_models(train, FEATURES, _cfg())
    first = predict_seed_kwh(models, train, FEATURES, num_threads=1)
    second = predict_seed_kwh(models, train, FEATURES, num_threads=2)
    assert np.array_equal(first, second)


def test_predict_threads_lower_bound() -> None:
    train = _panel(n=50)
    models = fit_seed_models(train, FEATURES, _cfg())
    with pytest.raises(ModelError):
        predict_seed_kwh(models, train, FEATURES, num_threads=0)


def test_concurrent_failure_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.oibc.models.gbm as gbm_mod

    real_train = gbm_mod.lgb.train
    failure = RuntimeError("boom-seed-2")

    def _fail_once(params, *args, **kwargs):  # type: ignore[no-untyped-def]
        if params["seed"] == 1:
            raise failure
        return real_train(params, *args, **kwargs)

    monkeypatch.setattr(gbm_mod.lgb, "train", _fail_once)
    with pytest.raises(RuntimeError, match="boom-seed-2") as exc:
        fit_seed_models(_panel(n=100), FEATURES, _cfg(seeds=(0, 1), seed_workers=2, n_jobs=2))
    assert exc.value is failure


@pytest.mark.parametrize(
    ("seed_workers", "n_jobs", "seeds", "workers", "threads"),
    [(8, 6, (0, 1), 2, 3), (8, 2, (0, 1, 2), 2, 1), (2, 5, (0, 1, 2), 2, 2)],
)
def test_training_respects_thread_budget(
    monkeypatch: pytest.MonkeyPatch,
    seed_workers: int,
    n_jobs: int,
    seeds: tuple[int, ...],
    workers: int,
    threads: int,
) -> None:
    from threading import Barrier, Lock

    import src.oibc.models.gbm as gbm_mod

    real_train = gbm_mod.lgb.train
    barrier = Barrier(workers, timeout=10)
    lock = Lock()
    datasets = []
    inputs = []

    def checked_train(params, dataset, **kwargs):  # type: ignore[no-untyped-def]
        assert params["num_threads"] == threads
        with lock:
            datasets.append(dataset)
            inputs.append((dataset.data, dataset.label, dataset.weight))
            first_batch = len(datasets) <= workers
        if first_batch:
            barrier.wait()
        return real_train(params, dataset, **kwargs)

    monkeypatch.setattr(gbm_mod.lgb, "train", checked_train)
    models = fit_seed_models(
        _panel(), FEATURES, _cfg(seeds=seeds, seed_workers=seed_workers, n_jobs=n_jobs),
    )
    assert len(models) == len(seeds)
    assert len({id(dataset) for dataset in datasets}) == len(seeds)
    for arrays in zip(*inputs, strict=True):
        assert all(array is arrays[0] for array in arrays)
        assert not arrays[0].flags.writeable
