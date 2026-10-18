"""지역 단위 CatBoost·MLP 구성원 예측과 최종 혼합."""

from __future__ import annotations

import logging
import time
import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from src.oibc.config import FamilyConfig, ModelConfig
from src.oibc.models import regional as _regional
from src.oibc.models.gbm import ModelError

logger = logging.getLogger(__name__)


class FamilyError(ModelError):
    """모델 계열 구성원 입력 계약 위반."""


def _common_panel(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: Sequence[str],
    site_geo: pd.DataFrame,
    model_cfg: ModelConfig,
    n_units: int,
    kmeans_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    if not features or len(set(features)) != len(features):
        raise FamilyError("features must be non-empty and distinct")
    if train.empty or test.empty:
        raise FamilyError("train and test must not be empty")
    cfg = model_cfg.regional
    mean_cols = [f for f in features if f not in set(cfg.excluded_vars)]
    for v in cfg.sd_vars:
        if v not in mean_cols:
            raise FamilyError(f"sd_var {v!r} must be a mean feature")
    try:
        test_cap = np.asarray(test["capacity_kw"].to_numpy(dtype="float64"), dtype="float64")
    except (ValueError, TypeError, KeyError) as exc:
        raise FamilyError("capacity_kw must be numeric") from exc
    if not np.isfinite(test_cap).all() or (test_cap <= 0).any():
        raise FamilyError("capacity_kw must be finite and positive")
    try:
        member_units = _regional.site_units(site_geo, n_units, kmeans_seed, cfg)
        centroids = _regional.unit_centroids(site_geo, member_units)
        unit_of = {k: int(v) for k, v in member_units.items()}
        train_mat, feat_cols, train_code, train_cap, _ = _regional._panel_matrix(
            train, mean_cols, cfg.sd_vars, unit_of, centroids
        )
        test_mat, _, test_code, _, _ = _regional._panel_matrix(test, mean_cols, cfg.sd_vars, unit_of, centroids)
        target, weight = _regional._panel_targets(train, train_code, len(train_mat), train_cap, model_cfg)
    except _regional.RegionalError as exc:
        raise FamilyError(str(exc)) from exc
    mean = float(weight.mean())
    if not np.isfinite(mean) or mean <= 0:
        raise FamilyError("weights must have a finite positive mean")
    norm_weight = (weight / mean).astype("float64")
    return train_mat, test_mat, target, norm_weight, test_code, test_cap, feat_cols


def fit_predict_catboost(
    train: pd.DataFrame, test: pd.DataFrame, features: Sequence[str], site_geo: pd.DataFrame, model_cfg: ModelConfig
) -> np.ndarray:
    """cat_units x cat_seeds 평균 지역 CatBoost 예측 (len(test),) 발전소 kWh(≥0, 유한)를 반환한다. 위반 시 FamilyError."""
    from catboost import CatBoostRegressor

    fam = model_cfg.families
    if not fam.cat_units or not fam.cat_seeds:
        raise FamilyError("cat_units and cat_seeds must not be empty")
    total = np.zeros(len(test), dtype="float64")
    for n_units, kmeans_seed in fam.cat_units:
        t0 = time.perf_counter()
        train_mat, test_mat, target, weight, test_code, test_cap, _ = _common_panel(
            train, test, features, site_geo, model_cfg, int(n_units), int(kmeans_seed)
        )
        for seed in fam.cat_seeds:
            model = CatBoostRegressor(
                loss_function="MAE",
                iterations=int(fam.cat_iterations),
                learning_rate=float(fam.cat_learning_rate),
                depth=int(fam.cat_depth),
                thread_count=int(model_cfg.n_jobs),
                random_seed=int(seed),
                verbose=0,
                allow_writing_files=False,
            )
            model.fit(train_mat, target, sample_weight=weight)
            pred = np.asarray(model.predict(test_mat), dtype="float64")
            with np.errstate(over="ignore", invalid="ignore"):
                total += np.maximum(pred, 0.0)[test_code] * test_cap
        logger.info(
            "[FCST] family=cat n_units=%d kmeans_seed=%d train_rows=%d test_rows=%d sec=%.1f",
            int(n_units),
            int(kmeans_seed),
            len(train_mat),
            len(test_mat),
            time.perf_counter() - t0,
        )
    out: np.ndarray = total / (len(fam.cat_units) * len(fam.cat_seeds))
    if not np.isfinite(out).all() or (out < 0).any():
        raise FamilyError("prediction must be finite and non-negative")
    return out


def _mlp_design(train_mat: np.ndarray, test_mat: np.ndarray, feat_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    try:
        cs_idx = feat_cols.index("cs_ghi")
    except ValueError as exc:
        raise FamilyError("cs_ghi must be a mean feature") from exc
    medians = np.zeros(train_mat.shape[1], dtype="float64")
    observed = ~np.isnan(train_mat).all(axis=0)
    medians[observed] = np.nanmedian(train_mat[:, observed].astype("float64"), axis=0)
    train_filled = np.where(np.isnan(train_mat.astype("float64")), medians, train_mat.astype("float64"))
    test_filled = np.where(np.isnan(test_mat.astype("float64")), medians, test_mat.astype("float64"))
    cs_train = train_filled[:, cs_idx : cs_idx + 1]
    cs_test = test_filled[:, cs_idx : cs_idx + 1]
    train_ext = np.concatenate([train_filled, train_filled * cs_train], axis=1)
    test_ext = np.concatenate([test_filled, test_filled * cs_test], axis=1)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_ext)
    test_scaled = scaler.transform(test_ext)
    return train_scaled, test_scaled


def fit_predict_mlp(
    train: pd.DataFrame, test: pd.DataFrame, features: Sequence[str], site_geo: pd.DataFrame, model_cfg: ModelConfig
) -> np.ndarray:
    """mlp_units x mlp_seeds 평균 지역 MLP 예측 (len(test),) 발전소 kWh(≥0, 유한)를 반환한다. 위반 시 FamilyError."""
    fam = model_cfg.families
    if not fam.mlp_units or not fam.mlp_seeds:
        raise FamilyError("mlp_units and mlp_seeds must not be empty")
    total = np.zeros(len(test), dtype="float64")
    for n_units, kmeans_seed in fam.mlp_units:
        t0 = time.perf_counter()
        train_mat, test_mat, target, weight, test_code, test_cap, feat_cols = _common_panel(
            train, test, features, site_geo, model_cfg, int(n_units), int(kmeans_seed)
        )
        train_scaled, test_scaled = _mlp_design(train_mat, test_mat, feat_cols)
        train_scaled.flags.writeable = False
        target_c = np.asarray(target, dtype="float64")
        weight_c = np.asarray(weight, dtype="float64")

        def _fit_one(
            seed: int,
            _train: np.ndarray = train_scaled,
            _test: np.ndarray = test_scaled,
            _target: np.ndarray = target_c,
            _weight: np.ndarray = weight_c,
        ) -> np.ndarray:
            model = MLPRegressor(
                hidden_layer_sizes=tuple(int(h) for h in fam.mlp_hidden),
                alpha=float(fam.mlp_alpha),
                batch_size=int(fam.mlp_batch_size),
                learning_rate_init=float(fam.mlp_learning_rate_init),
                max_iter=int(fam.mlp_max_iter),
                early_stopping=True,
                validation_fraction=float(fam.mlp_validation_fraction),
                n_iter_no_change=int(fam.mlp_n_iter_no_change),
                random_state=int(seed),
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                model.fit(_train, _target, sample_weight=_weight)
            return np.asarray(model.predict(_test), dtype="float64")

        # Python 3.11의 경고 필터는 스레드 간 공유되므로 지역 경고 억제를 순차 실행한다.
        member_preds = [_fit_one(seed) for seed in fam.mlp_seeds]
        for pred in member_preds:
            with np.errstate(over="ignore", invalid="ignore"):
                total += np.maximum(pred, 0.0)[test_code] * test_cap
        logger.info(
            "[FCST] family=mlp n_units=%d kmeans_seed=%d train_rows=%d test_rows=%d sec=%.1f",
            int(n_units),
            int(kmeans_seed),
            len(train_mat),
            len(test_mat),
            time.perf_counter() - t0,
        )
    out: np.ndarray = total / (len(fam.mlp_units) * len(fam.mlp_seeds))
    if not np.isfinite(out).all() or (out < 0).any():
        raise FamilyError("prediction must be finite and non-negative")
    return out


def blend_families(base: np.ndarray, cat: np.ndarray | None, mlp: np.ndarray | None, cfg: FamilyConfig) -> np.ndarray:
    """(1-cw-mw)·base + cw·cat + mw·mlp를 시드별 (n_seeds, n_rows)로 반환한다. 모양·값 위반 시 FamilyError."""
    try:
        base_arr = np.asarray(base, dtype="float64")
    except (TypeError, ValueError, OverflowError) as exc:
        raise FamilyError("base must be numeric") from exc
    if base_arr.ndim != 2 or base_arr.shape[0] == 0:
        raise FamilyError("base must be a non-empty 2-D array")
    if not np.isfinite(base_arr).all() or (base_arr < 0).any():
        raise FamilyError("base must be finite and non-negative")
    n_rows = base_arr.shape[1]
    extras: list[tuple[np.ndarray | None, float, str]] = [
        (cat, float(cfg.cat_weight), "cat"),
        (mlp, float(cfg.mlp_weight), "mlp"),
    ]
    parts: list[np.ndarray] = []
    for arr, w, _name in extras:
        if w > 0.0:
            if arr is None:
                raise FamilyError("active family prediction must not be None")
            try:
                vec = np.asarray(arr, dtype="float64")
            except (TypeError, ValueError, OverflowError) as exc:
                raise FamilyError("family prediction must be numeric") from exc
            if vec.shape != (n_rows,):
                raise FamilyError("family prediction must match base rows")
            if not np.isfinite(vec).all() or (vec < 0).any():
                raise FamilyError("family prediction must be finite and non-negative")
            parts.append(vec)
        else:
            parts.append(np.zeros(n_rows, dtype="float64"))
    if cfg.cat_weight == 0.0 and cfg.mlp_weight == 0.0:
        return np.array(base_arr, dtype="float64", copy=True)
    cat_vec, mlp_vec = parts
    with np.errstate(over="ignore", invalid="ignore"):
        out: np.ndarray = (
            (1.0 - float(cfg.cat_weight) - float(cfg.mlp_weight)) * base_arr
            + float(cfg.cat_weight) * cat_vec
            + float(cfg.mlp_weight) * mlp_vec
        )
    if not np.isfinite(out).all() or (out < 0).any():
        raise FamilyError("prediction must be finite and non-negative")
    return out
