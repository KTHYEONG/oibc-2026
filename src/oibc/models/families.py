"""지역 단위 CatBoost·MLP 구성원 예측과 최종 혼합."""

from __future__ import annotations

import logging
import time
import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

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


def _mlp_design(
    train_mat: np.ndarray,
    test_mat: np.ndarray,
    feat_cols: list[str],
    clip_quantiles: tuple[float, float] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """학습 행렬로만 적합한 결측 대체·범위 제한·표준화를 적용한다. cs_ghi가 없으면 FamilyError.

    범위 제한 경계는 확장된 학습 행렬의 분위수로만 정하며, 시험 행렬 통계는 쓰지 않는다.
    """
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
    if clip_quantiles is not None:
        lo_q, hi_q = clip_quantiles
        lo_b = np.quantile(train_ext, lo_q, axis=0)
        hi_b = np.quantile(train_ext, hi_q, axis=0)
        train_ext = np.clip(train_ext, lo_b, hi_b)
        test_ext = np.clip(test_ext, lo_b, hi_b)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_ext)
    test_scaled = scaler.transform(test_ext)
    return train_scaled, test_scaled


def _fit_mlp_member(
    fam: FamilyConfig, seed: int, train: np.ndarray, target: np.ndarray, weight: np.ndarray, test: np.ndarray
) -> np.ndarray:
    # BLAS 스레드 수가 GEMM 분할을 바꿔 결과 비트가 달라지므로 1로 고정한다(작업자 수·기계 코어 수와 무관한 재현).
    with threadpool_limits(limits=1, user_api="blas"):
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
            model.fit(train, target, sample_weight=weight)
        return np.asarray(model.predict(test), dtype="float64")


def fit_predict_mlp(
    train: pd.DataFrame, test: pd.DataFrame, features: Sequence[str], site_geo: pd.DataFrame, model_cfg: ModelConfig
) -> np.ndarray:
    """mlp_units x mlp_seeds 평균 지역 MLP 예측 (len(test),) 발전소 kWh(≥0, 유한)를 반환한다. 위반 시 FamilyError.

    n_jobs > 1이면 구성원을 프로세스 풀에서 학습하며, 결과는 n_jobs와 무관하게 비트 단위로 같다.
    """
    fam = model_cfg.families
    if not fam.mlp_units or not fam.mlp_seeds:
        raise FamilyError("mlp_units and mlp_seeds must not be empty")
    t0 = time.perf_counter()
    designs: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    test_cap = np.zeros(0, dtype="float64")
    for n_units, kmeans_seed in fam.mlp_units:
        train_mat, test_mat, target, weight, test_code, test_cap, feat_cols = _common_panel(
            train, test, features, site_geo, model_cfg, int(n_units), int(kmeans_seed)
        )
        train_scaled, test_scaled = _mlp_design(train_mat, test_mat, feat_cols, fam.mlp_clip_quantiles)
        train_scaled.flags.writeable = False
        designs.append(
            (
                train_scaled,
                test_scaled,
                np.asarray(target, dtype="float64"),
                np.asarray(weight, dtype="float64"),
                test_code,
            )
        )
        logger.info(
            "[FCST] family=mlp n_units=%d kmeans_seed=%d train_rows=%d test_rows=%d",
            int(n_units),
            int(kmeans_seed),
            len(train_mat),
            len(test_mat),
        )
        del train_mat, test_mat
    jobs = [(d, int(seed)) for d in range(len(designs)) for seed in fam.mlp_seeds]
    # 큰 학습 행렬부터 배분해 마지막 대기열 꼬리를 줄인다. 합산은 아래에서 원래 순서로 한다.
    order = sorted(range(len(jobs)), key=lambda j: -len(designs[jobs[j][0]][0]))
    workers = min(int(model_cfg.n_jobs), len(jobs))
    calls = [
        (fam, jobs[j][1], designs[jobs[j][0]][0], designs[jobs[j][0]][2], designs[jobs[j][0]][3], designs[jobs[j][0]][1])
        for j in order
    ]
    if workers <= 1:
        results = [_fit_mlp_member(*call) for call in calls]
    else:
        results = Parallel(n_jobs=workers, backend="loky")(delayed(_fit_mlp_member)(*call) for call in calls)
    preds = dict(zip(order, results, strict=True))
    total = np.zeros(len(test), dtype="float64")
    for j, (d, _seed) in enumerate(jobs):
        with np.errstate(over="ignore", invalid="ignore"):
            total += np.maximum(preds[j], 0.0)[designs[d][4]] * test_cap
    logger.info(
        "[FCST] family=mlp members=%d workers=%d sec=%.1f", len(jobs), workers, time.perf_counter() - t0
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
