"""시드 앙상블 LightGBM 학습·예측."""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.oibc.config import ModelConfig

logger = logging.getLogger(__name__)


class ModelError(ValueError):
    """모델 입력 계약 위반."""


def _require_columns(panel: pd.DataFrame, cols: Sequence[str]) -> None:
    if panel.columns.has_duplicates:
        raise ModelError("duplicate columns")
    missing = [c for c in cols if c not in panel.columns]
    if missing:
        raise ModelError(f"missing columns {missing[:5]}")


def _numeric(panel: pd.DataFrame, cols: Sequence[str]) -> np.ndarray:
    _require_columns(panel, cols)
    try:
        return np.asarray(panel.loc[:, list(cols)].to_numpy(dtype="float64"), dtype="float64")
    except (ValueError, TypeError) as exc:
        raise ModelError("columns must be numeric") from exc


def _capacity(panel: pd.DataFrame) -> np.ndarray:
    cap = _numeric(panel, ["capacity_kw"])[:, 0]
    if not np.isfinite(cap).all() or (cap <= 0).any():
        raise ModelError("capacity_kw must be finite and positive")
    return cap


def _features(panel: pd.DataFrame, features: Sequence[str]) -> np.ndarray:
    if not features or len(set(features)) != len(features):
        raise ModelError("features must be non-empty and distinct")
    _require_columns(panel, features)
    # 열 단위로 C-order 버퍼에 채운다: DataFrame 전체 변환은 F-order 사본 + C-order 사본이 동시에 생겨 peak가 2배가 된다.
    data = np.empty((len(panel), len(features)), dtype=np.float32)
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            for j, col in enumerate(features):
                data[:, j] = panel[col].to_numpy(dtype="float32")
    except (ValueError, TypeError) as exc:
        raise ModelError("columns must be numeric") from exc
    if np.isinf(data).any():
        raise ModelError("features must not contain infinity")
    return data


def training_target(panel: pd.DataFrame, cfg: ModelConfig) -> np.ndarray:
    """이용률 타깃 clip(energy/capacity_kw, 0, target_clip_max)를 float32로 반환한다."""
    energy = _numeric(panel, ["energy"])[:, 0]
    if energy.size == 0 or not np.isfinite(energy).all():
        raise ModelError("energy must be finite for all train rows")
    cap = _capacity(panel)
    util = np.clip(energy / cap, 0.0, cfg.target_clip_max)
    out: np.ndarray = util.astype("float32")
    return out


def sample_weight(panel: pd.DataFrame, cfg: ModelConfig) -> np.ndarray:
    """가중치 모드별 학습 가중치(하한 min_weight)를 float64로 반환한다."""
    smp, k = _numeric(panel, ["smp", "k"]).T
    if not np.isfinite(smp).all() or not np.isfinite(k).all():
        raise ModelError("smp and k must be finite")
    cap = _capacity(panel)
    with np.errstate(over="ignore", invalid="ignore"):
        if cfg.weight == "smp_k_cap":
            raw = smp * k * cap
        elif cfg.weight == "k_cap":
            raw = k * cap
        else:
            raw = np.ones_like(cap)
    if not np.isfinite(raw).all():
        raise ModelError("weights must be finite")
    weights: np.ndarray = np.maximum(raw, cfg.min_weight).astype("float64")
    return weights


def fit_seed_models(train: pd.DataFrame, features: Sequence[str], cfg: ModelConfig) -> list[lgb.Booster]:
    """cfg.seeds 순서의 시드별 LightGBM을 시드마다 독립 Dataset으로 (동시) 학습해 반환한다."""
    if train.empty:
        raise ModelError("train must not be empty")
    data = _features(train, features)
    target = training_target(train, cfg)
    weight = sample_weight(train, cfg)
    data.flags.writeable = False
    target.flags.writeable = False
    weight.flags.writeable = False
    workers = min(cfg.seed_workers, len(cfg.seeds), cfg.n_jobs)
    threads = max(1, cfg.n_jobs // workers)
    n_rows, n_feats = data.shape

    def _fit_one(seed: int) -> lgb.Booster:
        params: dict[str, object] = {
            "objective": cfg.objective,
            "learning_rate": cfg.learning_rate,
            "num_leaves": cfg.num_leaves,
            "min_child_samples": cfg.min_child_samples,
            "subsample": cfg.subsample,
            "subsample_freq": cfg.subsample_freq,
            "colsample_bytree": cfg.colsample_bytree,
            "reg_lambda": cfg.reg_lambda,
            # 자동 col/row-wise 선택은 실행 시간 측정으로 정해져 프로세스마다 결과가 달라진다.
            "force_col_wise": True,
            "num_threads": threads,
            "seed": seed,
            "verbose": -1,
        }
        dataset = lgb.Dataset(data, label=target, weight=weight, free_raw_data=False)
        t0 = time.perf_counter()
        booster = lgb.train(params, dataset, num_boost_round=cfg.n_rounds)
        sec = time.perf_counter() - t0
        logger.info("[FCST] fit seed=%d rows=%d feats=%d threads=%d sec=%.1f", seed, n_rows, n_feats, threads, sec)
        return booster

    t_all = time.perf_counter()
    if workers <= 1:
        models = [_fit_one(seed) for seed in cfg.seeds]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            models = list(pool.map(_fit_one, cfg.seeds))
    logger.info("[FCST] fit_all seeds=%d workers=%d sec=%.1f", len(cfg.seeds), workers, time.perf_counter() - t_all)
    return models


def predict_seed_kwh(
    models: Sequence[lgb.Booster], frame: pd.DataFrame, features: Sequence[str], *, num_threads: int | None = None,
) -> np.ndarray:
    """(n_seeds, n_rows) 발전소 kWh 예측(≥0, 유한)을 반환한다. num_threads는 예측 스레드 수(None이면 모델 설정)."""
    if len(models) == 0:
        raise ModelError("models must not be empty")
    if num_threads is not None and num_threads < 1:
        raise ModelError("num_threads must be >= 1")
    data = _features(frame, features)
    cap = _capacity(frame)
    rows = np.empty((len(models), len(frame)), dtype="float64")
    if frame.empty:
        return rows
    for i, model in enumerate(models):
        pred = np.asarray(
            model.predict(data) if num_threads is None else model.predict(data, num_threads=num_threads),
            dtype="float64",
        )
        if pred.shape != (len(frame),) or not np.isfinite(pred).all():
            raise ModelError("prediction must be a finite row vector")
        with np.errstate(over="ignore", invalid="ignore"):
            scaled = np.maximum(pred, 0.0) * cap
        if not np.isfinite(scaled).all():
            raise ModelError("prediction must be finite")
        rows[i] = scaled
    return rows


def bag(preds: np.ndarray) -> np.ndarray:
    """시드 축 평균 예측을 반환한다."""
    if preds.ndim != 2 or preds.shape[0] == 0:
        raise ModelError("preds must be a non-empty 2-D array")
    if not np.isfinite(preds).all() or (preds < 0).any():
        raise ModelError("preds must be finite and non-negative")
    mean: np.ndarray = np.asarray((preds / preds.shape[0]).sum(axis=0), dtype="float64")
    return mean
