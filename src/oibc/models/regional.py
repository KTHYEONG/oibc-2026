"""지역 단위 집계 직접 예측(용량 가중 단면 패널)."""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from src.oibc.config import ModelConfig, RegionalConfig
from src.oibc.models.gbm import ModelError

logger = logging.getLogger(__name__)


class RegionalError(ModelError):
    """지역 모델 입력 계약 위반."""


def _site_arrays(site_geo: pd.DataFrame) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    if site_geo.columns.has_duplicates:
        raise RegionalError("duplicate columns")
    missing = [c for c in ("site_id", "lat", "lon", "capacity_kw") if c not in site_geo.columns]
    if missing:
        raise RegionalError(f"missing columns {missing[:5]}")
    ids = site_geo["site_id"].tolist()
    if len(ids) == 0 or any(not isinstance(v, str) or not v for v in ids) or len(set(ids)) != len(ids):
        raise RegionalError("site_id must be unique non-empty strings")
    try:
        lat = np.asarray(site_geo["lat"].to_numpy(dtype="float64"), dtype="float64")
        lon = np.asarray(site_geo["lon"].to_numpy(dtype="float64"), dtype="float64")
        cap = np.asarray(site_geo["capacity_kw"].to_numpy(dtype="float64"), dtype="float64")
    except (ValueError, TypeError) as exc:
        raise RegionalError("site_geo columns must be numeric") from exc
    if not np.isfinite(lat).all() or not np.isfinite(lon).all():
        raise RegionalError("lat and lon must be finite")
    if not np.isfinite(cap).all() or (cap <= 0).any():
        raise RegionalError("capacity_kw must be finite and positive")
    return ids, lat, lon, cap


def site_units(site_geo: pd.DataFrame, n_units: int, kmeans_seed: int, cfg: RegionalConfig) -> pd.Series:
    """site_id(string index) → unit 라벨(int64, 0..n_units-1)을 반환한다. 입력 위반 시 RegionalError."""
    ids, lat, lon, cap = _site_arrays(site_geo)
    if isinstance(n_units, bool) or not isinstance(n_units, (int, np.integer)) or n_units < 1:
        raise RegionalError("n_units must be an integer >= 1")
    if isinstance(kmeans_seed, bool) or not isinstance(kmeans_seed, (int, np.integer)) or kmeans_seed < 0:
        raise RegionalError("kmeans_seed must be an integer >= 0")
    n_units, kmeans_seed = int(n_units), int(kmeans_seed)
    if n_units > len(ids):
        raise RegionalError("n_units must not exceed site count")
    if n_units == 1:
        labels = np.zeros(len(ids), dtype="int64")
    else:
        coords = np.column_stack((lat, lon * math.cos(math.radians(cfg.ref_lat_deg))))
        labels = np.asarray(
            KMeans(n_clusters=n_units, n_init=cfg.kmeans_n_init, random_state=kmeans_seed).fit_predict(
                coords, sample_weight=cap
            ),
            dtype="int64",
        )
    return pd.Series(labels, index=pd.Index(ids, dtype="object"))


def unit_centroids(site_geo: pd.DataFrame, units: pd.Series) -> pd.DataFrame:
    """unit index별 (unit_lat, unit_lon) 평균 좌표를 반환한다."""
    ids, lat, lon, _ = _site_arrays(site_geo)
    if units.index.has_duplicates:
        raise RegionalError("duplicate unit index")
    try:
        order = [units.loc[i] for i in ids]
    except KeyError as exc:
        raise RegionalError("units must cover every site_id") from exc
    labels = np.asarray(order)
    if labels.dtype.kind not in {"i", "u"}:
        raise RegionalError("unit labels must be integers")
    uniq = np.unique(labels)
    lat_mean = np.array([lat[labels == u].mean() for u in uniq])
    lon_mean = np.array([lon[labels == u].mean() for u in uniq])
    return pd.DataFrame({"unit_lat": lat_mean, "unit_lon": lon_mean}, index=pd.Index(uniq))[["unit_lat", "unit_lon"]]


def _panel_matrix(
    frame: pd.DataFrame,
    mean_cols: Sequence[str],
    sd_vars: Sequence[str],
    unit_of: dict[str, int],
    centroids: pd.DataFrame,
) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray, np.ndarray]:
    if frame.columns.has_duplicates:
        raise RegionalError("duplicate columns")
    missing = [c for c in ("site_id", "ts", "capacity_kw", *mean_cols) if c not in frame.columns]
    if missing:
        raise RegionalError(f"missing columns {missing[:5]}")
    try:
        cap = np.asarray(frame["capacity_kw"].to_numpy(dtype="float64"), dtype="float64")
    except (ValueError, TypeError) as exc:
        raise RegionalError("capacity_kw must be numeric") from exc
    if not np.isfinite(cap).all() or (cap <= 0).any():
        raise RegionalError("capacity_kw must be finite and positive")
    site_pos = pd.Index(list(unit_of), dtype="object").get_indexer(frame["site_id"].astype("object"))
    if (site_pos < 0).any():
        raise RegionalError("units must cover every site_id")
    unit_arr = np.fromiter(unit_of.values(), dtype="int64", count=len(unit_of))[site_pos]
    if frame["ts"].isna().any():
        raise RegionalError("ts must not contain missing values")
    try:
        ts_i8 = pd.DatetimeIndex(frame["ts"]).as_unit("ns").asi8
    except (TypeError, ValueError, OverflowError) as exc:
        raise RegionalError("ts values must be mutually comparable") from exc
    # (unit, ts) 정렬 순서를 정수 키 하나로 만들어 Timestamp 객체 루프 없이 factorize한다.
    ts_code, ts_uniq = pd.factorize(ts_i8, sort=True)
    pair_key, row_code = np.unique(unit_arr * len(ts_uniq) + ts_code, return_inverse=True)
    row_code = row_code.astype("int64", copy=False)
    n_panel = len(pair_key)
    panel_units = pair_key // len(ts_uniq)
    panel_cap = np.bincount(row_code, weights=cap, minlength=n_panel).astype("float64")
    cen_pos = centroids.index.get_indexer(panel_units)
    if (cen_pos < 0).any():
        raise RegionalError("centroids must cover every unit")
    panel_lat = centroids["unit_lat"].to_numpy(dtype="float64")[cen_pos]
    panel_lon = centroids["unit_lon"].to_numpy(dtype="float64")[cen_pos]
    means: dict[str, np.ndarray] = {}
    sum_sq: dict[str, np.ndarray] = {}
    for col in mean_cols:
        try:
            x = np.asarray(frame[col].to_numpy(dtype="float64"), dtype="float64")
        except (ValueError, TypeError) as exc:
            raise RegionalError(f"feature {col!r} must be numeric") from exc
        if np.isinf(x).any():
            raise RegionalError(f"feature {col!r} must not contain infinity")
        valid = ~np.isnan(x)
        denom = np.bincount(row_code[valid], weights=cap[valid], minlength=n_panel)
        num = np.bincount(row_code[valid], weights=(cap[valid] * x[valid]), minlength=n_panel)
        mean = np.full(n_panel, np.nan)
        hit = denom > 0
        mean[hit] = num[hit] / denom[hit]
        means[col] = mean
        if col in set(sd_vars):
            num2 = np.bincount(row_code[valid], weights=(cap[valid] * x[valid] * x[valid]), minlength=n_panel)
            ex2 = np.full(n_panel, np.nan)
            ex2[hit] = num2[hit] / denom[hit]
            sum_sq[col] = ex2
    feat_cols = [*mean_cols, *(f"{v}_sd" for v in sd_vars), "unit_lat", "unit_lon"]
    mat = np.empty((n_panel, len(feat_cols)), dtype="float32")
    for j, col in enumerate(mean_cols):
        mat[:, j] = means[col]
    for j, v in enumerate(sd_vars):
        var = sum_sq[v] - means[v] * means[v]
        mat[:, len(mean_cols) + j] = np.sqrt(np.clip(var, 0.0, None))
    mat[:, -2] = panel_lat
    mat[:, -1] = panel_lon
    return mat, feat_cols, row_code, panel_cap, panel_units


def _panel_targets(
    frame: pd.DataFrame, row_code: np.ndarray, n_panel: int, panel_cap: np.ndarray, cfg: ModelConfig
) -> tuple[np.ndarray, np.ndarray]:
    missing = [c for c in ("energy", "smp", "k") if c not in frame.columns]
    if missing:
        raise RegionalError(f"missing columns {missing[:5]}")
    try:
        energy = np.asarray(frame["energy"].to_numpy(dtype="float64"), dtype="float64")
        smp = np.asarray(frame["smp"].to_numpy(dtype="float64"), dtype="float64")
        k = np.asarray(frame["k"].to_numpy(dtype="float64"), dtype="float64")
    except (ValueError, TypeError) as exc:
        raise RegionalError("energy, smp and k must be numeric") from exc
    if energy.size == 0 or not np.isfinite(energy).all():
        raise RegionalError("energy must be finite for all train rows")
    if not np.isfinite(smp).all() or not np.isfinite(k).all():
        raise RegionalError("smp and k must be finite")
    order = np.argsort(row_code, kind="stable")
    sorted_code = row_code[order]
    starts = np.flatnonzero(np.diff(sorted_code, prepend=np.int64(-1)) != 0)
    first_pos = np.empty(n_panel, dtype="int64")
    first_pos[sorted_code[starts]] = order[starts]
    if (smp != smp[first_pos[row_code]]).any() or (k != k[first_pos[row_code]]).any():
        raise RegionalError("smp and k must be constant within (unit, ts)")
    energy_sum = np.bincount(row_code, weights=energy, minlength=n_panel).astype("float64")
    target = np.clip(energy_sum / panel_cap, 0.0, cfg.target_clip_max).astype("float32")
    panel_smp, panel_k = smp[first_pos], k[first_pos]
    with np.errstate(over="ignore", invalid="ignore"):
        if cfg.weight == "smp_k_cap":
            raw = panel_smp * panel_k * panel_cap
        elif cfg.weight == "k_cap":
            raw = panel_k * panel_cap
        else:
            raw = np.ones_like(panel_cap)
    if not np.isfinite(raw).all():
        raise RegionalError("weights must be finite")
    return target, np.maximum(raw, cfg.min_weight).astype("float64")


def fit_predict_regional(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: Sequence[str],
    site_geo: pd.DataFrame,
    model_cfg: ModelConfig,
) -> np.ndarray:
    """구성원 평균 지역 예측 (n_seeds, len(test)) 발전소 kWh(≥0, 유한)를 반환한다. 계약 위반 시 RegionalError."""
    if not features or len(set(features)) != len(features):
        raise RegionalError("features must be non-empty and distinct")
    if train.empty or test.empty:
        raise RegionalError("train and test must not be empty")
    cfg = model_cfg.regional
    if not cfg.units:
        raise RegionalError("units must not be empty")
    mean_cols = [f for f in features if f not in set(cfg.excluded_vars)]
    for v in cfg.sd_vars:
        if v not in mean_cols:
            raise RegionalError(f"sd_var {v!r} must be a mean feature")
    try:
        test_cap = np.asarray(test["capacity_kw"].to_numpy(dtype="float64"), dtype="float64")
    except (ValueError, TypeError, KeyError) as exc:
        raise RegionalError("capacity_kw must be numeric") from exc
    if not np.isfinite(test_cap).all() or (test_cap <= 0).any():
        raise RegionalError("capacity_kw must be finite and positive")
    seeds = list(model_cfg.seeds)
    workers = min(model_cfg.seed_workers, len(seeds), model_cfg.n_jobs)
    threads = max(1, model_cfg.n_jobs // workers)
    total = np.zeros((len(seeds), len(test)), dtype="float64")
    for n_units, kmeans_seed in cfg.units:
        t0 = time.perf_counter()
        member_units = site_units(site_geo, n_units, kmeans_seed, cfg)
        centroids = unit_centroids(site_geo, member_units)
        unit_of = {k: int(v) for k, v in member_units.items()}
        train_mat, _, train_code, train_cap, _ = _panel_matrix(train, mean_cols, cfg.sd_vars, unit_of, centroids)
        test_mat, _, test_code, _, _ = _panel_matrix(test, mean_cols, cfg.sd_vars, unit_of, centroids)
        target, weight = _panel_targets(train, train_code, len(train_mat), train_cap, model_cfg)
        train_mat.flags.writeable = False
        target.flags.writeable = False
        weight.flags.writeable = False

        def _fit_one(
            seed: int,
            _mat: np.ndarray = train_mat,
            _target: np.ndarray = target,
            _weight: np.ndarray = weight,
        ) -> lgb.Booster:
            params: dict[str, object] = {
                "objective": "l1",
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
            dataset = lgb.Dataset(_mat, label=_target, weight=_weight, free_raw_data=False)
            return lgb.train(params, dataset, num_boost_round=cfg.n_rounds)

        if workers <= 1:
            models = [_fit_one(seed) for seed in seeds]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                models = list(pool.map(_fit_one, seeds))
        for i, model in enumerate(models):
            pred = np.asarray(model.predict(test_mat), dtype="float64")
            with np.errstate(over="ignore", invalid="ignore"):
                total[i] += np.maximum(pred, 0.0)[test_code] * test_cap
        logger.info(
            "[FCST] regional n_units=%d kmeans_seed=%d train_rows=%d test_rows=%d sec=%.1f",
            n_units,
            kmeans_seed,
            len(train_mat),
            len(test_mat),
            time.perf_counter() - t0,
        )
    out = total / len(cfg.units)
    if not np.isfinite(out).all() or (out < 0).any():
        raise RegionalError("prediction must be finite and non-negative")
    return out


def blend_seed_preds(plant: np.ndarray, regional: np.ndarray, weight: float) -> np.ndarray:
    """(1-weight)*plant + weight*regional을 시드별로 반환한다. 모양·값 위반 시 RegionalError."""
    try:
        w = float(weight)
    except (ValueError, TypeError) as exc:
        raise RegionalError("weight must be a real number") from exc
    if not math.isfinite(w) or not 0.0 <= w <= 1.0:
        raise RegionalError("weight must be finite and in [0, 1]")
    plant_arr = np.asarray(plant, dtype="float64")
    regional_arr = np.asarray(regional, dtype="float64")
    if plant_arr.ndim != 2 or regional_arr.ndim != 2 or plant_arr.shape != regional_arr.shape:
        raise RegionalError("plant and regional must share the same 2-D shape")
    if not np.isfinite(plant_arr).all() or (plant_arr < 0).any():
        raise RegionalError("plant must be finite and non-negative")
    if not np.isfinite(regional_arr).all() or (regional_arr < 0).any():
        raise RegionalError("regional must be finite and non-negative")
    if w == 0.0:
        return np.array(plant_arr, dtype="float64", copy=True)
    out: np.ndarray = (1.0 - w) * plant_arr + w * regional_arr
    return out
