"""주간 모델 패널 구축."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.oibc.config import CompetitionSpec, DataPaths, FeatureConfig
from src.oibc.features.weather import (
    FeatureError,
    _check_forecast_keys,
    clear_sky_ghi,
    day_features,
    fleet_features,
    neighbor_features,
    round_revision_features,
)
from src.oibc.timeaxis import op_date, require_hourly_kst

logger = logging.getLogger(__name__)

_DT = "datetime64[us, Asia/Seoul]"

PANEL_KEYS: tuple[str, ...] = ("site_id", "ts", "op_date", "period")
NON_FEATURE_COLS: frozenset[str] = frozenset(
    {*PANEL_KEYS, "energy", "smp", "k", "current_group", "wa_station", "wb_station"}
)
BASE_COLS: tuple[str, ...] = ("elev", "azim", "cosz", "doy_sin", "doy_cos", "hour", "capacity_kw", "lat", "lon")


class PanelError(ValueError):
    """패널 계약 위반."""


def _rounds() -> tuple[int, int]:
    return CompetitionSpec().rounds


def _day_columns(cfg: FeatureConfig) -> list[str]:
    cols: list[str] = []
    for name in (*cfg.day_mean_vars, *cfg.day_max_vars):
        cols.extend(f"{name}_d{off:+d}" for off in cfg.day_shift_offsets)
        cols.append(f"{name}_roll{cfg.day_roll_window}")
    cols.extend(f"{name}_daymean" for name in cfg.day_mean_vars)
    cols.extend(f"{name}_daymax" for name in cfg.day_max_vars)
    cols.extend(f"{name}_daysum" for name in cfg.day_sum_vars)
    return list(dict.fromkeys(cols))


def _all_feature_columns(cfg: FeatureConfig) -> list[str]:
    rounds = _rounds()
    cols = list(BASE_COLS)
    cols.extend(f"{v}_r{r}" for v in (*cfg.wa_vars, *cfg.wb_vars) for r in rounds)
    cols.extend(f"{v}_r{r}_n{off:+d}" for v in cfg.neighbor_vars for r in rounds for off in cfg.neighbor_offsets)
    cols.append("cs_ghi")
    for r in rounds:
        cols.extend((f"cs_clear_wa_r{r}", f"cs_clear_wb_r{r}", f"uv_per_cosz_r{r}"))
    for v in cfg.rd_vars:
        cols.extend((f"{v}_rmean", f"{v}_rdiff"))
    cols.extend(_day_columns(cfg))
    cols.extend(f"{v}_fleet" for v in cfg.fleet_vars)
    cols.extend(f"{v}_vsfleet" for v in cfg.vsfleet_vars)
    return list(dict.fromkeys(cols))


def feature_columns(panel: pd.DataFrame, cfg: FeatureConfig) -> list[str]:
    """cfg.groups에 맞는 모델 입력 열 목록을 결정적 순서로 반환한다."""
    rounds = _rounds()
    cols = list(BASE_COLS)
    cols.extend(f"{v}_r{r}" for v in (*cfg.wa_vars, *cfg.wb_vars) for r in rounds)
    cols.extend(f"{v}_r{r}_n{off:+d}" for v in cfg.neighbor_vars for r in rounds for off in cfg.neighbor_offsets)
    if "cs" in cfg.groups:
        cols.append("cs_ghi")
        for r in rounds:
            cols.extend((f"cs_clear_wa_r{r}", f"cs_clear_wb_r{r}", f"uv_per_cosz_r{r}"))
    if "rd" in cfg.groups:
        for v in cfg.rd_vars:
            cols.extend((f"{v}_rmean", f"{v}_rdiff"))
    if "day" in cfg.groups:
        cols.extend(_day_columns(cfg))
    if "fleet" in cfg.groups:
        cols.extend(f"{v}_fleet" for v in cfg.fleet_vars)
        cols.extend(f"{v}_vsfleet" for v in cfg.vsfleet_vars)
    missing = [c for c in cols if c not in panel.columns]
    if missing:
        raise PanelError(f"panel: missing feature columns {missing[:5]}")
    return list(dict.fromkeys(cols))


def _left_join(
    panel: pd.DataFrame, right: pd.DataFrame, left_on: list[str], right_on: list[str] | None = None
) -> pd.DataFrame:
    return panel.merge(right, left_on=left_on, right_on=right_on or left_on, how="left", validate="many_to_one")


def _check_keys(frame: pd.DataFrame, keys: list[str]) -> None:
    if frame[keys].isna().any().any() or frame.duplicated(keys).any():
        raise PanelError(f"panel: missing or duplicate keys {keys}")
    if "ts" in keys:
        require_hourly_kst(frame["ts"], "panel.ts")
    if "op_date" in frame and not frame["op_date"].eq(op_date(frame["ts"])).all():
        raise PanelError("panel: inconsistent op_date")


def _doy_sin_cos(ts: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    doy = pd.DatetimeIndex(ts).dayofyear.to_numpy(dtype="float64")
    ang = 2.0 * np.pi * (doy - 1.0) / 365.0
    return np.sin(ang).astype("float32"), np.cos(ang).astype("float32")


def _provider_subset(names: tuple[str, ...], prefix: str) -> tuple[str, ...]:
    return tuple(v for v in names if v.startswith(prefix))


def build_panel(paths: DataPaths, spec: CompetitionSpec, cfg: FeatureConfig) -> pd.DataFrame:
    """학습(관측 발전량 주간 행)과 평가(253x8760 격자 주간 행) 모델 패널을 반환한다."""
    t0 = time.perf_counter()
    rounds = tuple(sorted(spec.rounds))
    sites = pd.read_parquet(paths.canonical_dir / "sites.parquet")
    generation = pd.read_parquet(paths.canonical_dir / "generation.parquet")
    smp = pd.read_parquet(paths.canonical_dir / "smp.parquet")
    grid = pd.read_parquet(paths.derived_dir / "grid.parquet")
    solar = pd.read_parquet(paths.derived_dir / "solar.parquet")
    if len(sites) != spec.n_sites:
        raise PanelError(f"sites: rows {len(sites)} != {spec.n_sites}")
    for frame, keys in (
        (sites, ["site_id"]),
        (generation, ["site_id", "ts"]),
        (grid, ["site_id", "ts"]),
        (solar, ["site_id", "ts"]),
        (smp, ["ts"]),
    ):
        _check_keys(frame, keys)
    if (generation.loc[generation["energy"].notna(), "ts"] > spec.train_end).any():
        raise PanelError("generation: evaluation labels forbidden")
    if not generation["site_id"].isin(sites["site_id"]).all():
        raise PanelError("generation: unknown site")
    expected_grid = pd.MultiIndex.from_product(
        [sites["site_id"], pd.date_range(spec.eval_start, spec.eval_end, freq="h")], names=["site_id", "ts"]
    ).sort_values()
    actual_grid = pd.MultiIndex.from_frame(grid.loc[grid["period"] == "eval", ["site_id", "ts"]]).sort_values()
    if not actual_grid.equals(expected_grid):
        raise PanelError("panel: eval grid incomplete or misaligned")

    train_obs = generation[generation["energy"].notna()].copy()
    train = _left_join(train_obs, solar, ["site_id", "ts"])
    if train["elev"].isna().any():
        raise PanelError("panel: missing solar for train rows")
    train = train[train["elev"] > cfg.day_min_elev_deg].copy()
    train["period"] = "train"
    del train_obs

    grid_eval = grid[grid["period"] == "eval"].copy()
    eval_panel = _left_join(grid_eval, solar, ["site_id", "ts"])
    if eval_panel["elev"].isna().any():
        raise PanelError("panel: missing solar for eval rows")
    eval_panel = eval_panel[eval_panel["elev"] > cfg.day_min_elev_deg].copy()
    eval_panel["energy"] = np.float32(np.nan)
    del grid_eval
    if eval_panel["site_id"].nunique() != spec.n_sites:
        raise PanelError("panel: eval rows missing some sites")

    site_meta = sites.set_index("site_id")
    if not np.isfinite(sites[["capacity_kw", "lat", "lon"]].to_numpy(dtype="float64")).all():
        raise PanelError("sites: non-finite metadata")
    for frame, is_train in ((train, True), (eval_panel, False)):
        idx = pd.Index(frame["site_id"])
        frame["capacity_kw"] = site_meta["capacity_kw"].reindex(idx).to_numpy(dtype="float32")
        frame["lat"] = site_meta["lat"].reindex(idx).to_numpy(dtype="float64")
        frame["lon"] = site_meta["lon"].reindex(idx).to_numpy(dtype="float64")
        frame["current_group"] = site_meta["current_group"].reindex(idx).array
        if is_train:
            frame["wa_station"] = site_meta["wa_station"].reindex(idx).array
            frame["wb_station"] = site_meta["wb_station"].reindex(idx).array
    del site_meta, grid, solar

    panel = pd.concat([train, eval_panel], ignore_index=True)
    del train, eval_panel
    panel["energy"] = panel["energy"].astype("float32")

    smp_small = smp[["ts", "smp", "k"]].copy()
    panel = _left_join(panel, smp_small, ["ts"])
    del smp_small, smp
    if panel[["smp", "k"]].isna().any().any():
        raise PanelError("panel: missing smp for some rows")

    panel["site_id"] = panel["site_id"].astype("string")
    panel["ts"] = panel["ts"].astype(_DT)
    panel["op_date"] = panel["op_date"].astype(_DT)
    panel["period"] = panel["period"].astype("string")
    panel["current_group"] = panel["current_group"].astype("string")
    panel["wa_station"] = panel["wa_station"].astype("string")
    panel["wb_station"] = panel["wb_station"].astype("string")
    sin_v, cos_v = _doy_sin_cos(panel["ts"])
    panel["doy_sin"] = sin_v
    panel["doy_cos"] = cos_v
    panel["hour"] = pd.DatetimeIndex(panel["ts"]).hour.to_numpy(dtype="int8")
    for col in ("elev", "azim", "cosz", "capacity_kw"):
        panel[col] = panel[col].to_numpy(dtype="float64").astype("float32")
    panel["lat"] = panel["lat"].to_numpy(dtype="float64")
    panel["lon"] = panel["lon"].to_numpy(dtype="float64")

    for provider, station in (("wa", "wa_station"), ("wb", "wb_station")):
        prefix = f"{provider}_"
        filled = pd.read_parquet(paths.derived_dir / f"forecast_filled_{provider}.parquet")
        try:
            _check_forecast_keys(filled, station, spec)
        except FeatureError as exc:
            raise PanelError(f"forecast_{provider}: {exc}") from exc
        var_all = tuple(v for v in (*cfg.wa_vars, *cfg.wb_vars) if v.startswith(prefix))
        for r in rounds:
            sub = filled[filled["round"] == r][[station, "ts", *var_all]].rename(
                columns={v: f"{v}_r{r}" for v in var_all}
            )
            panel = _left_join(panel, sub, [station, "ts"])
            del sub
        nb_vars = _provider_subset(cfg.neighbor_vars, prefix)
        if nb_vars:
            nb = neighbor_features(filled, station, nb_vars, list(cfg.neighbor_offsets), spec)
            panel = _left_join(panel, nb, [station, "ts"])
            del nb
        sub_cfg = cfg.model_copy(
            update={
                "day_mean_vars": _provider_subset(cfg.day_mean_vars, prefix),
                "day_max_vars": _provider_subset(cfg.day_max_vars, prefix),
                "day_sum_vars": _provider_subset(cfg.day_sum_vars, prefix),
            }
        )
        if sub_cfg.day_mean_vars or sub_cfg.day_max_vars or sub_cfg.day_sum_vars:
            day = day_features(filled, station, sub_cfg, spec)
            panel = _left_join(panel, day, [station, "ts"])
            del day
        fleet_vars = _provider_subset(cfg.fleet_vars, prefix)
        if fleet_vars:
            fleet = fleet_features(filled, fleet_vars, cfg.day_round)
            panel = _left_join(panel, fleet, ["ts"])
            del fleet
        del filled

    revision = round_revision_features(
        panel[["ts", *(f"{v}_r{r}" for v in cfg.rd_vars for r in rounds)]], cfg.rd_vars, spec
    )
    for var in cfg.rd_vars:
        panel[f"{var}_rmean"] = revision[f"{var}_rmean"]
        panel[f"{var}_rdiff"] = revision[f"{var}_rdiff"]
    for var in cfg.vsfleet_vars:
        panel[f"{var}_vsfleet"] = (panel[f"{var}_r{cfg.day_round}"] - panel[f"{var}_fleet"]).to_numpy(dtype="float64").astype("float32")

    cosz64 = panel["cosz"].to_numpy(dtype="float64")
    doy = pd.DatetimeIndex(panel["ts"]).dayofyear.to_numpy(dtype="float64")
    panel["cs_ghi"] = np.asarray(clear_sky_ghi(cosz64, doy), dtype="float32")
    for r in rounds:
        panel[f"cs_clear_wa_r{r}"] = (panel["cs_ghi"] * (1.0 - panel[f"wa_cloud_r{r}"] / 100.0)).to_numpy(dtype="float64").astype("float32")
        panel[f"cs_clear_wb_r{r}"] = (panel["cs_ghi"] * (1.0 - panel[f"wb_cloud_r{r}"] / 100.0)).to_numpy(dtype="float64").astype("float32")
        panel[f"uv_per_cosz_r{r}"] = (panel[f"wb_uv_idx_r{r}"] / np.maximum(cosz64, 0.05)).astype("float32")

    for col in panel.select_dtypes(include="float64").columns:
        panel[col] = panel[col].astype("float32")

    feat_cols = _all_feature_columns(cfg)
    train_mask = (panel["period"] == "train").to_numpy()
    eval_mask = (panel["period"] == "eval").to_numpy()
    parity_bad: list[str] = []
    nan_summary: list[str] = []
    for col in feat_cols:
        series = panel[col]
        n_train = int(series[train_mask].isna().sum()) if train_mask.any() else 0
        n_eval = int(series[eval_mask].isna().sum()) if eval_mask.any() else 0
        if n_train + n_eval > 0:
            nan_summary.append(f"{col}:{n_train}/{n_eval}")
        if n_eval > 0 and n_train == 0:
            parity_bad.append(col)
    logger.info("[DATA] panel NaN train/eval %s", ";".join(nan_summary))
    if parity_bad:
        raise PanelError(f"panel: eval-only NaN columns {parity_bad[:5]}")

    ordered = [*PANEL_KEYS, "energy", "current_group", "wa_station", "wb_station", "smp", "k", *feat_cols]
    panel = panel[ordered].sort_values(["site_id", "ts"]).reset_index(drop=True)
    elapsed = time.perf_counter() - t0
    logger.info(
        "[DATA] panel rows_train=%d rows_eval=%d cols=%d sec=%.1f",
        int(train_mask.sum()),
        int(eval_mask.sum()),
        len(panel.columns),
        elapsed,
    )
    return panel


def build_features(paths: DataPaths, spec: CompetitionSpec, cfg: FeatureConfig) -> Path:
    """build_panel 결과를 `features_dir/panel.parquet`(zstd)에 쓰고 경로를 반환한다."""
    out_dir = paths.features_dir
    out = out_dir / "panel.parquet"
    if any(out.resolve().is_relative_to(p.resolve()) for p in (paths.raw_dir, paths.canonical_dir, paths.derived_dir)):
        raise PanelError("features_dir: output must not overwrite input data")
    panel = build_panel(paths, spec, cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out, compression="zstd", index=False)
    return out
