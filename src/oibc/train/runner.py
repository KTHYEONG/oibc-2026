"""폴드 검증·평가 예측 실행기."""

from __future__ import annotations

import gc
import hashlib
import logging
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from src.oibc.config import (
    ArtifactPaths,
    CompetitionSpec,
    DataPaths,
    RunConfig,
    ValidationConfig,
)
from src.oibc.cv.splits import SplitError, assert_causal_split, fold_masks, site_history
from src.oibc.data.schemas import _DT
from src.oibc.evaluate.compare import DailyRuns, compare_runs, comparison_to_dict
from src.oibc.evaluate.registry import ExperimentRecord, append_record, read_records
from src.oibc.evaluate.settlement import aggregate_groups, daily_totals, settle
from src.oibc.features.panel import feature_columns
from src.oibc.models.gbm import bag, fit_seed_models, predict_seed_kwh
from src.oibc.timeaxis import k_weight, op_date, require_hourly_kst

logger = logging.getLogger(__name__)

_BAG_NOTE = "운영 예측은 seed-bag(분산 축소, 채택 규칙 예외; probe-fe-20261016 근거)"


class RunnerError(ValueError):
    """실행기 계약 위반."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RunnerError(message)


def reference_assignment(sites: pd.DataFrame) -> pd.DataFrame:
    """current_group A/B를 G1/G2로 바꾼 (site_id, group) 배정을 반환한다."""
    if "site_id" not in sites.columns or "current_group" not in sites.columns:
        raise RunnerError("sites: missing site_id or current_group")
    if sites.empty:
        raise RunnerError("sites: empty")
    mapping = {"A": "G1", "B": "G2"}
    groups = sites["current_group"].astype("string")
    if bool((~groups.isin(set(mapping))).any()):
        raise RunnerError("sites: unknown current_group")
    out = pd.DataFrame(
        {
            "site_id": sites["site_id"].astype("string").reset_index(drop=True),
            "group": groups.map(mapping).astype("string").reset_index(drop=True),
        }
    )
    if out["site_id"].isna().any() or out.duplicated(subset=["site_id"]).any():
        raise RunnerError("sites: bad site_id")
    return out.sort_values("site_id", kind="mergesort").reset_index(drop=True)


def _actual_col(frame: pd.DataFrame) -> str:
    if "energy" in frame.columns:
        return "energy"
    if "actual_kwh" in frame.columns:
        return "actual_kwh"
    raise RunnerError("frame: missing energy column")


def site_frame(day_rows: pd.DataFrame, day_pred_kwh: np.ndarray, night_rows: pd.DataFrame) -> pd.DataFrame:
    """주간 예측과 야간 0 예측을 합친 (site_id, ts, pred_kwh, actual_kwh) 정산 입력을 반환한다."""
    try:
        preds = np.asarray(day_pred_kwh, dtype="float64")
    except (ValueError, TypeError) as exc:
        raise RunnerError("day_pred_kwh: must be numeric") from exc
    if preds.ndim != 1 or preds.shape[0] != len(day_rows):
        raise RunnerError("day_pred_kwh: length must match day_rows")
    if not np.all(np.isfinite(preds)) or bool((preds < 0).any()):
        raise RunnerError("day_pred_kwh: must be finite and >= 0")
    day_col = _actual_col(day_rows) if len(day_rows) else "energy"
    night_col = _actual_col(night_rows) if len(night_rows) else "energy"
    for label, frame in (("day_rows", day_rows), ("night_rows", night_rows)):
        if "site_id" not in frame.columns or "ts" not in frame.columns:
            raise RunnerError(f"{label}: missing site_id or ts")
        if frame[["site_id", "ts"]].isna().any().any() or frame.duplicated(subset=["site_id", "ts"]).any():
            raise RunnerError(f"{label}: duplicate key (site_id, ts)")
    try:
        if len(day_rows):
            require_hourly_kst(day_rows["ts"], "day_rows.ts")
        if len(night_rows):
            require_hourly_kst(night_rows["ts"], "night_rows.ts")
    except ValueError as exc:
        raise RunnerError(str(exc)) from exc
    if len(day_rows) and len(night_rows) and _key_index(day_rows).intersection(_key_index(night_rows)).size:
        raise RunnerError("site_frame: day/night key overlap")
    day_actual = (
        pd.to_numeric(day_rows[day_col], errors="coerce").to_numpy(dtype="float64")
        if len(day_rows)
        else np.empty(0, dtype="float64")
    )
    night_actual = (
        pd.to_numeric(night_rows[night_col], errors="coerce").to_numpy(dtype="float64")
        if len(night_rows)
        else np.empty(0, dtype="float64")
    )
    if (len(day_actual) and not np.all(np.isfinite(day_actual))) or (
        len(night_actual) and not np.all(np.isfinite(night_actual))
    ):
        raise RunnerError("actual_kwh: must be finite")
    site_ids = (
        day_rows["site_id"].astype("string").tolist() + night_rows["site_id"].astype("string").tolist()
    )
    stamps = (
        pd.DatetimeIndex(day_rows["ts"]).tolist() + pd.DatetimeIndex(night_rows["ts"]).tolist()
    )
    out = pd.DataFrame(
        {
            "site_id": pd.Series(site_ids, dtype="string"),
            "ts": pd.Series(pd.DatetimeIndex(stamps), dtype=_DT),
            "pred_kwh": pd.Series(
                np.concatenate([preds, np.zeros(len(night_rows), dtype="float64")]),
                dtype="float64",
            ),
            "actual_kwh": pd.Series(
                np.concatenate([day_actual, night_actual]),
                dtype="float64",
            ),
        }
    )
    return out.sort_values(["site_id", "ts"], kind="mergesort").reset_index(drop=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        res = subprocess.run(  # noqa: S603
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if res.returncode != 0:
        return None
    return res.stdout.strip() or None


def _default_panel_path() -> Path:
    return DataPaths().features_dir / "panel.parquet"


def _read(path: Path, columns: list[str], *, all_columns: bool = False) -> pd.DataFrame:
    try:
        frame = pd.read_parquet(path, columns=None if all_columns else columns)
    except (OSError, ValueError) as exc:
        raise RunnerError(f"{path.name}: unreadable ({exc})") from exc
    if frame.columns.has_duplicates or not set(columns) <= set(frame.columns):
        raise RunnerError(f"{path.name}: missing or duplicate columns")
    return frame


def _key_index(frame: pd.DataFrame) -> pd.MultiIndex:
    return pd.MultiIndex.from_arrays(
        [frame["site_id"].astype("string"), pd.DatetimeIndex(frame["ts"]).as_unit("us")],
        names=["site_id", "ts"],
    )


def _check_keys(frame: pd.DataFrame) -> None:
    keys = ["site_id", "ts"] if "site_id" in frame else ["ts"]
    if frame[keys].isna().any().any() or frame.duplicated(keys).any():
        raise RunnerError(f"missing or duplicate keys {keys}")
    require_hourly_kst(frame["ts"], "ts")
    frame["ts"] = frame["ts"].astype(_DT)
    if "op_date" in frame and not frame["op_date"].eq(op_date(frame["ts"])).all():
        raise RunnerError("inconsistent op_date")


def _resolve_assignment(panel: pd.DataFrame, spec: CompetitionSpec | None = None) -> pd.DataFrame:
    comp = spec if spec is not None else CompetitionSpec()
    sites = _read(DataPaths().canonical_dir / "sites.parquet", ["site_id", "current_group", "capacity_kw"])
    assignment = reference_assignment(sites)
    capacity = sites["capacity_kw"].to_numpy(dtype="float64")
    totals = sites.assign(group=sites["current_group"].map({"A": "G1", "B": "G2"})).groupby("group")["capacity_kw"].sum()
    if (
        len(sites) != comp.n_sites
        or not np.isfinite(capacity).all()
        or (capacity <= 0).any()
        or set(totals.index) != set(comp.groups)
        or (totals < comp.min_group_capacity_kw).any()
        or not panel["site_id"].isin(sites["site_id"]).all()
    ):
        raise RunnerError("sites: invalid partition, capacity or unknown panel site")
    metadata = sites.set_index("site_id")
    _require(
        panel["current_group"].eq(panel["site_id"].map(metadata["current_group"])).all()
        and np.array_equal(panel["capacity_kw"].to_numpy(), panel["site_id"].map(metadata["capacity_kw"]).to_numpy()),
        "panel: site metadata changed since feature construction",
    )
    return assignment


def _resolve_smp(panel: pd.DataFrame, spec: CompetitionSpec | None = None) -> pd.DataFrame:
    comp = spec if spec is not None else CompetitionSpec()
    smp = _read(DataPaths().canonical_dir / "smp.parquet", ["ts", "smp", "k", "period"])
    _check_keys(smp)
    if (
        not np.isfinite(smp["smp"].to_numpy(dtype="float64")).all() or (smp["smp"] < 0).any()
        or not smp["k"].eq(k_weight(smp["ts"], comp)).all()
        or not smp["period"].eq(np.where(smp["ts"] <= comp.train_end, "train", "eval")).all()
    ):
        raise RunnerError("smp: invalid price, weight or period")
    return smp.sort_values("ts").reset_index(drop=True)


def _night_frame(
    _panel_train_ts: pd.Series, day_min_elev: float, *, spec: CompetitionSpec | None = None,
    day_rows: pd.DataFrame | None = None,
) -> pd.DataFrame:
    comp = spec if spec is not None else CompetitionSpec()
    generation = _read(DataPaths().canonical_dir / "generation.parquet", ["site_id", "ts", "op_date", "energy"])
    solar = _read(DataPaths().derived_dir / "solar.parquet", ["site_id", "ts", "elev"])
    _check_keys(generation)
    _check_keys(solar)
    obs = generation[generation["energy"].notna()].copy()
    if (obs["ts"] > comp.train_end).any():
        raise RunnerError("generation: evaluation labels forbidden")
    merged = obs.merge(solar, on=["site_id", "ts"], how="left", validate="one_to_one")
    if not np.isfinite(merged[["energy", "elev"]].to_numpy(dtype="float64")).all():
        raise RunnerError("generation: non-finite labels or missing solar")
    if day_rows is not None:
        observed_day = merged.loc[merged["elev"] > day_min_elev]
        _require(
            _key_index(observed_day).sort_values().equals(_key_index(day_rows).sort_values()),
            "panel: training daylight observations mismatch",
        )
    night = merged[merged["elev"] <= day_min_elev].copy()
    out = pd.DataFrame(
        {
            "site_id": night["site_id"].astype("string").reset_index(drop=True),
            "ts": pd.Series(pd.DatetimeIndex(night["ts"]), dtype=_DT).reset_index(drop=True),
            "op_date": pd.Series(pd.DatetimeIndex(night["op_date"]), dtype=_DT).reset_index(drop=True),
            "energy": night["energy"].to_numpy(dtype="float64"),
        }
    )
    return out.sort_values(["site_id", "ts"], kind="mergesort").reset_index(drop=True)


def _load_panel(path: Path, cfg: RunConfig, comp: CompetitionSpec) -> pd.DataFrame:
    panel = _read(path, ["site_id", "ts", "op_date", "period", "energy", "elev", "current_group", "capacity_kw"], all_columns=True)
    _check_keys(panel)
    train = panel["period"].eq("train")
    evaluation = panel["period"].eq("eval")
    if (
        not (train | evaluation).all()
        or not train.any()
        or (panel.loc[train, "ts"] > comp.train_end).any()
        or not panel.loc[evaluation, "ts"].between(comp.eval_start, comp.eval_end).all()
        or not np.isfinite(panel.loc[train, "energy"].to_numpy(dtype="float64")).all()
        or panel.loc[evaluation, "energy"].notna().any()
        or not np.isfinite(panel["elev"].to_numpy(dtype="float64")).all()
        or (panel["elev"] <= cfg.features.day_min_elev_deg).any()
    ):
        raise RunnerError("panel: invalid period, labels or daylight threshold")
    return panel


def _safe_component(value: str) -> None:
    if value in {".", ".."} or re.fullmatch(r"[A-Za-z0-9_.-]+", value) is None:
        raise RunnerError("unsafe artifact path component")


def _normalize_daily(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    work = frame.copy()
    if "op_date" in work.columns and not isinstance(work.index, pd.DatetimeIndex):
        work = work.set_index("op_date")
    try:
        work.index = pd.DatetimeIndex(work.index, dtype=_DT)
    except (ValueError, TypeError) as exc:
        raise RunnerError(f"{label}: bad op_date index") from exc
    return work.sort_index()


def load_daily_runs(
    run_id: str,
    subset: Literal["all", "warm"],
    *,
    bagged: bool = False,
    paths: ArtifactPaths | None = None,
    vcfg: ValidationConfig | None = None,
) -> DailyRuns:
    """저장된 일별 정산을 compare_runs 입력 형태로 읽는다."""
    if subset not in ("all", "warm"):
        raise RunnerError(f"subset: unknown {subset!r}")
    base = paths if paths is not None else ArtifactPaths()
    cfg = vcfg if vcfg is not None else ValidationConfig()
    _safe_component(run_id)
    run_dir = base.runs / run_id
    out: dict[str, list[pd.DataFrame]] = {}
    for fold in cfg.folds:
        _safe_component(fold.name)
        tags = ["bag"] if bagged else [f"s{seed}" for seed in cfg.seeds]
        frames: list[pd.DataFrame] = []
        for tag in tags:
            path = run_dir / f"{fold.name}_{tag}_{subset}.parquet"
            if not path.exists():
                raise RunnerError(f"runs: missing {path.name}")
            daily = _normalize_daily(pd.read_parquet(path), f"{run_id}/{path.name}")
            if (
                daily.empty or not daily.index.is_unique
                or not {"penalty", "revenue"} <= set(daily.columns)
                or not np.isfinite(daily[["penalty", "revenue"]].to_numpy(dtype="float64")).all()
                or (daily["penalty"] < 0).any()
                or not ((daily.index >= fold.test_start) & (daily.index <= fold.test_end)).all()
                or not (daily.index == daily.index.floor("D")).all()
            ):
                raise RunnerError(f"{path.name}: invalid daily settlement")
            frames.append(daily)
        out[fold.name] = frames
    return out


def run_validation(
    cfg: RunConfig,
    *,
    panel_path: Path | None = None,
    vcfg: ValidationConfig | None = None,
    paths: ArtifactPaths | None = None,
    spec: CompetitionSpec | None = None,
) -> ExperimentRecord:
    """폴드별 시드 검증을 수행해 일별 정산을 저장하고 registry에 기록한 레코드를 반환한다."""
    base = paths if paths is not None else ArtifactPaths()
    val = vcfg if vcfg is not None else ValidationConfig()
    comp = spec if spec is not None else CompetitionSpec()
    if tuple(cfg.model.seeds) != tuple(val.seeds):
        raise RunnerError("seeds: model and validation seeds must match")
    run_dir = base.runs / cfg.run_id
    if run_dir.exists():
        raise RunnerError(f"runs: exists {cfg.run_id!r}")
    if any(r.run_id == cfg.run_id for r in read_records(base.registry)):
        raise RunnerError(f"registry: exists {cfg.run_id!r}")
    for fold in val.folds:
        _safe_component(fold.name)
    if cfg.baseline_run_id is not None:
        for subset in ("all", "warm"):
            load_daily_runs(cfg.baseline_run_id, subset, paths=base, vcfg=val)
    panel_file = panel_path if panel_path is not None else _default_panel_path()
    panel_hash = _sha256(panel_file) if panel_file.exists() else None
    panel = _load_panel(panel_file, cfg, comp)
    train_panel = panel[panel["period"] == "train"].reset_index(drop=True)
    assignment = _resolve_assignment(panel, comp)
    smp_full = _resolve_smp(panel, comp)
    night_all = _night_frame(train_panel["ts"], cfg.features.day_min_elev_deg, spec=comp, day_rows=train_panel)
    if not night_all["site_id"].isin(assignment["site_id"]).all():
        raise RunnerError("generation: unknown site")
    features = feature_columns(panel, cfg.features)
    metrics: dict[str, dict[str, float]] = {}
    bag_w: dict[str, dict[str, float]] = {}
    history_rows = pd.concat([train_panel[["site_id", "op_date"]], night_all[["site_id", "op_date"]]], ignore_index=True)
    run_dir.mkdir(parents=True, exist_ok=False)
    for fold in val.folds:
        try:
            train_mask, test_mask = fold_masks(train_panel["op_date"], fold, val)
            assert_causal_split(
                train_panel.loc[train_mask, "op_date"], train_panel.loc[test_mask, "op_date"], fold, val
            )
        except SplitError as exc:
            raise RunnerError(str(exc)) from exc
        history = site_history(history_rows["site_id"], history_rows["op_date"], fold, val)
        warm_ids = set(history.loc[history["is_warm"], "site_id"].astype("string").tolist())
        train_fold = train_panel.loc[train_mask].reset_index(drop=True)
        test_day = train_panel.loc[test_mask].reset_index(drop=True)
        t0 = time.perf_counter()
        models = fit_seed_models(train_fold, features, cfg.model)
        fit_sec = time.perf_counter() - t0
        preds = predict_seed_kwh(models, test_day, features)
        night_mask = (night_all["op_date"] >= fold.test_start) & (night_all["op_date"] <= fold.test_end)
        night_fold = night_all.loc[night_mask].reset_index(drop=True)
        test_ts = pd.DatetimeIndex(test_day["ts"].unique()).union(
            pd.DatetimeIndex(night_fold["ts"].unique())
        )
        smp = smp_full[smp_full["ts"].isin(test_ts)].reset_index(drop=True)
        if smp.empty:
            raise RunnerError(f"fold {fold.name}: missing smp")
        warm_day = test_day["site_id"].astype("string").isin(warm_ids).to_numpy()
        warm_night = (
            night_fold["site_id"].astype("string").isin(warm_ids).to_numpy()
            if len(night_fold)
            else np.empty(0, dtype=bool)
        )
        bag_pred = bag(preds)
        for idx, seed in enumerate(cfg.model.seeds):
            for subset in ("all", "warm"):
                sel_day = np.ones(len(test_day), dtype=bool) if subset == "all" else warm_day
                sel_night = (
                    np.ones(len(night_fold), dtype=bool) if subset == "all" else warm_night
                )
                day_part = test_day.loc[sel_day].reset_index(drop=True)
                night_part = night_fold.loc[sel_night].reset_index(drop=True)
                if len(day_part) + len(night_part) == 0:
                    raise RunnerError(f"fold {fold.name}: empty {subset}")
                frame = site_frame(day_part, preds[idx][sel_day], night_part)
                grouped = aggregate_groups(frame, assignment, spec=comp)
                result = settle(grouped, smp, spec=comp)
                daily = daily_totals(result)
                daily.to_parquet(run_dir / f"{fold.name}_s{seed}_{subset}.parquet")
                if subset == "all":
                    metrics[f"{fold.name}/{seed}"] = {
                        "W_all": float(daily["penalty"].sum()),
                        "W_warm": 0.0,
                        "revenue_all": float(daily["revenue"].sum()),
                        "fit_sec": float(fit_sec),
                    }
                else:
                    metrics[f"{fold.name}/{seed}"]["W_warm"] = float(daily["penalty"].sum())
                if subset == "warm":
                    logger.info(
                        "[EVAL] run=%s fold=%s seed=%s W_all=%r W_warm=%r",
                        cfg.run_id,
                        fold.name,
                        seed,
                        metrics[f"{fold.name}/{seed}"]["W_all"],
                        metrics[f"{fold.name}/{seed}"].get("W_warm", 0.0),
                    )
        for subset in ("all", "warm"):
            sel_day = np.ones(len(test_day), dtype=bool) if subset == "all" else warm_day
            sel_night = np.ones(len(night_fold), dtype=bool) if subset == "all" else warm_night
            frame = site_frame(
                test_day.loc[sel_day].reset_index(drop=True),
                bag_pred[sel_day],
                night_fold.loc[sel_night].reset_index(drop=True),
            )
            grouped = aggregate_groups(frame, assignment, spec=comp)
            result = settle(grouped, smp, spec=comp)
            daily = daily_totals(result)
            daily.to_parquet(run_dir / f"{fold.name}_bag_{subset}.parquet")
            entry = bag_w.setdefault(fold.name, {})
            entry[f"W_{subset}"] = float(daily["penalty"].sum())
        logger.info(
            "[EVAL] run=%s fold=%s bag W_all=%r W_warm=%r",
            cfg.run_id,
            fold.name,
            bag_w[fold.name]["W_all"],
            bag_w[fold.name]["W_warm"],
        )
        del models, preds, bag_pred, train_fold, test_day
        gc.collect()
    comparison: dict[str, object] = {"bag": bag_w}
    if cfg.baseline_run_id is not None:
        cand_all = load_daily_runs(cfg.run_id, "all", paths=base, vcfg=val)
        base_all = load_daily_runs(cfg.baseline_run_id, "all", paths=base, vcfg=val)
        cand_warm = load_daily_runs(cfg.run_id, "warm", paths=base, vcfg=val)
        base_warm = load_daily_runs(cfg.baseline_run_id, "warm", paths=base, vcfg=val)
        comparison["vs_baseline"] = {
            "all": comparison_to_dict(compare_runs(cand_all, base_all, val)),
            "warm": comparison_to_dict(compare_runs(cand_warm, base_warm, val)),
        }
    config = dict(cfg.model_dump(mode="json"))
    if _sha256(panel_file) != panel_hash:
        raise RunnerError("panel: changed during validation")
    config["panel_path"] = str(panel_file.resolve())
    config["panel_sha256"] = panel_hash
    config["validation"] = val.model_dump(mode="json")
    config["competition"] = comp.model_dump(mode="json")
    note = f"{cfg.note} | {_BAG_NOTE}" if cfg.note else _BAG_NOTE
    record = ExperimentRecord(
        run_id=cfg.run_id,
        created_at=datetime.now().astimezone(),
        hypothesis=cfg.hypothesis,
        config=config,
        git_commit=_git_commit(),
        seeds=tuple(val.seeds),
        folds=tuple(f.name for f in val.folds),
        metrics=metrics,
        comparison=comparison,
        baseline_run_id=cfg.baseline_run_id,
        note=note,
    )
    append_record(record, base.registry)
    return record


def forecast_eval(
    cfg: RunConfig,
    *,
    panel_path: Path | None = None,
    spec: CompetitionSpec | None = None,
) -> pd.DataFrame:
    """학습 기간 전체로 학습한 seed-bag 모델의 평가 기간 (group, ts, forecast_kwh) 예측을 반환한다."""
    comp = spec if spec is not None else CompetitionSpec()
    panel_file = panel_path if panel_path is not None else _default_panel_path()
    panel = _load_panel(panel_file, cfg, comp)
    train = panel[panel["period"] == "train"].reset_index(drop=True)
    eval_day = panel[panel["period"] == "eval"].reset_index(drop=True)
    features = feature_columns(panel, cfg.features)
    assignment = _resolve_assignment(panel, comp)
    smp = _resolve_smp(panel, comp)
    eval_ts = pd.DatetimeIndex(smp.loc[smp["period"] == "eval", "ts"]).sort_values()
    if not eval_ts.equals(pd.date_range(comp.eval_start, comp.eval_end, freq="h").as_unit("us")):
        raise RunnerError("smp: eval window mismatch")
    grid = _read(DataPaths().derived_dir / "grid.parquet", ["site_id", "ts", "period"])
    grid = grid[grid["period"] == "eval"].reset_index(drop=True)
    _check_keys(grid)
    expected = pd.MultiIndex.from_product([assignment["site_id"], eval_ts], names=["site_id", "ts"])
    if not _key_index(grid).sort_values().equals(expected.sort_values()):
        raise RunnerError("grid: incomplete evaluation site-hours")
    solar = _read(DataPaths().derived_dir / "solar.parquet", ["site_id", "ts", "elev"])
    _check_keys(solar)
    grid = grid.merge(solar, on=["site_id", "ts"], how="left", validate="one_to_one")
    if not np.isfinite(grid["elev"].to_numpy(dtype="float64")).all():
        raise RunnerError("grid: missing solar")
    expected_day = _key_index(grid.loc[grid["elev"] > cfg.features.day_min_elev_deg])
    if not _key_index(eval_day).sort_values().equals(expected_day.sort_values()):
        raise RunnerError("panel: missing or unexpected evaluation daylight rows")
    models = fit_seed_models(train, features, cfg.model)
    preds = predict_seed_kwh(models, eval_day, features)
    mean = bag(preds)
    if mean.shape != (len(eval_day),) or not np.isfinite(mean).all() or (mean < 0).any():
        raise RunnerError("forecast: invalid seed-bag predictions")
    del models, preds
    gc.collect()
    day = eval_day[["site_id", "ts"]].assign(forecast_kwh=mean)
    full = grid[["site_id", "ts"]].merge(day, on=["site_id", "ts"], how="left", validate="one_to_one")
    full["forecast_kwh"] = full["forecast_kwh"].fillna(0.0)
    full = full.merge(assignment, on="site_id", how="left", validate="many_to_one")
    grouped = (
        full.groupby(["group", "ts"], observed=True, sort=False)["forecast_kwh"]
        .sum()
        .reset_index()
    )
    grouped["group"] = grouped["group"].astype("string")
    grouped["ts"] = pd.Series(pd.DatetimeIndex(grouped["ts"]), dtype=_DT)
    grouped["forecast_kwh"] = grouped["forecast_kwh"].to_numpy(dtype="float64")
    out = grouped[["group", "ts", "forecast_kwh"]].sort_values(
        ["group", "ts"], kind="mergesort"
    ).reset_index(drop=True)
    if not np.isfinite(out["forecast_kwh"].to_numpy()).all():
        raise RunnerError("forecast: group sum overflow")
    for group in comp.groups:
        got = pd.DatetimeIndex(out.loc[out["group"] == group, "ts"]).sort_values()
        _require(got.equals(eval_ts), "forecast: ts mismatch with smp eval")
    site_total = float(mean.sum())
    group_total = float(out["forecast_kwh"].sum())
    _require(bool(np.isfinite([group_total, site_total]).all()), "forecast: total overflow")
    _require(bool(np.isclose(group_total, site_total, rtol=1e-9, atol=1e-6)), "forecast: group total mismatch")
    return out
