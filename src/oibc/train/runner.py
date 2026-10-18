"""폴드 검증·평가 예측 실행기."""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import re
import subprocess
import time
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
import pandas as pd

from src.oibc.config import (
    REFERENCE_COMP_ID,
    ArtifactPaths,
    CalibrationConfig,
    CompetitionSpec,
    DataPaths,
    RunConfig,
    ValidationConfig,
)
from src.oibc.cv.splits import SplitError, assert_causal_split, fold_masks, site_history
from src.oibc.data.schemas import _DT
from src.oibc.evaluate.compare import CompareError, DailyRuns, block_bootstrap_ci, compare_runs, comparison_to_dict
from src.oibc.evaluate.registry import ExperimentRecord, append_record, read_records
from src.oibc.evaluate.settlement import SettlementError, aggregate_groups, daily_totals, settle
from src.oibc.features.panel import feature_columns
from src.oibc.models.calibrate import (
    CalibrationError,
    CloudCalibration,
    apply_calibration,
    cloud_index,
    fit_cloud_calibration,
    scale_factors,
)
from src.oibc.models.families import FamilyError, blend_families, fit_predict_catboost, fit_predict_mlp
from src.oibc.models.gbm import bag, fit_seed_models, predict_seed_kwh
from src.oibc.models.regional import blend_seed_preds, fit_predict_regional
from src.oibc.timeaxis import k_weight, op_date, require_hourly_kst

logger = logging.getLogger(__name__)

_BAG_NOTE = "운영 예측은 seed-bag: |F−A| 볼록성으로 평균 예측의 W ≤ 단일 W 평균(Jensen), 실측 −0.4% (probe 2026-10-16)"  # noqa: RUF001


class RunnerError(ValueError):
    """실행기 계약 위반."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RunnerError(message)


def _site_geo(panel: pd.DataFrame) -> pd.DataFrame:
    """패널 전체에서 site별 정적 표를 반환한다."""
    for col in ("site_id", "lat", "lon", "capacity_kw"):
        if col not in panel.columns:
            raise RunnerError(f"panel: missing {col}")
    if panel.empty:
        raise RunnerError("panel: empty")
    site_ids = panel["site_id"].astype("string")
    if site_ids.isna().any() or bool((site_ids == "").any()):
        raise RunnerError("panel: missing site_id")
    lat = pd.to_numeric(panel["lat"], errors="coerce").to_numpy(dtype="float64")
    lon = pd.to_numeric(panel["lon"], errors="coerce").to_numpy(dtype="float64")
    cap = pd.to_numeric(panel["capacity_kw"], errors="coerce").to_numpy(dtype="float64")
    if not np.all(np.isfinite(lat)) or not np.all(np.isfinite(lon)):
        raise RunnerError("panel: site lat/lon must be finite")
    if not np.all(np.isfinite(cap)) or bool((cap <= 0).any()):
        raise RunnerError("panel: site capacity_kw must be finite and > 0")
    work = pd.DataFrame(
        {
            "site_id": site_ids.reset_index(drop=True),
            "lat": lat,
            "lon": lon,
            "capacity_kw": cap,
        }
    )
    nunique = work.groupby("site_id", observed=True)[["lat", "lon", "capacity_kw"]].nunique(dropna=False)
    if bool((nunique != 1).any().any()):
        raise RunnerError("panel: inconsistent site_geo within site")
    out = work.drop_duplicates(subset=["site_id"]).sort_values("site_id", kind="mergesort").reset_index(drop=True)
    return pd.DataFrame(
        {
            "site_id": out["site_id"].astype("string"),
            "lat": out["lat"].to_numpy(dtype="float64"),
            "lon": out["lon"].to_numpy(dtype="float64"),
            "capacity_kw": out["capacity_kw"].to_numpy(dtype="float64"),
        }
    )


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
    site_col = pd.concat(
        [day_rows["site_id"].astype("string"), night_rows["site_id"].astype("string")],
        ignore_index=True,
    ).astype("string")
    ts_col = pd.concat(
        [rows["ts"].astype(_DT) if len(rows) else pd.Series(dtype=_DT) for rows in (day_rows, night_rows)],
        ignore_index=True,
    ).astype(_DT)
    out = pd.DataFrame(
        {
            "site_id": pd.Series(site_col, dtype="string"),
            "ts": pd.Series(ts_col, dtype=_DT),
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


def _settle_subset(
    test_day: pd.DataFrame,
    night_fold: pd.DataFrame,
    pred: np.ndarray,
    sel_day: np.ndarray,
    sel_night: np.ndarray,
    assignment: pd.DataFrame,
    smp: pd.DataFrame,
    comp: CompetitionSpec,
    *,
    fold_name: str,
    subset: str,
) -> pd.DataFrame:
    """선택 마스크를 적용해 일별 정산을 계산한다."""
    day_part = test_day.loc[sel_day].reset_index(drop=True)
    night_part = night_fold.loc[sel_night].reset_index(drop=True)
    if len(day_part) + len(night_part) == 0:
        raise RunnerError(f"fold {fold_name}: empty {subset}")
    frame = site_frame(day_part, np.asarray(pred, dtype="float64")[sel_day], night_part)
    grouped = aggregate_groups(frame, assignment, spec=comp)
    return daily_totals(settle(grouped, smp, spec=comp))


def load_oof(run_id: str, fold: str, *, paths: ArtifactPaths | None = None) -> pd.DataFrame:
    """저장된 폴드 OOF를 스키마 검증 후 반환한다. 파일 누락·스키마·키·값 위반 시 RunnerError."""
    _safe_component(run_id)
    _safe_component(fold)
    base = paths if paths is not None else ArtifactPaths()
    path = base.runs / run_id / f"{fold}_oof.parquet"
    if not path.exists():
        raise RunnerError(f"oof: missing {path.name} (run={run_id} fold={fold}); re-run with --save-oof")
    try:
        frame = pd.read_parquet(path)
    except (OSError, ValueError) as exc:
        raise RunnerError(f"oof: unreadable ({exc})") from exc
    required = ["site_id", "ts", "op_date", "actual_kwh", "pred_bag", "is_warm"]
    if frame.columns.has_duplicates or not set(required) <= set(frame.columns):
        raise RunnerError("oof: missing columns")
    seed_cols = [c for c in frame.columns if re.fullmatch(r"pred_s-?\d+", str(c))]
    if not seed_cols:
        raise RunnerError("oof: missing pred_s columns")
    numeric_cols = ["actual_kwh", *seed_cols, "pred_bag"]
    expected = ["site_id", "ts", "op_date", "actual_kwh", *seed_cols, "pred_bag", "is_warm"]
    if (
        frame.columns.tolist() != expected
        or not isinstance(frame["site_id"].dtype, pd.StringDtype)
        or any(str(frame[c].dtype) != _DT for c in ("ts", "op_date"))
        or any(str(frame[c].dtype) != "float64" for c in numeric_cols)
        or str(frame["is_warm"].dtype) != "bool"
    ):
        raise RunnerError("oof: invalid schema")
    if frame[["site_id", "ts"]].isna().any().any() or frame.duplicated(subset=["site_id", "ts"]).any():
        raise RunnerError("oof: duplicate key (site_id, ts)")
    try:
        require_hourly_kst(frame["ts"], "ts")
    except ValueError as exc:
        raise RunnerError(str(exc)) from exc
    if not frame["op_date"].eq(op_date(frame["ts"])).all():
        raise RunnerError("oof: inconsistent op_date")
    actual = frame["actual_kwh"].to_numpy()
    if not np.all(np.isfinite(actual)):
        raise RunnerError("oof: actual_kwh must be finite")
    for col in [*seed_cols, "pred_bag"]:
        vals = frame[col].to_numpy()
        if not np.all(np.isfinite(vals)) or bool((vals < 0).any()):
            raise RunnerError(f"oof: {col} must be finite and >= 0")
    return frame.sort_values(["site_id", "ts"], kind="mergesort").reset_index(drop=True)


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


def _git_tree_dirty() -> bool | None:
    try:
        res = subprocess.run(  # noqa: S603
            ["git", "status", "--porcelain", "--untracked-files=no"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if res.returncode != 0:
        return None
    return bool(res.stdout.strip())


def _normalize_diff_value(value: object) -> object:
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if isinstance(value, tuple):
        return list(value)
    return value


def resolve_composition(
    composition_id: str, sites: pd.DataFrame,
    *, paths: ArtifactPaths | None = None, spec: CompetitionSpec | None = None,
) -> tuple[pd.DataFrame, str]:
    """composition_id에 해당하는 검증된 (site_id, group) 배정과 assignment_sha256을 반환한다."""
    from src.oibc.compose.constraints import CompositionError, validate_assignment
    from src.oibc.compose.store import assignment_sha256, load_composition

    comp = spec if spec is not None else CompetitionSpec()
    try:
        if composition_id == REFERENCE_COMP_ID:
            assignment = reference_assignment(sites)
            validate_assignment(assignment, sites, spec=comp)
        else:
            assignment, _ = load_composition(composition_id, sites, paths=paths, spec=comp)
        return assignment, assignment_sha256(assignment)
    except CompositionError as exc:
        raise RunnerError(str(exc)) from exc


def _config_diff_keys(baseline: ExperimentRecord, cfg: RunConfig) -> list[str]:
    current = cfg.model_dump(mode="json")
    base_config = baseline.config if isinstance(baseline.config, dict) else {}
    out: list[str] = []
    base_comp = base_config.get("composition_id", REFERENCE_COMP_ID)
    if current.get("composition_id", REFERENCE_COMP_ID) != base_comp:
        out.append("composition_id")
    for section in ("features", "model"):
        cur_section = current.get(section)
        base_section = base_config.get(section)
        if not isinstance(cur_section, dict) or not isinstance(base_section, dict):
            raise RunnerError(f"baseline: missing or invalid {section} config")
        out.extend(
            f"{section}.{key}"
            for key in sorted(set(cur_section) | set(base_section))
            if (
                sorted(cur_section[key]) != sorted(base_section[key])
                if section == "features" and key == "groups" and key in base_section
                else _normalize_diff_value(cur_section.get(key)) != _normalize_diff_value(base_section.get(key))
            )
        )
    return sorted(out)


def _check_comparable(
    baseline: ExperimentRecord,
    cfg: RunConfig,
    panel_hash: str | None,
    val: ValidationConfig,
    comp: CompetitionSpec,
) -> list[str]:
    if tuple(baseline.seeds) != tuple(val.seeds):
        raise RunnerError("baseline: seeds mismatch")
    if tuple(baseline.folds) != tuple(f.name for f in val.folds):
        raise RunnerError("baseline: folds mismatch")
    base_config = baseline.config if isinstance(baseline.config, dict) else {}
    if base_config.get("validation") != val.model_dump(mode="json"):
        raise RunnerError("baseline: validation config mismatch")
    if base_config.get("competition") != comp.model_dump(mode="json"):
        raise RunnerError("baseline: competition spec mismatch")
    if panel_hash is None or base_config.get("panel_sha256") != panel_hash:
        raise RunnerError("baseline: panel hash mismatch")
    current_commit = _git_commit()
    if (current_commit is None or current_commit != baseline.git_commit) and not cfg.allow_commit_drift:
        raise RunnerError("baseline: git commit drift")
    dirty = _git_tree_dirty()
    if (dirty is not False or base_config.get("git_dirty") is not False) and not cfg.allow_commit_drift:
        raise RunnerError("baseline: dirty work tree")
    return _config_diff_keys(baseline, cfg)


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


def _resolve_assignment(
    panel: pd.DataFrame,
    composition_id: str = REFERENCE_COMP_ID,
    spec: CompetitionSpec | None = None,
    paths: ArtifactPaths | None = None,
) -> tuple[pd.DataFrame, str]:
    comp = spec if spec is not None else CompetitionSpec()
    sites = _read(DataPaths().canonical_dir / "sites.parquet", ["site_id", "current_group", "capacity_kw"])
    assignment, sha256 = resolve_composition(composition_id, sites, paths=paths, spec=comp)
    if not panel["site_id"].isin(sites["site_id"]).all():
        raise RunnerError("sites: invalid partition, capacity or unknown panel site")
    metadata = sites.set_index("site_id")
    _require(
        panel["current_group"].eq(panel["site_id"].map(metadata["current_group"])).all()
        and np.array_equal(panel["capacity_kw"].to_numpy(), panel["site_id"].map(metadata["capacity_kw"]).to_numpy()),
        "panel: site metadata changed since feature construction",
    )
    return assignment, sha256


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
    save_oof: bool = False,
) -> ExperimentRecord:
    """폴드별 시드 검증을 수행해 일별 정산(선택 시 발전소 단위 OOF)을 저장하고 registry에 기록한 레코드를 반환한다."""
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
    config_diff: list[str] = []
    if cfg.baseline_run_id is not None:
        matches = [r for r in read_records(base.registry) if r.run_id == cfg.baseline_run_id]
        if not matches:
            raise RunnerError(f"registry: unknown baseline {cfg.baseline_run_id!r}")
        config_diff = _check_comparable(matches[0], cfg, panel_hash, val, comp)
    panel = _load_panel(panel_file, cfg, comp)
    families_active = cfg.model.families.cat_weight > 0 or cfg.model.families.mlp_weight > 0
    site_geo = _site_geo(panel) if cfg.model.regional.weight > 0 or families_active else None
    features = feature_columns(panel, cfg.features)
    keep = list(dict.fromkeys(["site_id", "ts", "op_date", "energy", "capacity_kw", "smp", "k", *features]))
    train_panel = panel.loc[panel["period"] == "train", keep].reset_index(drop=True)
    assignment, comp_sha = _resolve_assignment(panel, cfg.composition_id, comp, base)
    # _night_frame이 generation·solar 전체를 읽으므로 그 전에 panel을 해제해 peak RSS를 낮춘다.
    del panel
    gc.collect()
    sites = _read(DataPaths().canonical_dir / "sites.parquet", ["site_id", "capacity_kw"])
    site_cap = sites.set_index("site_id")["capacity_kw"]
    caps = {
        g: float(site_cap[assignment.loc[assignment["group"] == g, "site_id"]].sum())
        for g in comp.groups
    }
    logger.info(
        "[VPP] run=%s composition_id=%s sha256=%s caps=%s",
        cfg.run_id, cfg.composition_id, comp_sha, caps,
    )
    smp_full = _resolve_smp(train_panel, comp)
    night_all = _night_frame(train_panel["ts"], cfg.features.day_min_elev_deg, spec=comp, day_rows=train_panel)
    if not night_all["site_id"].isin(assignment["site_id"]).all():
        raise RunnerError("generation: unknown site")
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
        preds = predict_seed_kwh(models, test_day, features, num_threads=cfg.model.n_jobs)
        del models
        if cfg.model.regional.weight > 0:
            assert site_geo is not None
            t1 = time.perf_counter()
            reg = fit_predict_regional(train_fold, test_day, features, site_geo, cfg.model)
            preds = blend_seed_preds(preds, reg, cfg.model.regional.weight)
            regional_sec = time.perf_counter() - t1
            del reg
        if families_active:
            assert site_geo is not None
            t2 = time.perf_counter()
            try:
                cat = (
                    fit_predict_catboost(train_fold, test_day, features, site_geo, cfg.model)
                    if cfg.model.families.cat_weight > 0
                    else None
                )
                mlp = (
                    fit_predict_mlp(train_fold, test_day, features, site_geo, cfg.model)
                    if cfg.model.families.mlp_weight > 0
                    else None
                )
                preds = blend_families(preds, cat, mlp, cfg.model.families)
            except FamilyError as exc:
                raise RunnerError(str(exc)) from exc
            family_sec = time.perf_counter() - t2
            del cat, mlp
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
                daily = _settle_subset(
                    test_day, night_fold, preds[idx], sel_day, sel_night,
                    assignment, smp, comp, fold_name=fold.name, subset=subset,
                )
                daily.to_parquet(run_dir / f"{fold.name}_s{seed}_{subset}.parquet")
                if subset == "all":
                    metrics[f"{fold.name}/{seed}"] = {
                        "W_all": float(daily["penalty"].sum()),
                        "W_warm": 0.0,
                        "revenue_all": float(daily["revenue"].sum()),
                        "fit_sec": float(fit_sec),
                    }
                    if cfg.model.regional.weight > 0:
                        metrics[f"{fold.name}/{seed}"]["regional_sec"] = float(regional_sec)
                    if families_active:
                        metrics[f"{fold.name}/{seed}"]["family_sec"] = float(family_sec)
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
            daily = _settle_subset(
                test_day, night_fold, bag_pred, sel_day, sel_night,
                assignment, smp, comp, fold_name=fold.name, subset=subset,
            )
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
        if save_oof:
            day_col = _actual_col(test_day) if len(test_day) else "energy"
            night_col = _actual_col(night_fold) if len(night_fold) else "energy"
            day_actual = (
                pd.to_numeric(test_day[day_col], errors="coerce").to_numpy(dtype="float64")
                if len(test_day)
                else np.empty(0, dtype="float64")
            )
            night_actual = (
                pd.to_numeric(night_fold[night_col], errors="coerce").to_numpy(dtype="float64")
                if len(night_fold)
                else np.empty(0, dtype="float64")
            )
            seed_cols = [f"pred_s{s}" for s in cfg.model.seeds]
            oof = pd.DataFrame(
                {
                    "site_id": pd.Series(
                        pd.concat(
                            [
                                test_day["site_id"].astype("string"),
                                night_fold["site_id"].astype("string"),
                            ],
                            ignore_index=True,
                        ).astype("string"),
                        dtype="string",
                    ),
                    "ts": pd.Series(
                        pd.concat(
                            [test_day["ts"].astype(_DT), night_fold["ts"].astype(_DT)],
                            ignore_index=True,
                        ).astype(_DT),
                        dtype=_DT,
                    ),
                    "op_date": pd.Series(
                        pd.concat(
                            [test_day["op_date"].astype(_DT), night_fold["op_date"].astype(_DT)],
                            ignore_index=True,
                        ).astype(_DT),
                        dtype=_DT,
                    ),
                    "actual_kwh": pd.Series(
                        np.concatenate([day_actual, night_actual]), dtype="float64"
                    ),
                    **{
                        name: pd.Series(
                            np.concatenate(
                                [np.asarray(preds[idx], dtype="float64"),
                                 np.zeros(len(night_fold), dtype="float64")]
                            ),
                            dtype="float64",
                        )
                        for idx, name in enumerate(seed_cols)
                    },
                    "pred_bag": pd.Series(
                        np.concatenate(
                            [np.asarray(bag_pred, dtype="float64"),
                             np.zeros(len(night_fold), dtype="float64")]
                        ),
                        dtype="float64",
                    ),
                    "is_warm": pd.Series(
                        np.concatenate([warm_day, warm_night]),
                        dtype="bool",
                    ),
                }
            )
            oof = oof[["site_id", "ts", "op_date", "actual_kwh", *seed_cols, "pred_bag", "is_warm"]]
            oof = oof.sort_values(["site_id", "ts"], kind="mergesort").reset_index(drop=True)
            oof.to_parquet(run_dir / f"{fold.name}_oof.parquet", index=False)
            logger.info(
                "[DATA] run=%s fold=%s oof rows=%d sites=%d",
                cfg.run_id, fold.name, len(oof), int(oof["site_id"].nunique()),
            )
            del oof
        del preds, bag_pred, train_fold, test_day
        gc.collect()
    comparison: dict[str, object] = {"bag": bag_w, "config_diff": config_diff}
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
    config["composition_sha256"] = comp_sha
    config["oof_saved"] = bool(save_oof)
    if _sha256(panel_file) != panel_hash:
        raise RunnerError("panel: changed during validation")
    config["panel_path"] = str(panel_file.resolve())
    config["panel_sha256"] = panel_hash
    config["validation"] = val.model_dump(mode="json")
    config["competition"] = comp.model_dump(mode="json")
    dirty_state = _git_tree_dirty()
    config["git_dirty"] = bool(dirty_state) if dirty_state is not None else True
    note = f"{cfg.note} | {_BAG_NOTE}" if cfg.note else _BAG_NOTE
    if cfg.allow_commit_drift:
        note = f"{note} | commit_drift_allowed"
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
    paths: ArtifactPaths | None = None,
    calibration: CloudCalibration | None = None,
    calibration_cfg: CalibrationConfig | None = None,
) -> pd.DataFrame:
    """학습 기간 전체로 학습한 seed-bag 모델의 평가 기간 (group, ts, forecast_kwh) 예측을 반환한다."""
    if (calibration is None) != (calibration_cfg is None):
        raise RunnerError("forecast: calibration and calibration_cfg must be paired")
    comp = spec if spec is not None else CompetitionSpec()
    panel_file = panel_path if panel_path is not None else _default_panel_path()
    panel = _load_panel(panel_file, cfg, comp)
    families_active = cfg.model.families.cat_weight > 0 or cfg.model.families.mlp_weight > 0
    site_geo = _site_geo(panel) if cfg.model.regional.weight > 0 or families_active else None
    features = feature_columns(panel, cfg.features)
    train_keep = list(dict.fromkeys([*features, "site_id", "ts", "energy", "capacity_kw", "smp", "k"]))
    eval_keep = list(dict.fromkeys([*features, "site_id", "ts", "capacity_kw"]))
    if calibration is not None:
        assert calibration_cfg is not None
        eval_keep = list(dict.fromkeys([*eval_keep, *calibration_cfg.cloud_vars]))
        missing = [c for c in eval_keep if c not in panel.columns]
        if missing:
            raise RunnerError(f"forecast: missing calibration columns {missing[:5]}")
    train = panel.loc[panel["period"] == "train", train_keep].reset_index(drop=True)
    eval_day = panel.loc[panel["period"] == "eval", eval_keep].reset_index(drop=True)
    assignment, _ = _resolve_assignment(panel, cfg.composition_id, comp, paths)
    smp = _resolve_smp(panel, comp)
    del panel
    gc.collect()
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
    del solar
    if not np.isfinite(grid["elev"].to_numpy(dtype="float64")).all():
        raise RunnerError("grid: missing solar")
    expected_day = _key_index(grid.loc[grid["elev"] > cfg.features.day_min_elev_deg])
    if not _key_index(eval_day).sort_values().equals(expected_day.sort_values()):
        raise RunnerError("panel: missing or unexpected evaluation daylight rows")
    grid = grid[["site_id", "ts"]]
    del expected_day
    gc.collect()
    models = fit_seed_models(train, features, cfg.model)
    preds = predict_seed_kwh(models, eval_day, features, num_threads=cfg.model.n_jobs)
    del models
    if cfg.model.regional.weight > 0:
        assert site_geo is not None
        preds = blend_seed_preds(
            preds,
            fit_predict_regional(train, eval_day, features, site_geo, cfg.model),
            cfg.model.regional.weight,
        )
    if families_active:
        assert site_geo is not None
        try:
            cat = (
                fit_predict_catboost(train, eval_day, features, site_geo, cfg.model)
                if cfg.model.families.cat_weight > 0
                else None
            )
            mlp = (
                fit_predict_mlp(train, eval_day, features, site_geo, cfg.model)
                if cfg.model.families.mlp_weight > 0
                else None
            )
            preds = blend_families(preds, cat, mlp, cfg.model.families)
        except FamilyError as exc:
            raise RunnerError(str(exc)) from exc
        del cat, mlp
    mean = bag(preds)
    if mean.shape != (len(eval_day),) or not np.isfinite(mean).all() or (mean < 0).any():
        raise RunnerError("forecast: invalid seed-bag predictions")
    if calibration is not None:
        assert calibration_cfg is not None
        try:
            index = cloud_index(eval_day, calibration_cfg)
        except CalibrationError as exc:
            raise RunnerError(f"forecast: {exc}") from exc
        lookup = index.reindex(pd.DatetimeIndex(eval_day["ts"].unique()))
        if bool(lookup.isna().any()):
            raise RunnerError("forecast: missing cloud index")
        try:
            mean = apply_calibration(mean, eval_day["ts"], index, calibration)
        except CalibrationError as exc:
            raise RunnerError(f"forecast: {exc}") from exc
    del preds
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


def _summary_stats(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype="float64")
    scale = float(arr.max()) or 1.0
    return {
        "mean": float(np.mean(arr / scale) * scale),
        "sd": float(np.std(arr / scale) * scale),
        "p05": float(np.percentile(arr, 5)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
    }


def netting_report(
    run_id: str,
    composition_ids: Sequence[str],
    *,
    n_random: int = 200,
    seed: int = 0,
    pred_col: str = "pred_bag",
    paths: ArtifactPaths | None = None,
    spec: CompetitionSpec | None = None,
) -> dict[str, object]:
    """run의 폴드 OOF로 지정 구성·무작위·용량 균형 배정의 벌점 분해를 폴드별로 계산해 반환한다. OOF 누락·구성 위반 시 RunnerError."""
    from src.oibc.compose.baselines import capacity_balanced_assignment, random_assignment
    from src.oibc.config import CompetitionSpec as RecordedCompetitionSpec
    from src.oibc.evaluate.netting import NettingError, NettingEvaluator

    base = paths if paths is not None else ArtifactPaths()
    _require(
        isinstance(n_random, int) and not isinstance(n_random, bool) and n_random >= 1,
        "n_random: must be an integer >= 1",
    )
    _require(isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0, "seed: must be a non-negative integer")
    ids = list(composition_ids)
    _require(bool(ids) and len(set(ids)) == len(ids), "composition_ids: must be non-empty and distinct")
    _safe_component(run_id)
    for comp_id in ids:
        _safe_component(comp_id)
    records = read_records(base.registry)
    match = next((r for r in records if r.run_id == run_id), None)
    _require(match is not None, f"registry: unknown run_id {run_id!r}")
    assert match is not None
    try:
        spec_data = dict(match.config["competition"])
        for key in ("train_end", "eval_start", "eval_end"):
            spec_data[key] = pd.Timestamp(spec_data[key]).tz_convert("Asia/Seoul")
        recorded = RecordedCompetitionSpec.model_validate(spec_data)
    except (KeyError, TypeError, ValueError) as exc:
        raise RunnerError("registry: invalid competition spec") from exc
    comp = spec if spec is not None else recorded
    _require(comp == recorded, "registry: competition spec mismatch")
    folds = list(match.folds)
    _require(bool(folds), "registry: empty folds")
    sites = _read(DataPaths().canonical_dir / "sites.parquet", ["site_id", "current_group", "capacity_kw"])
    smp = _resolve_smp(pd.DataFrame(), comp)
    resolved: dict[str, pd.DataFrame] = {}
    hashes: dict[str, str] = {}
    for comp_id in ids:
        assignment, digest = resolve_composition(comp_id, sites, paths=base, spec=comp)
        resolved[comp_id] = assignment
        hashes[comp_id] = digest
    rng = np.random.default_rng(int(seed))
    try:
        random_sets = [random_assignment(sites, rng, spec=comp) for _ in range(int(n_random))]
        balanced_sets = [capacity_balanced_assignment(sites, rng, spec=comp) for _ in range(int(n_random))]
    except ValueError as exc:
        raise RunnerError(str(exc)) from exc
    fold_out: dict[str, object] = {}
    for fold in folds:
        oof = load_oof(run_id, fold, paths=base)
        _require(not oof.empty and bool((oof["ts"] <= comp.train_end).all()), "oof: empty or evaluation labels forbidden")
        try:
            evaluator = NettingEvaluator(oof, smp, pred_col=pred_col, spec=comp)
        except NettingError as exc:
            raise RunnerError(str(exc)) from exc
        comp_results: dict[str, object] = {}
        comp_penalties: dict[str, float] = {}
        for comp_id in ids:
            try:
                res = evaluator.evaluate(resolved[comp_id])
            except NettingError as exc:
                raise RunnerError(f"fold {fold}: {exc}") from exc
            share = float(res.netting_loss / res.penalty) if res.penalty > 0 else 0.0
            comp_results[comp_id] = {
                "penalty": float(res.penalty),
                "netting_loss": float(res.netting_loss),
                "netting_share": float(share),
            }
            comp_penalties[comp_id] = float(res.penalty)
        try:
            random_pen = evaluator.penalties(random_sets)
            balanced_pen = evaluator.penalties(balanced_sets)
        except NettingError as exc:
            raise RunnerError(f"fold {fold}: {exc}") from exc
        percentiles = {
            comp_id: float(np.mean(random_pen < comp_penalties[comp_id])) for comp_id in ids
        }
        fold_out[fold] = {
            "pool_penalty": float(evaluator.pool_penalty),
            "compositions": comp_results,
            "random": _summary_stats(random_pen),
            "capacity_balanced": _summary_stats(balanced_pen),
            "composition_percentile_vs_random": percentiles,
        }
        logger.info(
            "[VPP] netting run=%s fold=%s pool=%r comps=%s random_mean=%r",
            run_id,
            fold,
            float(evaluator.pool_penalty),
            {k: float(comp_penalties[k]) for k in ids},
            float(np.mean(random_pen)),
        )
    return {
        "run_id": run_id,
        "pred_col": pred_col,
        "n_random": int(n_random),
        "seed": int(seed),
        "compositions": dict(hashes),
        "folds": fold_out,
    }


class _CrossEntry(TypedDict):
    """교차 적합 순서쌍 한 건의 계수·ΔW·CI 기록."""

    fit: str
    apply: str
    causal: bool
    a: float
    b: float
    at_boundary: bool
    W_base: float
    W_cal: float
    rel_delta: float
    ci_low: float
    ci_high: float
    quarter_rel: dict[str, float]


def _quarter_rel_calibration(base_hourly: pd.DataFrame, cal_hourly: pd.DataFrame) -> dict[str, float]:
    """운영일 분기(YYYYQn)별 상대 ΔW를 반환한다. 기준이 0인 분기는 제외한다."""
    if not base_hourly["ts"].equals(cal_hourly["ts"]) or not base_hourly["group"].equals(cal_hourly["group"]):
        raise RunnerError("calibration: hourly alignment mismatch")
    opd = pd.DatetimeIndex(base_hourly["op_date"]).tz_convert("Asia/Seoul")
    labels = np.array([f"{y}Q{q}" for y, q in zip(opd.year.tolist(), opd.quarter.tolist(), strict=True)])
    base_pen = base_hourly["penalty"].to_numpy(dtype="float64")
    cal_pen = cal_hourly["penalty"].to_numpy(dtype="float64")
    out: dict[str, float] = {}
    for label in sorted(set(labels.tolist())):
        mask = labels == label
        denom = float(base_pen[mask].sum())
        if denom == 0.0:
            continue
        value = float(cal_pen[mask].sum()) / denom - 1.0
        if not np.isfinite(value):
            raise RunnerError("calibration: non-finite quarter delta")
        out[label] = value
    return out


def _rel_delta(w_cal: float, w_base: float) -> float:
    if w_base > 0.0:
        value = w_cal / w_base - 1.0
    elif w_cal == 0.0:
        return 0.0
    else:
        raise RunnerError("calibration: non-positive baseline W")
    if not np.isfinite(value):
        raise RunnerError("calibration: non-finite rel_delta")
    return float(value)


def _pair_ci(
    cal_daily: np.ndarray, base_daily: np.ndarray, w_cal: float, w_base: float, val: ValidationConfig,
) -> tuple[float, float]:
    if w_base == 0.0 and w_cal == 0.0:
        return 0.0, 0.0
    try:
        low, high = block_bootstrap_ci(cal_daily, base_daily, val)
    except CompareError as exc:
        raise RunnerError(str(exc)) from exc
    if not (np.isfinite(low) and np.isfinite(high)):
        raise RunnerError("calibration: non-finite CI")
    return float(low), float(high)


def calibration_report(
    run_id: str, *, cfg: CalibrationConfig | None = None, vcfg: ValidationConfig | None = None,
    paths: ArtifactPaths | None = None, spec: CompetitionSpec | None = None,
) -> dict[str, object]:
    """run의 폴드 OOF로 운량 보정의 교차 적합 ΔW·CI·분기 부호와 pooled 계수를 계산해 반환한다. OOF·패널·구성 위반 시 RunnerError."""
    from src.oibc.config import CompetitionSpec as RecordedCompetitionSpec

    base = paths if paths is not None else ArtifactPaths()
    cal_cfg = cfg if cfg is not None else CalibrationConfig()
    val = vcfg if vcfg is not None else ValidationConfig()
    _safe_component(run_id)
    records = read_records(base.registry)
    match = next((r for r in records if r.run_id == run_id), None)
    _require(match is not None, f"registry: unknown run_id {run_id!r}")
    assert match is not None
    try:
        spec_data = dict(match.config["competition"])
        for key in ("train_end", "eval_start", "eval_end"):
            spec_data[key] = pd.Timestamp(spec_data[key]).tz_convert("Asia/Seoul")
        recorded = RecordedCompetitionSpec.model_validate(spec_data)
    except (KeyError, TypeError, ValueError) as exc:
        raise RunnerError("registry: invalid competition spec") from exc
    comp = spec if spec is not None else recorded
    _require(comp == recorded, "registry: competition spec mismatch")
    folds = list(match.folds)
    _require(bool(folds), "registry: empty folds")
    if len(folds) < 2:
        raise RunnerError("calibration: need >= 2 folds")
    composition_id = match.config.get("composition_id") or REFERENCE_COMP_ID
    sites = _read(DataPaths().canonical_dir / "sites.parquet", ["site_id", "current_group", "capacity_kw"])
    assignment, comp_sha = resolve_composition(composition_id, sites, paths=base, spec=comp)
    recorded_sha = match.config.get("composition_sha256")
    if isinstance(recorded_sha, str) and recorded_sha:
        _require(recorded_sha == comp_sha, "calibration: composition hash mismatch")
    panel_path_raw = match.config.get("panel_path")
    panel_sha = match.config.get("panel_sha256")
    if not isinstance(panel_path_raw, str) or not panel_path_raw or not isinstance(panel_sha, str) or not panel_sha:
        raise RunnerError("registry: missing panel provenance")
    panel_file = Path(panel_path_raw)
    try:
        current_hash = _sha256(panel_file)
    except OSError as exc:
        raise RunnerError(f"panel: unreadable ({exc})") from exc
    _require(current_hash == panel_sha, "calibration: panel hash mismatch")
    try:
        panel = pd.read_parquet(panel_file, columns=["ts", "period", *cal_cfg.cloud_vars])
    except (OSError, ValueError) as exc:
        raise RunnerError(f"panel: unreadable ({exc})") from exc
    if panel.columns.has_duplicates or not {"ts", "period", *cal_cfg.cloud_vars} <= set(panel.columns):
        raise RunnerError("panel: missing or duplicate columns")
    train_panel = panel.loc[panel["period"].eq("train")].reset_index(drop=True)
    if train_panel.empty:
        raise RunnerError("panel: empty train period")
    try:
        index = cloud_index(train_panel, cal_cfg)
    except CalibrationError as exc:
        raise RunnerError(str(exc)) from exc
    smp = _resolve_smp(pd.DataFrame(), comp)
    grouped: dict[str, pd.DataFrame] = {}
    hourly: dict[str, pd.DataFrame] = {}
    base_w: dict[str, float] = {}
    base_daily: dict[str, np.ndarray] = {}
    base_hourly_full: dict[str, pd.DataFrame] = {}
    for fold in folds:
        _safe_component(fold)
        oof = load_oof(run_id, fold, paths=base)
        _require(not oof.empty and bool((oof["ts"] <= comp.train_end).all()), "oof: empty or evaluation labels forbidden")
        frame = pd.DataFrame(
            {
                "site_id": oof["site_id"].astype("string").reset_index(drop=True),
                "ts": pd.Series(pd.DatetimeIndex(oof["ts"]), dtype=_DT),
                "pred_kwh": oof["pred_bag"].to_numpy(dtype="float64"),
                "actual_kwh": oof["actual_kwh"].to_numpy(dtype="float64"),
            }
        )
        try:
            groups = aggregate_groups(frame, assignment, spec=comp)
            base_res = settle(groups, smp, spec=comp)
        except SettlementError as exc:
            raise RunnerError(f"fold {fold}: {exc}") from exc
        grouped[fold] = groups
        hourly[fold] = base_res.hourly[["ts", "forecast_kwh", "actual_kwh", "smp", "k"]].copy()
        base_hourly_full[fold] = base_res.hourly
        base_w[fold] = float(base_res.penalty)
        daily = daily_totals(base_res)
        base_daily[fold] = daily["penalty"].to_numpy(dtype="float64")
    order = {name: pos for pos, name in enumerate(folds)}
    cross: list[_CrossEntry] = []
    pair_cal_daily: dict[tuple[str, str], np.ndarray] = {}
    pair_base_daily: dict[tuple[str, str], np.ndarray] = {}
    for apply in folds:
        for fit in folds:
            if fit == apply:
                continue
            try:
                cal = fit_cloud_calibration(hourly[fit], index, cal_cfg, comp)
                factors = scale_factors(grouped[apply]["ts"], index, cal)
            except CalibrationError as exc:
                raise RunnerError(str(exc)) from exc
            if not np.all(np.isfinite(factors)) or bool((factors < 0).any()):
                raise RunnerError("calibration: non-finite scale factors")
            cal_groups = grouped[apply].copy()
            cal_groups["forecast_kwh"] = (
                grouped[apply]["forecast_kwh"].to_numpy(dtype="float64") * factors
            )
            try:
                cal_res = settle(cal_groups, smp, spec=comp)
            except SettlementError as exc:
                raise RunnerError(f"fold {apply}: {exc}") from exc
            cal_daily = daily_totals(cal_res)["penalty"].to_numpy(dtype="float64")
            if cal_daily.shape != base_daily[apply].shape:
                raise RunnerError(f"fold {apply}: daily length mismatch")
            w_base = base_w[apply]
            w_cal = float(cal_res.penalty)
            rel = _rel_delta(w_cal, w_base)
            low, high = _pair_ci(cal_daily, base_daily[apply], w_cal, w_base, val)
            quarters = _quarter_rel_calibration(base_hourly_full[apply], cal_res.hourly)
            cross.append(
                {
                    "fit": fit,
                    "apply": apply,
                    "causal": bool(order[fit] < order[apply]),
                    "a": float(cal.a),
                    "b": float(cal.b),
                    "at_boundary": bool(cal.at_boundary),
                    "W_base": float(w_base),
                    "W_cal": float(w_cal),
                    "rel_delta": float(rel),
                    "ci_low": float(low),
                    "ci_high": float(high),
                    "quarter_rel": {k: float(v) for k, v in quarters.items()},
                }
            )
            pair_cal_daily[(fit, apply)] = cal_daily
            pair_base_daily[(fit, apply)] = base_daily[apply]
            logger.info(
                "[EVAL] calibration run=%s fit=%s apply=%s a=%r b=%r rel_delta=%r",
                run_id, fit, apply, float(cal.a), float(cal.b), float(rel),
            )
    pooled_w_cal = float(sum(e["W_cal"] for e in cross))
    pooled_w_base = float(sum(e["W_base"] for e in cross))
    pooled_rel = _rel_delta(pooled_w_cal, pooled_w_base)
    pair_order = sorted(cross, key=lambda e: (order[e["apply"]], order[e["fit"]]))
    pooled_cal_daily = np.concatenate([pair_cal_daily[(e["fit"], e["apply"])] for e in pair_order])
    pooled_base_daily = np.concatenate([pair_base_daily[(e["fit"], e["apply"])] for e in pair_order])
    pooled_low, pooled_high = _pair_ci(pooled_cal_daily, pooled_base_daily, pooled_w_cal, pooled_w_base, val)
    all_hourly = pd.concat([hourly[fold] for fold in folds], ignore_index=True)
    try:
        production = fit_cloud_calibration(all_hourly, index, cal_cfg, comp)
    except CalibrationError as exc:
        raise RunnerError(str(exc)) from exc
    all_groups = pd.concat([grouped[fold] for fold in folds], ignore_index=True)
    try:
        prod_factors = scale_factors(all_groups["ts"], index, production)
    except CalibrationError as exc:
        raise RunnerError(str(exc)) from exc
    prod_groups = all_groups.copy()
    prod_groups["forecast_kwh"] = all_groups["forecast_kwh"].to_numpy(dtype="float64") * prod_factors
    try:
        prod_res = settle(prod_groups, smp, spec=comp)
    except SettlementError as exc:
        raise RunnerError(f"production: {exc}") from exc
    in_sample = _rel_delta(float(prod_res.penalty), float(sum(base_w.values())))
    reasons: list[str] = [
        f"pair_not_improved={e['fit']}->{e['apply']}:rel_delta={e['rel_delta']:.4g}>=0"
        for e in cross
        if not e["rel_delta"] < 0.0
    ]
    if not pooled_rel <= -val.min_rel_improvement:
        reasons.append(f"pooled={pooled_rel:.4g}<min={val.min_rel_improvement:g}")
    if not pooled_high < 0.0:
        reasons.append(f"pooled_ci_high={pooled_high:.4g}>=0")
    reasons.extend(f"at_boundary={e['fit']}->{e['apply']}" for e in cross if e["at_boundary"])
    if production.at_boundary:
        reasons.append("at_boundary=production")
    decision = "adopt" if not reasons else "reject"
    payload: dict[str, object] = {
        "run_id": run_id,
        "panel_sha256": panel_sha,
        "composition_id": composition_id,
        "composition_sha256": comp_sha,
        "config": cal_cfg.model_dump(mode="json"),
        "cross_fit": cross,
        "pooled": {"rel_delta": float(pooled_rel), "ci_low": float(pooled_low), "ci_high": float(pooled_high)},
        "production": {
            "a": float(production.a),
            "b": float(production.b),
            "at_boundary": bool(production.at_boundary),
            "n_fit_hours": int(production.n_fit_hours),
            "rel_delta_in_sample": float(in_sample),
        },
        "decision": decision,
        "reasons": list(reasons),
    }
    try:
        json.dumps(payload, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise RunnerError(f"calibration: non-serializable report ({exc})") from exc
    logger.info("[EVAL] calibration run=%s decision=%s pooled_rel=%r", run_id, decision, float(pooled_rel))
    return payload
