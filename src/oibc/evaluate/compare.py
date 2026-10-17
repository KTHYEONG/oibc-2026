"""후보·기준 일별 정산 비교와 채택 판정."""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import NormalDist
from typing import Literal

import numpy as np
import pandas as pd

from src.oibc.config import ValidationConfig
from src.oibc.data.schemas import _DT

logger = logging.getLogger(__name__)

DailyRuns = Mapping[str, Sequence[pd.DataFrame]]

Decision = Literal["adopt", "reject", "inconclusive"]


class CompareError(ValueError):
    """비교 입력 계약 위반."""


@dataclass(frozen=True)
class FoldComparison:
    """한 폴드에서 후보 대 기준의 시드 평균 비교 결과."""

    fold: str
    w_candidate: float
    w_baseline: float
    rel_delta: float
    ci_low: float
    ci_high: float
    quarter_rel: dict[str, float]
    seed_range_candidate: float
    seed_range_baseline: float
    se_boot: float
    se_train: float
    ci_total_low: float
    ci_total_high: float


@dataclass(frozen=True)
class Comparison:
    """전체 폴드 비교 결과와 채택 판정."""

    folds: tuple[FoldComparison, ...]
    pooled_rel_delta: float
    quarter_win_ratio: float
    max_seed_range: float
    decision: Decision
    reasons: tuple[str, ...]
    train_noise_sd: float
    train_noise_df: int
    pooled_se_train: float
    pooled_ci_total_low: float
    pooled_ci_total_high: float


def _frame_vectors(frame: pd.DataFrame, label: str) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    if "penalty" not in frame.columns or "revenue" not in frame.columns:
        raise CompareError(f"{label}: missing penalty or revenue column")
    index = frame.index
    if not isinstance(index, pd.DatetimeIndex):
        raise CompareError(f"{label}: op_date index must be a DatetimeIndex")
    dti = pd.DatetimeIndex(index)
    if str(dti.dtype) != _DT or not dti.is_unique or not dti.is_monotonic_increasing or len(dti) == 0:
        raise CompareError(f"{label}: op_date index must be unique, ascending {_DT}")
    penalty = pd.to_numeric(frame["penalty"], errors="coerce").to_numpy(dtype="float64")
    if not np.all(np.isfinite(penalty)) or bool((penalty < 0).any()):
        raise CompareError(f"{label}: penalty must be finite and >= 0")
    revenue = pd.to_numeric(frame["revenue"], errors="coerce").to_numpy(dtype="float64")
    if not np.all(np.isfinite(revenue)):
        raise CompareError(f"{label}: revenue must be finite")
    return dti, penalty, revenue


def _seed_range(totals: np.ndarray) -> float:
    if totals.size <= 1:
        return 0.0
    mean = float(totals.mean())
    if mean == 0.0:
        return 0.0
    return float(totals.max() - totals.min()) / mean


def _block_ci(
    cand: np.ndarray, base: np.ndarray, rng: np.random.Generator, cfg: ValidationConfig
) -> tuple[float, float]:
    n_days = cand.size
    width = cfg.bootstrap_block_days
    n_blocks = (n_days + width - 1) // width
    edges = [min((i + 1) * width, n_days) for i in range(n_blocks)]
    starts = [i * width for i in range(n_blocks)]
    cand_blocks = np.array([cand[s:e].sum() for s, e in zip(starts, edges, strict=True)], dtype=np.float64)
    base_blocks = np.array([base[s:e].sum() for s, e in zip(starts, edges, strict=True)], dtype=np.float64)
    picks = rng.integers(0, n_blocks, size=(cfg.bootstrap_n, n_blocks))
    sum_c = cand_blocks[picks].sum(axis=1)
    sum_b = base_blocks[picks].sum(axis=1)
    if bool((sum_b == 0.0).any()):
        raise CompareError("bootstrap replicate with zero baseline total")
    ratios = sum_c / sum_b - 1.0
    low_pct = (1.0 - cfg.ci_level) / 2.0 * 100.0
    high_pct = (1.0 + cfg.ci_level) / 2.0 * 100.0
    return float(np.percentile(ratios, low_pct)), float(np.percentile(ratios, high_pct))


def block_bootstrap_ci(cand_daily: np.ndarray, base_daily: np.ndarray, cfg: ValidationConfig) -> tuple[float, float]:
    """운영일 순서의 일별 벌점 두 배열로 상대 ΔW의 블록 부트스트랩 CI를 반환한다. 길이·값 위반 시 CompareError."""
    try:
        cand = np.asarray(cand_daily, dtype="float64")
        base = np.asarray(base_daily, dtype="float64")
    except (ValueError, TypeError) as exc:
        raise CompareError("daily penalties must be numeric") from exc
    if cand.ndim != 1 or base.ndim != 1:
        raise CompareError("daily penalties must be 1-D")
    if cand.size < 1 or base.size < 1:
        raise CompareError("daily penalties must not be empty")
    if cand.shape != base.shape:
        raise CompareError("daily penalty shapes must match")
    if not np.all(np.isfinite(cand)) or not np.all(np.isfinite(base)):
        raise CompareError("daily penalties must be finite")
    if bool((cand < 0).any()) or bool((base < 0).any()):
        raise CompareError("daily penalties must be >= 0")
    rng = np.random.default_rng(cfg.bootstrap_seed)
    return _block_ci(cand, base, rng, cfg)


def _quarter_rel(
    cand: np.ndarray, base: np.ndarray, days: pd.DatetimeIndex, fold: str, reasons: list[str]
) -> dict[str, float]:
    local = days.tz_convert("Asia/Seoul")
    labels = np.array([f"{y}Q{q}" for y, q in zip(local.year.tolist(), local.quarter.tolist(), strict=True)])
    out: dict[str, float] = {}
    for label in dict.fromkeys(labels.tolist()):
        mask = labels == label
        denom = float(base[mask].sum())
        if denom == 0.0:
            reasons.append(f"quarter_excluded={fold}:{label}")
            continue
        out[label] = float(cand[mask].sum()) / denom - 1.0
    return out


@dataclass
class _FoldIntermediate:
    fold: str
    w_candidate: float
    w_baseline: float
    rel_delta: float
    ci_low: float
    ci_high: float
    quarter_rel: dict[str, float]
    seed_range_candidate: float
    seed_range_baseline: float
    cand_totals: np.ndarray
    base_totals: np.ndarray


def _fold_intermediate(
    fold: str,
    cand_frames: Sequence[pd.DataFrame],
    base_frames: Sequence[pd.DataFrame],
    rng: np.random.Generator,
    reasons: list[str],
    cfg: ValidationConfig,
) -> _FoldIntermediate:
    if len(cand_frames) == 0 or len(base_frames) == 0:
        raise CompareError(f"fold {fold}: need at least one seed result per side")
    if len(cand_frames) != len(base_frames):
        logger.warning("[EVAL] fold=%s seed count differs candidate=%d baseline=%d", fold, len(cand_frames), len(base_frames))
    parsed = [_frame_vectors(f, f"{fold}/candidate[{i}]") for i, f in enumerate(cand_frames)]
    parsed += [_frame_vectors(f, f"{fold}/baseline[{i}]") for i, f in enumerate(base_frames)]
    first_days = parsed[0][0]
    for dti, _, _ in parsed[1:]:
        if not dti.equals(first_days):
            raise CompareError(f"fold {fold}: op_date index mismatch")
    first_rev = parsed[0][2]
    for _, _, rev in parsed[1:]:
        if not np.allclose(rev, first_rev, rtol=1e-9, atol=1e-6):
            raise CompareError(f"fold {fold}: revenue mismatch")
    n_c = len(cand_frames)
    cand_mat = np.array([p[1] for p in parsed[:n_c]], dtype=np.float64)
    base_mat = np.array([p[1] for p in parsed[n_c:]], dtype=np.float64)
    cand = cand_mat.mean(axis=0)
    base = base_mat.mean(axis=0)
    w_c, w_b = float(cand.sum()), float(base.sum())
    if w_b <= 0.0:
        raise CompareError(f"fold {fold}: baseline total W must be > 0")
    rel = w_c / w_b - 1.0
    ci_low, ci_high = _block_ci(cand, base, rng, cfg)
    quarters = _quarter_rel(cand, base, first_days, fold, reasons)
    return _FoldIntermediate(
        fold=fold,
        w_candidate=w_c,
        w_baseline=w_b,
        rel_delta=rel,
        ci_low=ci_low,
        ci_high=ci_high,
        quarter_rel=quarters,
        seed_range_candidate=_seed_range(cand_mat.sum(axis=1)),
        seed_range_baseline=_seed_range(base_mat.sum(axis=1)),
        cand_totals=cand_mat.sum(axis=1),
        base_totals=base_mat.sum(axis=1),
    )


def _z_value(ci_level: float) -> float:
    return float(NormalDist().inv_cdf((1.0 + ci_level) / 2.0))


def compare_runs(candidate: DailyRuns, baseline: DailyRuns, cfg: ValidationConfig) -> Comparison:
    """후보와 기준의 폴드·시드별 일별 정산으로 상대 ΔW, CI, 분기 부호를 계산하고 채택을 판정한다. 계약 위반 시 CompareError."""
    want = {f.name for f in cfg.folds}
    if set(candidate) != want or set(baseline) != want:
        raise CompareError(f"fold set must equal {sorted(want)}")
    rng = np.random.default_rng(cfg.bootstrap_seed)
    reasons: list[str] = []
    partials = tuple(
        _fold_intermediate(f.name, candidate[f.name], baseline[f.name], rng, reasons, cfg) for f in cfg.folds
    )
    sq_sum = 0.0
    df = 0
    for part in partials:
        for side, totals in (("candidate", part.cand_totals), ("baseline", part.base_totals)):
            n = int(totals.size)
            mean = float(totals.mean())
            if mean == 0.0:
                reasons.append(f"noise_excluded={part.fold}:{side}")
                continue
            dev = totals / mean - 1.0
            sq_sum += float(np.sum(dev * dev))
            df += n - 1
    sd = float(math.sqrt(sq_sum / df)) if df > 0 else 0.0
    z = _z_value(cfg.ci_level)
    se_boot_list: list[float] = []
    se_train_list: list[float] = []
    total_low: list[float] = []
    total_high: list[float] = []
    for part in partials:
        se_boot = (part.ci_high - part.ci_low) / (2.0 * z)
        n_c = int(part.cand_totals.size)
        n_b = int(part.base_totals.size)
        se_train = sd * math.sqrt(1.0 / n_c + 1.0 / n_b)
        width = math.sqrt(se_boot * se_boot + se_train * se_train)
        se_boot_list.append(se_boot)
        se_train_list.append(se_train)
        total_low.append(part.rel_delta - z * width)
        total_high.append(part.rel_delta + z * width)
    folds = tuple(
        FoldComparison(
            fold=part.fold,
            w_candidate=part.w_candidate,
            w_baseline=part.w_baseline,
            rel_delta=part.rel_delta,
            ci_low=part.ci_low,
            ci_high=part.ci_high,
            quarter_rel=part.quarter_rel,
            seed_range_candidate=part.seed_range_candidate,
            seed_range_baseline=part.seed_range_baseline,
            se_boot=se_boot_list[i],
            se_train=se_train_list[i],
            ci_total_low=total_low[i],
            ci_total_high=total_high[i],
        )
        for i, part in enumerate(partials)
    )
    pooled = float(sum(f.rel_delta for f in folds) / len(folds))
    wins = sum(1 for f in folds for v in f.quarter_rel.values() if v < 0.0)
    total = sum(len(f.quarter_rel) for f in folds)
    ratio = float(wins / total) if total > 0 else 0.0
    largest = max(max(f.seed_range_candidate, f.seed_range_baseline) for f in folds)
    n_folds = float(len(folds))
    pooled_se_train = float(math.sqrt(sum(s * s for s in se_train_list)) / n_folds)
    pooled_se_total = float(math.sqrt(sum(a * a + b * b for a, b in zip(se_boot_list, se_train_list, strict=True))) / n_folds)
    pooled_ci_low = float(pooled - z * pooled_se_total)
    pooled_ci_high = float(pooled + z * pooled_se_total)
    improved = [f.fold for f in folds if f.rel_delta >= 0.0]
    reasons.extend(f"fold_not_improved={name}" for name in improved)
    if -pooled < cfg.min_rel_improvement:
        reasons.append(f"pooled={pooled:.4g}<min={cfg.min_rel_improvement:g}")
    if df < cfg.min_noise_df:
        reasons.append(f"train_noise_df={df}<min={cfg.min_noise_df}")
    noise_threshold = cfg.noise_z * pooled_se_train
    if not -pooled > noise_threshold:
        reasons.append(f"train_noise={-pooled:.4g} <= z*se={noise_threshold:.4g}")
    if not pooled_ci_high < 0.0:
        reasons.append(f"ci_total_high={pooled_ci_high:.4g}>=0")
    if ratio < cfg.min_quarter_win_ratio:
        reasons.append(f"quarters={ratio:.4g}<min={cfg.min_quarter_win_ratio:g}")
    all_improved = not improved
    adoptable = (
        all_improved
        and -pooled >= cfg.min_rel_improvement
        and df >= cfg.min_noise_df
        and -pooled > noise_threshold
        and pooled_ci_high < 0.0
        and ratio >= cfg.min_quarter_win_ratio
    )
    if adoptable:
        decision: Decision = "adopt"
        kept = [r for r in reasons if r.startswith("quarter_excluded=") or r.startswith("noise_excluded=")]
        reasons = kept
    elif all(f.rel_delta > 0.0 for f in folds) and pooled >= cfg.min_rel_improvement and pooled_ci_low > 0.0:
        decision = "reject"
    else:
        decision = "inconclusive"
    logger.info(
        "[EVAL] decision=%s pooled=%r train_noise_sd=%r df=%r pooled_ci_total=[%r,%r] quarters=%r reasons=%r",
        decision,
        pooled,
        sd,
        df,
        pooled_ci_low,
        pooled_ci_high,
        ratio,
        reasons,
    )
    result = Comparison(
        folds=folds,
        pooled_rel_delta=pooled,
        quarter_win_ratio=ratio,
        max_seed_range=largest,
        decision=decision,
        reasons=tuple(reasons),
        train_noise_sd=sd,
        train_noise_df=df,
        pooled_se_train=pooled_se_train,
        pooled_ci_total_low=pooled_ci_low,
        pooled_ci_total_high=pooled_ci_high,
    )
    comparison_to_dict(result)
    return result


def comparison_to_dict(comparison: Comparison) -> dict[str, object]:
    """Comparison을 JSON 직렬화 가능한 dict로 변환한다."""
    payload: dict[str, object] = {
        "folds": [
            {
                "fold": f.fold,
                "w_candidate": float(f.w_candidate),
                "w_baseline": float(f.w_baseline),
                "rel_delta": float(f.rel_delta),
                "ci_low": float(f.ci_low),
                "ci_high": float(f.ci_high),
                "quarter_rel": {k: float(v) for k, v in f.quarter_rel.items()},
                "seed_range_candidate": float(f.seed_range_candidate),
                "seed_range_baseline": float(f.seed_range_baseline),
                "se_boot": float(f.se_boot),
                "se_train": float(f.se_train),
                "ci_total_low": float(f.ci_total_low),
                "ci_total_high": float(f.ci_total_high),
            }
            for f in comparison.folds
        ],
        "pooled_rel_delta": float(comparison.pooled_rel_delta),
        "quarter_win_ratio": float(comparison.quarter_win_ratio),
        "max_seed_range": float(comparison.max_seed_range),
        "decision": comparison.decision,
        "reasons": list(comparison.reasons),
        "train_noise_sd": float(comparison.train_noise_sd),
        "train_noise_df": int(comparison.train_noise_df),
        "pooled_se_train": float(comparison.pooled_se_train),
        "pooled_ci_total_low": float(comparison.pooled_ci_total_low),
        "pooled_ci_total_high": float(comparison.pooled_ci_total_high),
    }
    try:
        json.dumps(payload, allow_nan=False)
    except ValueError as exc:
        raise CompareError("comparison values must be finite") from exc
    return payload
