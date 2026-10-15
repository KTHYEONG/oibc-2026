"""후보·기준 일별 정산 비교와 채택 판정."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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


@dataclass(frozen=True)
class Comparison:
    """전체 폴드 비교 결과와 채택 판정."""

    folds: tuple[FoldComparison, ...]
    pooled_rel_delta: float
    quarter_win_ratio: float
    max_seed_range: float
    decision: Decision
    reasons: tuple[str, ...]


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


def _fold_result(
    fold: str,
    cand_frames: Sequence[pd.DataFrame],
    base_frames: Sequence[pd.DataFrame],
    rng: np.random.Generator,
    cfg: ValidationConfig,
    reasons: list[str],
) -> FoldComparison:
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
    n_c, n_b = len(cand_frames), len(base_frames)
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
    result = FoldComparison(
        fold=fold,
        w_candidate=w_c,
        w_baseline=w_b,
        rel_delta=rel,
        ci_low=ci_low,
        ci_high=ci_high,
        quarter_rel=quarters,
        seed_range_candidate=_seed_range(cand_mat.sum(axis=1)),
        seed_range_baseline=_seed_range(base_mat.sum(axis=1)),
    )
    logger.info(
        "[EVAL] fold=%s W_c=%r W_b=%r rel=%r ci=[%r,%r] seeds=%d/%d",
        fold,
        w_c,
        w_b,
        rel,
        ci_low,
        ci_high,
        n_c,
        n_b,
    )
    return result


def compare_runs(candidate: DailyRuns, baseline: DailyRuns, cfg: ValidationConfig) -> Comparison:
    """후보와 기준의 폴드·시드별 일별 정산으로 상대 ΔW, CI, 분기 부호를 계산하고 채택을 판정한다. 계약 위반 시 CompareError."""
    want = {f.name for f in cfg.folds}
    if set(candidate) != want or set(baseline) != want:
        raise CompareError(f"fold set must equal {sorted(want)}")
    rng = np.random.default_rng(cfg.bootstrap_seed)
    reasons: list[str] = []
    folds = tuple(
        _fold_result(f.name, candidate[f.name], baseline[f.name], rng, cfg, reasons) for f in cfg.folds
    )
    pooled = float(sum(f.rel_delta for f in folds) / len(folds))
    wins = sum(1 for f in folds for v in f.quarter_rel.values() if v < 0.0)
    total = sum(len(f.quarter_rel) for f in folds)
    ratio = float(wins / total) if total > 0 else 0.0
    largest = max(max(f.seed_range_candidate, f.seed_range_baseline) for f in folds)
    improved = [f.fold for f in folds if f.rel_delta >= 0.0]
    reasons.extend(f"fold_not_improved={name}" for name in improved)
    if -pooled < cfg.min_rel_improvement:
        reasons.append(f"pooled={pooled:.4g}<min={cfg.min_rel_improvement:g}")
    if -pooled <= largest:
        reasons.append(f"seed_noise={largest:.4g}>=gain={-pooled:.4g}")
    if ratio < cfg.min_quarter_win_ratio:
        reasons.append(f"quarters={ratio:.4g}<min={cfg.min_quarter_win_ratio:g}")
    if not improved and -pooled >= cfg.min_rel_improvement and -pooled > largest and ratio >= cfg.min_quarter_win_ratio:
        decision: Decision = "adopt"
        kept = [r for r in reasons if r.startswith("quarter_excluded=")]
        reasons = kept
    elif all(f.rel_delta > 0.0 for f in folds) and pooled >= cfg.min_rel_improvement:
        decision = "reject"
    else:
        decision = "inconclusive"
    logger.info(
        "[EVAL] decision=%s pooled=%r seed_noise=%r quarters=%r reasons=%r",
        decision,
        pooled,
        largest,
        ratio,
        reasons,
    )
    return Comparison(
        folds=folds,
        pooled_rel_delta=pooled,
        quarter_win_ratio=ratio,
        max_seed_range=largest,
        decision=decision,
        reasons=tuple(reasons),
    )


def comparison_to_dict(comparison: Comparison) -> dict[str, object]:
    """Comparison을 JSON 직렬화 가능한 dict로 변환한다."""
    return {
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
            }
            for f in comparison.folds
        ],
        "pooled_rel_delta": float(comparison.pooled_rel_delta),
        "quarter_win_ratio": float(comparison.quarter_win_ratio),
        "max_seed_range": float(comparison.max_seed_range),
        "decision": comparison.decision,
        "reasons": list(comparison.reasons),
    }
