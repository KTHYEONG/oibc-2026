"""집합 잔차 상쇄 분해."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.oibc.config import CompetitionSpec
from src.oibc.timeaxis import TimeAxisError, k_weight, require_hourly_kst

logger = logging.getLogger(__name__)


class NettingError(ValueError):
    """상쇄 분해 입력 위반."""


@dataclass(frozen=True)
class NettingResult:
    """배정 1개의 정산 벌점 분해(원 단위)."""

    penalty: float
    pool_penalty: float
    netting_loss: float


def _spec_or_default(spec: CompetitionSpec | None) -> CompetitionSpec:
    return spec if spec is not None else CompetitionSpec()


class NettingEvaluator:
    """폴드 OOF와 SMP로 잔차 행렬을 1회 구성해 여러 배정의 벌점을 계산한다."""

    def __init__(
        self,
        oof: pd.DataFrame,
        smp: pd.DataFrame,
        *,
        pred_col: str = "pred_bag",
        spec: CompetitionSpec | None = None,
    ) -> None:
        """OOF(site_id, ts, actual_kwh, pred_col)와 SMP(ts, smp, k)를 검증해 행렬을 만든다. 위반 시 NettingError."""
        cfg = _spec_or_default(spec)
        if oof.columns.has_duplicates or smp.columns.has_duplicates or oof.empty:
            raise NettingError("netting: duplicate columns or empty oof")
        for col in ("site_id", "ts", "actual_kwh", pred_col):
            if col not in oof.columns:
                raise NettingError(f"oof: missing column {col!r}")
        for col in ("ts", "smp", "k"):
            if col not in smp.columns:
                raise NettingError(f"smp: missing column {col!r}")
        if oof[["site_id", "ts"]].isna().any().any():
            raise NettingError("oof: null key")
        site_keys = oof["site_id"].astype("string")
        if pd.MultiIndex.from_arrays([site_keys, oof["ts"]]).has_duplicates:
            raise NettingError("oof: duplicate key (site_id, ts)")
        try:
            require_hourly_kst(oof["ts"], "oof.ts")
        except (TimeAxisError, ValueError) as exc:
            raise NettingError(str(exc)) from exc
        try:
            require_hourly_kst(smp["ts"], "smp.ts")
        except (TimeAxisError, ValueError) as exc:
            raise NettingError(str(exc)) from exc
        if smp.duplicated(subset=["ts"]).any():
            raise NettingError("smp: duplicate ts")
        actual = pd.to_numeric(oof["actual_kwh"], errors="coerce").to_numpy(dtype="float64")
        pred = pd.to_numeric(oof[pred_col], errors="coerce").to_numpy(dtype="float64")
        if not np.all(np.isfinite(actual)):
            raise NettingError("actual_kwh: must be finite")
        if not np.all(np.isfinite(pred)):
            raise NettingError(f"{pred_col}: must be finite")
        smp_vals = pd.to_numeric(smp["smp"], errors="coerce").to_numpy(dtype="float64")
        if not np.all(np.isfinite(smp_vals)) or bool((smp_vals < 0).any()):
            raise NettingError("smp.smp: must be finite and >= 0")
        expected_k = k_weight(smp["ts"], cfg).to_numpy(dtype="float64")
        got_k = pd.to_numeric(smp["k"], errors="coerce").to_numpy(dtype="float64")
        if not np.array_equal(got_k, expected_k):
            raise NettingError("smp.k: mismatch with k_weight")
        oof_ts = pd.DatetimeIndex(oof["ts"]).as_unit("us")
        smp_ts = pd.DatetimeIndex(smp["ts"]).as_unit("us")
        stamps = oof_ts.unique().sort_values()
        positions = smp_ts.get_indexer(stamps)
        if (positions < 0).any():
            raise NettingError("oof: ts missing in smp")
        sites = pd.Index(site_keys.unique()).sort_values()
        mat = np.zeros((len(stamps), len(sites)), dtype="float64")
        with np.errstate(over="ignore", invalid="ignore"):
            mat[stamps.get_indexer(oof_ts), sites.get_indexer(site_keys)] = pred - actual
            pool_error = mat.sum(axis=1)
            weight = smp_vals[positions] * got_k[positions] * cfg.penalty_rate
            pool = float(weight @ np.abs(pool_error))
        if not np.isfinite(pool):
            raise NettingError("netting: non-finite pool")
        self._spec = cfg
        self._sites = tuple(sites)
        self._matrix = mat
        self._pool_error = pool_error
        self._weight = weight
        self._pool = pool
        logger.info("[EVAL] netting T=%d N=%d pool=%r", len(stamps), len(sites), pool)

    @property
    def site_ids(self) -> tuple[str, ...]:
        """OOF 사이트 식별자 모음."""
        return self._sites

    @property
    def pool_penalty(self) -> float:
        """전 발전소를 한 집합으로 묶었을 때의 벌점(배정 무관 하한)."""
        return self._pool

    def _check_assignment(self, assignment: pd.DataFrame) -> np.ndarray:
        if list(assignment.columns) != ["site_id", "group"]:
            raise NettingError(f"assignment: columns must be ['site_id', 'group'] got={list(assignment.columns)}")
        if assignment["site_id"].astype("string").duplicated().any():
            raise NettingError("assignment: duplicate site_id")
        if assignment.isna().any().any():
            raise NettingError("assignment: null value")
        allowed = set(self._spec.groups)
        if (~assignment["group"].astype("string").isin(allowed)).any():
            raise NettingError("assignment: unknown group")
        want = set(self._sites)
        got = set(assignment["site_id"].astype("string").tolist())
        if not want <= got:
            raise NettingError("assignment: uncovered oof site")
        mapping = pd.Series(assignment["group"].to_numpy(), index=assignment["site_id"].astype("string"))
        return np.where(mapping.reindex(self._sites).eq(self._spec.groups[0]), 1.0, -1.0)

    def evaluate(self, assignment: pd.DataFrame) -> NettingResult:
        """(site_id, group) 배정의 벌점과 분해를 반환한다. 배정이 OOF 사이트를 덮지 않으면 NettingError."""
        penalty = float(self.penalties([assignment])[0])
        loss = penalty - self._pool
        if loss < -1e-6 * penalty:
            raise NettingError(f"netting_loss: numerical anomaly loss={loss!r} penalty={penalty!r}")
        return NettingResult(penalty=penalty, pool_penalty=self._pool, netting_loss=loss)

    def penalties(self, assignments: Sequence[pd.DataFrame]) -> np.ndarray:
        """여러 배정의 벌점 배열(float64, 입력 순서)을 한 번의 행렬 연산으로 반환한다."""
        n = len(self._sites)
        m = len(assignments)
        if m == 0:
            return np.empty((0,), dtype="float64")
        membership = np.empty((n, m), dtype="float64")
        for j, assignment in enumerate(assignments):
            membership[:, j] = self._check_assignment(assignment)
        with np.errstate(over="ignore", invalid="ignore"):
            total = self._matrix @ membership
            np.abs(total, out=total)
            # 두 그룹은 |a|+|b|=max(|a+b|, |a-b|)이므로 T*M 버퍼 하나면 된다.
            np.maximum(total, np.abs(self._pool_error)[:, None], out=total)
            out = self._weight @ total
        if not np.isfinite(out).all():
            raise NettingError("netting: non-finite group penalty")
        return np.asarray(out, dtype="float64")
