"""시간순 폴드 분할."""

from __future__ import annotations

import logging

import pandas as pd

from src.oibc.config import FoldSpec, ValidationConfig
from src.oibc.data.schemas import _DT
from src.oibc.timeaxis import TimeAxisError, require_op_dates

logger = logging.getLogger(__name__)


class SplitError(ValueError):
    """폴드 계약 위반."""


def _train_last_day(fold: FoldSpec, cfg: ValidationConfig) -> pd.Timestamp:
    return fold.test_start - pd.Timedelta(days=cfg.embargo_days + 1)


def _checked_dti(op_dates: pd.Series) -> pd.DatetimeIndex:
    try:
        require_op_dates(op_dates)
    except TimeAxisError as e:
        raise SplitError(str(e)) from e
    return pd.DatetimeIndex(op_dates)


def fold_masks(
    op_dates: pd.Series, fold: FoldSpec, cfg: ValidationConfig
) -> tuple[pd.Series, pd.Series]:
    """행별 운영일에서 (학습, 시험) 불리언 마스크를 반환한다. 운영일 계약 위반 시 SplitError."""
    if op_dates.empty:
        empty = pd.Series([], dtype="bool", index=op_dates.index)
        return empty, empty.copy()
    dti = _checked_dti(op_dates)
    last_train = _train_last_day(fold, cfg)
    train = pd.Series(dti <= last_train, index=op_dates.index, dtype="bool")
    test = pd.Series(
        (dti >= fold.test_start) & (dti <= fold.test_end),
        index=op_dates.index,
        dtype="bool",
    )
    if bool((train & test).any()):
        raise SplitError("train and test masks overlap")
    return train, test


def site_history(
    site_ids: pd.Series, op_dates: pd.Series, fold: FoldSpec, cfg: ValidationConfig
) -> pd.DataFrame:
    """관측 행(site_id, op_date)에서 폴드 학습 구간의 발전소별 이력을 반환한다."""
    if len(site_ids) != len(op_dates) or not site_ids.index.equals(op_dates.index):
        raise SplitError("site_ids and op_dates must have equal length and index")
    if bool(site_ids.isna().any()):
        raise SplitError("site_ids: null value")
    if len(site_ids) == 0:
        out = pd.DataFrame(
            {
                "site_id": pd.Series([], dtype=site_ids.dtype),
                "n_train_days": pd.Series([], dtype="int32"),
                "first_op_date": pd.Series([], dtype=_DT),
                "is_warm": pd.Series([], dtype="bool"),
            }
        )
        logger.info("[FCST] fold=%s sites=0 warm=0 cold=0", fold.name)
        return out
    dti = _checked_dti(op_dates)
    last_train = _train_last_day(fold, cfg)
    train_flag = dti <= last_train
    sites = pd.Series(site_ids.to_numpy(), index=site_ids.index)
    train_sites = sites[train_flag]
    train_dates = dti[train_flag]
    grouped = pd.DataFrame({"site_id": train_sites.to_numpy(), "op_date": train_dates}).groupby(
        "site_id", sort=True
    )["op_date"]
    n_map = grouped.nunique()
    first_map = grouped.min()
    ordered = pd.unique(sites.to_numpy())
    n_days = [int(n_map.get(s, 0)) for s in ordered]
    firsts = [first_map.get(s, pd.NaT) for s in ordered]
    out = pd.DataFrame(
        {
            "site_id": pd.Series(ordered, dtype=sites.dtype),
            "n_train_days": pd.Series(n_days, dtype="int32"),
            "first_op_date": pd.Series(pd.DatetimeIndex(firsts, tz="Asia/Seoul")).astype(_DT),
            "is_warm": pd.Series(
                [n >= cfg.min_warm_days for n in n_days], dtype="bool"
            ),
        }
    )
    n_warm = int(out["is_warm"].sum())
    logger.info(
        "[FCST] fold=%s sites=%d warm=%d cold=%d",
        fold.name,
        len(out),
        n_warm,
        len(out) - n_warm,
    )
    return out


def assert_causal_split(
    train_op_dates: pd.Series,
    test_op_dates: pd.Series,
    fold: FoldSpec,
    cfg: ValidationConfig,
) -> None:
    """학습 행의 운영일이 엠바고 경계 이전이고 시험 행이 폴드 구간 안인지 검사한다. 위반 시 SplitError."""
    if len(train_op_dates) == 0 or len(test_op_dates) == 0:
        raise SplitError("train and test sets must be non-empty")
    train_dti = _checked_dti(train_op_dates)
    test_dti = _checked_dti(test_op_dates)
    last_train = _train_last_day(fold, cfg)
    if bool((train_dti > last_train).any()):
        raise SplitError("train op_date violates the embargo boundary")
    if bool(((test_dti < fold.test_start) | (test_dti > fold.test_end)).any()):
        raise SplitError("test op_date outside the fold window")
