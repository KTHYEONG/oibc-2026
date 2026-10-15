import pandas as pd
import pytest

from src.oibc.config import FoldSpec, ValidationConfig
from src.oibc.cv.splits import SplitError, assert_causal_split, fold_masks, site_history
from src.oibc.timeaxis import TimeAxisError

KST = "Asia/Seoul"


def _fold_b(**over: object) -> tuple[FoldSpec, ValidationConfig]:
    cfg = ValidationConfig(**over) if over else ValidationConfig()
    fold = next(f for f in cfg.folds if f.name == "B")
    return fold, cfg


def _ops(vals: list[str]) -> pd.Series:
    return pd.Series(pd.DatetimeIndex(pd.to_datetime(vals).tz_localize(KST)))


def test_embargo_boundary() -> None:
    fold, cfg = _fold_b()
    op = _ops(["2024-05-29", "2024-05-30", "2024-05-31", "2024-06-01"])
    train, test = fold_masks(op, fold, cfg)
    assert train.tolist() == [True, True, False, False]
    assert test.tolist() == [False, False, False, True]


def test_embargo_zero_includes_boundary() -> None:
    fold, cfg = _fold_b(embargo_days=0)
    op = _ops(["2024-05-29", "2024-05-30", "2024-05-31", "2024-06-01"])
    train, test = fold_masks(op, fold, cfg)
    assert train.tolist() == [True, True, True, False]
    assert test.tolist() == [False, False, False, True]


def test_train_test_disjoint_all_folds() -> None:
    cfg = ValidationConfig()
    op = pd.Series(pd.date_range("2023-01-01", "2025-05-31", freq="D", tz=KST))
    for fold in cfg.folds:
        train, test = fold_masks(op, fold, cfg)
        assert not bool((train & test).any())
        assert train.dtype == bool and test.dtype == bool
        assert train.index.equals(op.index)


def test_eval_period_excluded() -> None:
    fold, cfg = _fold_b()
    op = _ops(["2025-06-01"])
    train, test = fold_masks(op, fold, cfg)
    assert train.tolist() == [False] and test.tolist() == [False]


@pytest.mark.parametrize(
    "op",
    [
        pd.Series(pd.to_datetime(["2024-05-29"])),
        pd.Series(pd.DatetimeIndex(pd.to_datetime(["2024-06-01 01:00"]).tz_localize(KST))),
    ],
)
def test_op_date_contract_violation(op: pd.Series) -> None:
    fold, cfg = _fold_b()
    with pytest.raises((SplitError, TimeAxisError)):
        fold_masks(op, fold, cfg)


def test_cold_start_classification() -> None:
    fold, cfg = _fold_b()
    sites = pd.Series(["X"] * 3 + ["Y"])
    op = _ops(["2024-05-27", "2024-05-28", "2024-05-29", "2024-06-05"])
    out = site_history(sites, op, fold, cfg)
    row_x = out.loc[out["site_id"] == "X"].iloc[0]
    row_y = out.loc[out["site_id"] == "Y"].iloc[0]
    assert int(row_x["n_train_days"]) == 3 and bool(row_x["is_warm"])
    assert int(row_y["n_train_days"]) == 0 and not bool(row_y["is_warm"])
    assert pd.isna(row_y["first_op_date"])


def test_history_causal_to_post_fold_rows() -> None:
    fold, cfg = _fold_b()
    sites = pd.Series(["X"] * 3 + ["Y"])
    op = _ops(["2024-05-27", "2024-05-28", "2024-05-29", "2024-06-05"])
    before = site_history(sites, op, fold, cfg)
    sites2 = pd.Series(["X"] * 3 + ["Y", "Y"])
    op2 = _ops(["2024-05-27", "2024-05-28", "2024-05-29", "2024-06-05", "2025-06-10"])
    after = site_history(sites2, op2, fold, cfg)
    pd.testing.assert_frame_equal(before, after)


def test_causal_guard_rejects_embargo_violation() -> None:
    fold, cfg = _fold_b()
    train = _ops(["2024-05-29", "2024-05-31"])
    test = _ops(["2024-06-05"])
    with pytest.raises(SplitError):
        assert_causal_split(train, test, fold, cfg)


def test_empty_split_rejected() -> None:
    fold, cfg = _fold_b()
    with pytest.raises(SplitError):
        assert_causal_split(_ops(["2024-05-29"]), _ops([]), fold, cfg)


def test_empty_masks() -> None:
    fold, cfg = _fold_b()
    op = pd.Series([], dtype="datetime64[us, Asia/Seoul]")
    train, test = fold_masks(op, fold, cfg)
    assert train.empty and test.empty
    assert train.dtype == bool and test.dtype == bool


def test_overlap_guard() -> None:
    fold, cfg = _fold_b()
    bad = ValidationConfig.model_construct(
        folds=cfg.folds,
        embargo_days=-1,
        min_warm_days=cfg.min_warm_days,
        seeds=cfg.seeds,
        bootstrap_block_days=cfg.bootstrap_block_days,
        bootstrap_n=cfg.bootstrap_n,
        bootstrap_seed=cfg.bootstrap_seed,
        ci_level=cfg.ci_level,
        min_rel_improvement=cfg.min_rel_improvement,
        min_quarter_win_ratio=cfg.min_quarter_win_ratio,
    )
    with pytest.raises(SplitError, match="overlap"):
        fold_masks(_ops(["2024-06-01"]), fold, bad)


def test_history_length_mismatch_rejected() -> None:
    fold, cfg = _fold_b()
    with pytest.raises(SplitError):
        site_history(pd.Series(["X", "Y"]), _ops(["2024-05-29"]), fold, cfg)
    with pytest.raises(SplitError):
        site_history(
            pd.Series(["X"], index=[1]),
            pd.Series(pd.DatetimeIndex(pd.to_datetime(["2024-05-29"]).tz_localize(KST)), index=[0]),
            fold,
            cfg,
        )


def test_empty_history() -> None:
    fold, cfg = _fold_b()
    out = site_history(
        pd.Series([], dtype="object"),
        pd.Series([], dtype="datetime64[us, Asia/Seoul]"),
        fold,
        cfg,
    )
    assert list(out.columns) == ["site_id", "n_train_days", "first_op_date", "is_warm"]
    assert out.empty


def test_causal_guard_rejects_test_outside_window() -> None:
    fold, cfg = _fold_b()
    with pytest.raises(SplitError):
        assert_causal_split(_ops(["2024-05-29"]), _ops(["2025-06-10"]), fold, cfg)


def test_history_null_site_rejected() -> None:
    fold, cfg = _fold_b()
    with pytest.raises(SplitError):
        site_history(pd.Series(["X", None]), _ops(["2024-05-29", "2024-05-30"]), fold, cfg)
