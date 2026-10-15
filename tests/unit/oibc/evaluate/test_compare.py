import json

import numpy as np
import pandas as pd
import pytest

from src.oibc.config import FoldSpec, ValidationConfig
from src.oibc.evaluate.compare import (
    CompareError,
    Comparison,
    compare_runs,
    comparison_to_dict,
)

KST = "Asia/Seoul"
DT = "datetime64[us, Asia/Seoul]"


def _cfg(**over: object) -> ValidationConfig:
    return ValidationConfig(bootstrap_n=200, **over)  # type: ignore[arg-type]


def _daily(first: str, n: int, penalty: float | list[float], revenue: float = 5.0) -> pd.DataFrame:
    idx = pd.DatetimeIndex(pd.date_range(first, periods=n, freq="D", tz=KST), dtype=DT)
    pen = (
        np.full(n, penalty, dtype=np.float64)
        if isinstance(penalty, float)
        else np.asarray(penalty, dtype=np.float64)
    )
    frame = pd.DataFrame(
        {
            "penalty": pen,
            "revenue": np.full(n, revenue, dtype=np.float64),
        },
        index=idx,
    )
    frame.index = pd.DatetimeIndex(frame.index, dtype=DT)
    return frame


def _runs(spec: dict[str, list[tuple[str, int, float]]]) -> dict[str, list[pd.DataFrame]]:
    return {fold: [_daily(first, n, pen) for first, n, pen in rows] for fold, rows in spec.items()}


def test_clear_improvement_adopted() -> None:
    cand = _runs({"A": [("2024-01-01", 120, 9.0)] * 3, "B": [("2024-06-01", 120, 9.0)] * 3})
    base = _runs({"A": [("2024-01-01", 120, 10.0)] * 3, "B": [("2024-06-01", 120, 10.0)] * 3})
    cmp = compare_runs(cand, base, _cfg())
    assert isinstance(cmp, Comparison)
    for fold in cmp.folds:
        assert fold.rel_delta == pytest.approx(-0.1)
        assert fold.ci_high < 0.0
    assert cmp.max_seed_range == 0.0
    assert cmp.decision == "adopt"


def test_one_fold_worse_inconclusive() -> None:
    cand = _runs({"A": [("2024-01-01", 60, 9.0)], "B": [("2024-06-01", 60, 10.1)]})
    base = _runs({"A": [("2024-01-01", 60, 10.0)], "B": [("2024-06-01", 60, 10.0)]})
    cmp = compare_runs(cand, base, _cfg())
    assert cmp.decision == "inconclusive"
    assert "fold_not_improved=B" in cmp.reasons


def test_seed_noise_inconclusive() -> None:
    cand = _runs({"A": [("2024-01-01", 100, 0.985)] * 3, "B": [("2024-06-01", 100, 0.985)] * 3})
    base = _runs(
        {
            "A": [("2024-01-01", 100, v) for v in (0.985, 1.0, 1.015)],
            "B": [("2024-06-01", 100, v) for v in (0.985, 1.0, 1.015)],
        }
    )
    cmp = compare_runs(cand, base, _cfg())
    assert cmp.decision == "inconclusive"
    assert cmp.max_seed_range == pytest.approx(0.03)
    assert any(r.startswith("seed_noise=") for r in cmp.reasons)


def test_clear_degradation_rejected() -> None:
    cand = _runs({"A": [("2024-01-01", 60, 10.5)], "B": [("2024-06-01", 60, 10.5)]})
    base = _runs({"A": [("2024-01-01", 60, 10.0)], "B": [("2024-06-01", 60, 10.0)]})
    assert compare_runs(cand, base, _cfg()).decision == "reject"


def test_mask_mismatch_rejected() -> None:
    base = _runs({"A": [("2024-01-01", 30, 10.0)], "B": [("2024-06-01", 30, 10.0)]})
    cand = _runs({"A": [("2024-01-01", 30, 9.0)], "B": [("2024-06-01", 30, 9.0)]})
    wrong_rev = _daily("2024-01-01", 30, 9.0, revenue=6.0)
    with pytest.raises(CompareError):
        compare_runs({"A": [wrong_rev], "B": cand["B"]}, base, _cfg())
    shifted = _daily("2024-01-02", 30, 9.0)
    with pytest.raises(CompareError):
        compare_runs({"A": [shifted], "B": cand["B"]}, base, _cfg())


def test_fold_set_mismatch_rejected() -> None:
    cand = _runs({"A": [("2024-01-01", 30, 9.0)]})
    base = _runs({"A": [("2024-01-01", 30, 10.0)], "B": [("2024-06-01", 30, 10.0)]})
    with pytest.raises(CompareError):
        compare_runs(cand, base, _cfg())


def test_paired_bootstrap_constant_ratio() -> None:
    shape = [5.0 + (i % 7) for i in range(90)]
    cand_vals = [v * 0.9 for v in shape]
    cand = {
        "A": [_daily("2024-01-01", 90, cand_vals)],
        "B": [_daily("2024-06-01", 90, cand_vals)],
    }
    base = {
        "A": [_daily("2024-01-01", 90, shape)],
        "B": [_daily("2024-06-01", 90, shape)],
    }
    cmp = compare_runs(cand, base, _cfg())
    for fold in cmp.folds:
        assert fold.ci_low == pytest.approx(-0.1, abs=1e-12)
        assert fold.ci_high == pytest.approx(-0.1, abs=1e-12)


def test_deterministic() -> None:
    cand = _runs({"A": [("2024-01-01", 60, 9.0)], "B": [("2024-06-01", 60, 9.5)]})
    base = _runs({"A": [("2024-01-01", 60, 10.0)], "B": [("2024-06-01", 60, 10.0)]})
    cfg = _cfg()
    first = compare_runs(cand, base, cfg)
    second = compare_runs(cand, base, cfg)
    assert first == second


def test_quarter_attribution() -> None:
    pen_c = [10.0] * 10
    pen_c[4] = 9.0
    pen_c[5] = 12.0
    cand = {"A": [_daily("2024-06-26", 10, pen_c)], "B": [_daily("2024-06-26", 10, pen_c)]}
    base = _runs({"A": [("2024-06-26", 10, 10.0)], "B": [("2024-06-26", 10, 10.0)]})
    cmp = compare_runs(cand, base, _cfg())
    assert set(cmp.folds[0].quarter_rel) == {"2024Q2", "2024Q3"}
    assert cmp.folds[0].quarter_rel["2024Q2"] < 0.0
    assert cmp.folds[0].quarter_rel["2024Q3"] > 0.0


def test_pooled_simple_mean() -> None:
    cand = {"A": [_daily("2024-01-01", 50, 1.9)], "B": [_daily("2024-06-01", 50, 19.8)]}
    base = {"A": [_daily("2024-01-01", 50, 2.0)], "B": [_daily("2024-06-01", 50, 20.0)]}
    cmp = compare_runs(cand, base, _cfg())
    assert cmp.pooled_rel_delta == pytest.approx(-0.03)


def test_serialization_roundtrip() -> None:
    cand = _runs({"A": [("2024-01-01", 60, 9.0)], "B": [("2024-06-01", 60, 9.0)]})
    base = _runs({"A": [("2024-01-01", 60, 10.0)], "B": [("2024-06-01", 60, 10.0)]})
    cmp = compare_runs(cand, base, _cfg())
    payload = comparison_to_dict(cmp)
    text = json.dumps(payload)
    loaded = json.loads(text)
    assert loaded["pooled_rel_delta"] == cmp.pooled_rel_delta
    assert loaded["folds"][0]["rel_delta"] == cmp.folds[0].rel_delta
    assert loaded["folds"][1]["quarter_rel"] == cmp.folds[1].quarter_rel


def test_seed_count_mismatch_allowed(caplog: pytest.LogCaptureFixture) -> None:
    cand = _runs({"A": [("2024-01-01", 60, 9.0)] * 2, "B": [("2024-06-01", 60, 9.0)] * 2})
    base = _runs({"A": [("2024-01-01", 60, 10.0)] * 3, "B": [("2024-06-01", 60, 10.0)] * 3})
    with caplog.at_level("WARNING"):
        cmp = compare_runs(cand, base, _cfg())
    assert cmp.decision == "adopt"


def test_zero_baseline_rejected() -> None:
    cand = _runs({"A": [("2024-01-01", 30, 0.0)], "B": [("2024-06-01", 30, 0.0)]})
    base = _runs({"A": [("2024-01-01", 30, 0.0)], "B": [("2024-06-01", 30, 10.0)]})
    with pytest.raises(CompareError):
        compare_runs(cand, base, _cfg())


def test_invalid_penalty_rejected() -> None:
    base = _runs({"A": [("2024-01-01", 30, 10.0)], "B": [("2024-06-01", 30, 10.0)]})
    bad = _daily("2024-01-01", 30, [9.0] * 29 + [float("nan")])
    with pytest.raises(CompareError):
        compare_runs({"A": [bad], "B": base["B"]}, base, _cfg())
    neg = _daily("2024-01-01", 30, [9.0] * 29 + [-1.0])
    with pytest.raises(CompareError):
        compare_runs({"A": [neg], "B": base["B"]}, base, _cfg())


def test_invalid_frame_rejected() -> None:
    base = _runs({"A": [("2024-01-01", 30, 10.0)], "B": [("2024-06-01", 30, 10.0)]})
    no_col = _daily("2024-01-01", 30, 9.0).drop(columns=["penalty"])
    with pytest.raises(CompareError):
        compare_runs({"A": [no_col], "B": base["B"]}, base, _cfg())
    no_dt = pd.DataFrame({"penalty": [9.0] * 30, "revenue": [5.0] * 30})
    with pytest.raises(CompareError):
        compare_runs({"A": [no_dt], "B": base["B"]}, base, _cfg())
    dup_idx = _daily("2024-01-01", 30, 9.0)
    dup_idx.index = pd.DatetimeIndex([dup_idx.index[0]] * 30, dtype=DT)
    with pytest.raises(CompareError):
        compare_runs({"A": [dup_idx], "B": base["B"]}, base, _cfg())
    bad_rev = _daily("2024-01-01", 30, 9.0)
    bad_rev["revenue"] = float("inf")
    with pytest.raises(CompareError):
        compare_runs({"A": [bad_rev], "B": base["B"]}, base, _cfg())
    with pytest.raises(CompareError):
        compare_runs({"A": [], "B": base["B"]}, base, _cfg())


def test_all_zero_candidate_seed_range() -> None:
    cand = _runs({"A": [("2024-01-01", 30, 0.0)] * 2, "B": [("2024-06-01", 30, 0.0)] * 2})
    base = _runs({"A": [("2024-01-01", 30, 10.0)], "B": [("2024-06-01", 30, 10.0)]})
    cmp = compare_runs(cand, base, _cfg())
    assert cmp.decision == "adopt"
    assert cmp.max_seed_range == 0.0


def test_zero_baseline_quarter_excluded() -> None:
    pen_b = [0.0] * 5 + [10.0] * 5
    pen_c = [5.0] * 5 + [9.0] * 5
    cand = {"A": [_daily("2024-06-26", 10, pen_c)], "B": [_daily("2024-06-26", 10, pen_c)]}
    base = {"A": [_daily("2024-06-26", 10, pen_b)], "B": [_daily("2024-06-26", 10, pen_b)]}
    cmp = compare_runs(cand, base, _cfg())
    assert set(cmp.folds[0].quarter_rel) == {"2024Q3"}
    assert any(r.startswith("quarter_excluded=") for r in cmp.reasons)


def test_bootstrap_zero_replicate_rejected() -> None:
    folds = (FoldSpec(name="A", test_start=pd.Timestamp("2024-01-01", tz=KST), test_end=pd.Timestamp("2024-01-31", tz=KST)),)
    cfg = ValidationConfig(folds=folds, bootstrap_block_days=1, bootstrap_n=100, bootstrap_seed=0)
    pen_b = [7.0] + [0.0] * 7
    pen_c = [6.0] + [0.0] * 7
    cand = {"A": [_daily("2024-01-01", 8, pen_c)]}
    base = {"A": [_daily("2024-01-01", 8, pen_b)]}
    with pytest.raises(CompareError):
        compare_runs(cand, base, cfg)
