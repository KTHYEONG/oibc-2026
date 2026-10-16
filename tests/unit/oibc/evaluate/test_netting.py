"""상쇄 분해 불변식 가드."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.oibc.config import CompetitionSpec
from src.oibc.evaluate.netting import NettingError, NettingEvaluator
from src.oibc.timeaxis import k_weight

KST = "Asia/Seoul"


def _smp(stamps: pd.DatetimeIndex, price: float = 100.0) -> pd.DataFrame:
    spec = CompetitionSpec()
    frame = pd.DataFrame({"ts": stamps, "smp": float(price)})
    frame["k"] = k_weight(frame["ts"], spec).to_numpy()
    return frame


def _oof(rows: list[tuple[str, pd.Timestamp, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "site_id": pd.Series([r[0] for r in rows], dtype="string"),
            "ts": pd.Series(pd.DatetimeIndex([r[1] for r in rows]), dtype="datetime64[us, Asia/Seoul]"),
            "actual_kwh": pd.Series([r[2] for r in rows], dtype="float64"),
            "pred_bag": pd.Series([r[3] for r in rows], dtype="float64"),
        }
    )


def _assign(pairs: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "site_id": pd.Series([p[0] for p in pairs], dtype="string"),
            "group": pd.Series([p[1] for p in pairs], dtype="string"),
        }
    )


def test_netting_hand_computed_opposite_errors() -> None:
    ts = pd.Timestamp("2023-01-10 12:00", tz=KST)
    smp = _smp(pd.DatetimeIndex([ts]))
    oof = _oof([("a", ts, 0.0, 10.0), ("b", ts, 0.0, -10.0)])
    ev = NettingEvaluator(oof, smp)
    same = ev.evaluate(_assign([("a", "G1"), ("b", "G1")]))
    split = ev.evaluate(_assign([("a", "G1"), ("b", "G2")]))
    assert same.penalty == pytest.approx(0.0)
    assert same.pool_penalty == pytest.approx(0.0)
    assert split.penalty == pytest.approx(2 * 0.10 * 100.0 * 3 * 10.0)
    assert split.pool_penalty == pytest.approx(0.0)
    assert split.netting_loss == pytest.approx(split.penalty)


def test_netting_same_sign_errors_no_loss() -> None:
    ts = pd.Timestamp("2023-01-10 12:00", tz=KST)
    smp = _smp(pd.DatetimeIndex([ts]))
    oof = _oof([("a", ts, 0.0, 5.0), ("b", ts, 0.0, 3.0)])
    ev = NettingEvaluator(oof, smp)
    res = ev.evaluate(_assign([("a", "G1"), ("b", "G2")]))
    assert res.netting_loss == pytest.approx(0.0)


def test_netting_matches_settle_pipeline() -> None:
    from src.oibc.evaluate.settlement import aggregate_groups, settle
    from src.oibc.train.runner import site_frame

    stamps = pd.DatetimeIndex(
        [pd.Timestamp(f"2023-01-10 {h:02d}:00", tz=KST) for h in (11, 12, 13)]
    )
    smp = _smp(stamps)
    oof = _oof(
        [
            ("a", stamps[0], 10.0, 12.0),
            ("a", stamps[1], 20.0, 17.0),
            ("b", stamps[0], 5.0, 9.0),
            ("b", stamps[2], 7.0, 6.0),
        ]
    )
    assignment = _assign([("a", "G1"), ("b", "G2")])
    ev = NettingEvaluator(oof, smp)
    got = ev.evaluate(assignment).penalty
    site = pd.DataFrame(
        {
            "site_id": oof["site_id"],
            "ts": oof["ts"],
            "pred_kwh": oof["pred_bag"].to_numpy(dtype="float64"),
            "actual_kwh": oof["actual_kwh"].to_numpy(dtype="float64"),
        }
    )
    site = site_frame(oof, oof["pred_bag"].to_numpy(), oof.iloc[:0])
    expected = settle(aggregate_groups(site, assignment), smp).penalty
    assert got == pytest.approx(expected, rel=1e-9)


def test_netting_unobserved_rows_contribute_zero() -> None:
    from src.oibc.evaluate.settlement import aggregate_groups, settle

    stamps = pd.DatetimeIndex(
        [pd.Timestamp(f"2023-01-10 {h:02d}:00", tz=KST) for h in (11, 12, 13)]
    )
    smp = _smp(stamps)
    oof = _oof(
        [
            ("a", stamps[0], 10.0, 12.0),
            ("a", stamps[1], 20.0, 17.0),
            ("b", stamps[0], 5.0, 9.0),
        ]
    )
    assignment = _assign([("a", "G1"), ("b", "G2")])
    ev = NettingEvaluator(oof, smp)
    site = pd.DataFrame(
        {
            "site_id": oof["site_id"],
            "ts": oof["ts"],
            "pred_kwh": oof["pred_bag"].to_numpy(dtype="float64"),
            "actual_kwh": oof["actual_kwh"].to_numpy(dtype="float64"),
        }
    )
    expected = settle(aggregate_groups(site, assignment), smp).penalty
    assert ev.evaluate(assignment).penalty == pytest.approx(expected, rel=1e-9)


def test_netting_penalties_matches_evaluate() -> None:
    stamps = pd.DatetimeIndex(
        [pd.Timestamp(f"2023-01-10 {h:02d}:00", tz=KST) for h in (11, 12, 13)]
    )
    smp = _smp(stamps)
    oof = _oof(
        [
            (s, t, float(i * 2), float(i * 2 + (1 if s == "a" else -1)))
            for i, t in enumerate(stamps)
            for s in ("a", "b", "c")
        ]
    )
    ev = NettingEvaluator(oof, smp)
    assigns = [
        _assign([("a", "G1"), ("b", "G1"), ("c", "G2")]),
        _assign([("a", "G1"), ("b", "G2"), ("c", "G2")]),
        _assign([("a", "G2"), ("b", "G2"), ("c", "G1")]),
        _assign([("a", "G1"), ("b", "G2"), ("c", "G1")]),
        _assign([("a", "G2"), ("b", "G1"), ("c", "G1")]),
    ]
    arr = ev.penalties(assigns)
    assert arr.dtype == np.float64
    for j, a in enumerate(assigns):
        assert arr[j] == pytest.approx(ev.evaluate(a).penalty)


def test_netting_peak_weight_applied() -> None:
    peak = pd.Timestamp("2023-01-10 12:00", tz=KST)
    off = pd.Timestamp("2023-01-10 18:00", tz=KST)
    oof_peak = _oof([("a", peak, 0.0, 10.0)])
    oof_off = _oof([("a", off, 0.0, 10.0)])
    smp = _smp(pd.DatetimeIndex([peak, off]))
    assign = _assign([("a", "G1")])
    p_peak = NettingEvaluator(oof_peak, smp).evaluate(assign).penalty
    p_off = NettingEvaluator(oof_off, smp).evaluate(assign).penalty
    assert p_peak == pytest.approx(3.0 * p_off)


def test_netting_zero_smp_costs_nothing() -> None:
    peak = pd.Timestamp("2023-01-10 12:00", tz=KST)
    free = pd.Timestamp("2023-01-10 13:00", tz=KST)
    smp = _smp(pd.DatetimeIndex([peak, free]))
    smp.loc[smp["ts"] == free, "smp"] = 0.0
    oof = _oof([("a", peak, 0.0, 5.0), ("a", free, 0.0, 1000.0)])
    ev = NettingEvaluator(oof, smp)
    assign = _assign([("a", "G1")])
    got = ev.evaluate(assign).penalty
    only = NettingEvaluator(_oof([("a", peak, 0.0, 5.0)]), smp).evaluate(assign).penalty
    assert got == pytest.approx(only)


def test_netting_rejects_ts_missing_in_smp() -> None:
    ts = pd.Timestamp("2023-01-10 12:00", tz=KST)
    other = pd.Timestamp("2023-01-10 13:00", tz=KST)
    with pytest.raises(NettingError):
        NettingEvaluator(_oof([("a", ts, 0.0, 1.0)]), _smp(pd.DatetimeIndex([other])))


def test_netting_rejects_uncovered_site() -> None:
    ts = pd.Timestamp("2023-01-10 12:00", tz=KST)
    ev = NettingEvaluator(
        _oof([("a", ts, 0.0, 1.0), ("b", ts, 0.0, 1.0)]), _smp(pd.DatetimeIndex([ts]))
    )
    with pytest.raises(NettingError):
        ev.evaluate(_assign([("a", "G1")]))


def test_netting_rejects_duplicate_key() -> None:
    ts = pd.Timestamp("2023-01-10 12:00", tz=KST)
    dup = _oof([("a", ts, 0.0, 1.0), ("a", ts, 0.0, 2.0)])
    with pytest.raises(NettingError):
        NettingEvaluator(dup, _smp(pd.DatetimeIndex([ts])))


def test_netting_rejects_k_mismatch() -> None:
    ts = pd.Timestamp("2023-01-10 12:00", tz=KST)
    smp = _smp(pd.DatetimeIndex([ts]))
    smp.loc[0, "k"] = 1
    with pytest.raises(NettingError):
        NettingEvaluator(_oof([("a", ts, 0.0, 1.0)]), smp)


def test_netting_constructor_guards() -> None:
    ts = pd.Timestamp("2023-01-10 12:00", tz=KST)
    good_oof = _oof([("a", ts, 1.0, 2.0)])
    good_smp = _smp(pd.DatetimeIndex([ts]))
    with pytest.raises(NettingError):
        NettingEvaluator(good_oof.drop(columns=["actual_kwh"]), good_smp)
    with pytest.raises(NettingError):
        NettingEvaluator(good_oof, good_smp.drop(columns=["k"]))
    null_oof = good_oof.copy()
    null_oof.loc[0, "site_id"] = pd.NA
    with pytest.raises(NettingError):
        NettingEvaluator(null_oof, good_smp)
    bad_hour = _oof([("a", pd.Timestamp("2023-01-10 12:30", tz=KST), 1.0, 2.0)])
    bad_smp = pd.DataFrame(
        {
            "ts": pd.DatetimeIndex([pd.Timestamp("2023-01-10 12:30", tz=KST)]),
            "smp": [100.0],
            "k": [1],
        }
    )
    with pytest.raises(NettingError):
        NettingEvaluator(bad_hour, bad_smp)
    valid_oof = _oof([( "a", ts, 1.0, 2.0)])
    with pytest.raises(NettingError):
        NettingEvaluator(valid_oof, bad_smp)
    dup_smp = pd.concat([good_smp, good_smp], ignore_index=True)
    with pytest.raises(NettingError):
        NettingEvaluator(good_oof, dup_smp)
    nan_oof = _oof([("a", ts, float("nan"), 2.0)])
    with pytest.raises(NettingError):
        NettingEvaluator(nan_oof, good_smp)
    inf_pred = _oof([("a", ts, 1.0, float("inf"))])
    with pytest.raises(NettingError):
        NettingEvaluator(inf_pred, good_smp)
    neg_smp = good_smp.copy()
    neg_smp.loc[0, "smp"] = -1.0
    with pytest.raises(NettingError):
        NettingEvaluator(good_oof, neg_smp)


def test_netting_assignment_guards_and_extras() -> None:
    ts = pd.Timestamp("2023-01-10 12:00", tz=KST)
    ev = NettingEvaluator(
        _oof([("a", ts, 0.0, 4.0), ("b", ts, 0.0, 2.0)]), _smp(pd.DatetimeIndex([ts]))
    )
    assert ev.site_ids == ("a", "b")
    assert ev.pool_penalty >= 0.0
    with pytest.raises(NettingError):
        ev.evaluate(_assign([("a", "G1"), ("b", "G1")]).rename(columns={"group": "grp"}))
    with pytest.raises(NettingError):
        ev.evaluate(_assign([("a", "G1"), ("a", "G2")]))
    with pytest.raises(NettingError):
        ev.evaluate(_assign([("a", "G1"), ("b", "GX")]))
    extra = ev.evaluate(_assign([("a", "G1"), ("b", "G2"), ("ghost", "G1")]))
    plain = ev.evaluate(_assign([("a", "G1"), ("b", "G2")]))
    assert extra.penalty == pytest.approx(plain.penalty)
    assert ev.penalties([]).shape == (0,)


def test_netting_rejects_numerical_anomaly() -> None:
    ts = pd.Timestamp("2023-01-10 12:00", tz=KST)
    ev = NettingEvaluator(_oof([("a", ts, 0.0, 4.0)]), _smp(pd.DatetimeIndex([ts])))
    ev._pool = 1e12
    with pytest.raises(NettingError):
        ev.evaluate(_assign([("a", "G1")]))


@pytest.mark.parametrize("defect", ["oof_columns", "smp_columns", "empty", "normalized_key"])
def test_netting_schema_fail_closed(defect: str) -> None:
    ts = pd.Timestamp("2023-01-10 12:00", tz=KST)
    oof = _oof([("a", ts, 1.0, 2.0)])
    smp = _smp(pd.DatetimeIndex([ts]))
    if defect == "oof_columns":
        oof = pd.concat([oof, oof[["pred_bag"]]], axis=1)
    elif defect == "smp_columns":
        smp = pd.concat([smp, smp[["k"]]], axis=1)
    elif defect == "empty":
        oof = oof.iloc[:0]
    else:
        oof = pd.concat([oof, oof], ignore_index=True)
        oof["site_id"] = pd.Series([1, "1"], dtype=object)
    with pytest.raises(NettingError):
        NettingEvaluator(oof, smp)


def test_netting_rejects_null_extra_site() -> None:
    ts = pd.Timestamp("2023-01-10 12:00", tz=KST)
    evaluator = NettingEvaluator(_oof([("a", ts, 0.0, 1.0)]), _smp(pd.DatetimeIndex([ts])))
    assignment = pd.DataFrame({"site_id": ["a", None], "group": ["G1", "G2"]})
    with pytest.raises(NettingError, match="null"):
        evaluator.penalties([assignment])


@pytest.mark.parametrize("defect", ["residual", "pool", "group"])
def test_netting_overflow_is_rejected(defect: str) -> None:
    ts = pd.Timestamp("2023-01-10 18:00", tz=KST)
    smp = _smp(pd.DatetimeIndex([ts]), price=1.0)
    if defect == "residual":
        oof = _oof([("a", ts, -1e308, 1e308)])
    elif defect == "pool":
        oof = _oof([("a", ts, 0.0, 1e308)])
        smp["smp"] = 1e308
    else:
        oof = _oof([("a", ts, 0.0, 1e308), ("b", ts, 0.0, -1e308)])
    with pytest.raises(NettingError):
        NettingEvaluator(oof, smp).penalties([_assign([("a", "G1"), ("b", "G2")])])


def test_netting_batch_matches_pipeline_with_missing_rows_and_shuffled_keys() -> None:
    from src.oibc.evaluate.settlement import aggregate_groups, settle

    rng = np.random.default_rng(31)
    stamps = pd.date_range("2023-01-10 10:00", periods=4, freq="h", tz=KST)
    rows = [(s, t, 20.0, float(rng.uniform(0, 40))) for s in ("a", "b", "c", "d") for t in stamps]
    oof = _oof(rows).drop(index=[1, 6, 12]).sample(frac=1, random_state=9)
    smp = _smp(stamps)
    evaluator = NettingEvaluator(oof, smp)
    assignments = [_assign([(s, str(rng.choice(["G1", "G2"]))) for s in ("a", "b", "c", "d")]) for _ in range(8)]
    site = oof.rename(columns={"pred_bag": "pred_kwh"})
    expected = [settle(aggregate_groups(site, a), smp).penalty for a in assignments]
    np.testing.assert_allclose(evaluator.penalties(assignments), expected, rtol=1e-9)
