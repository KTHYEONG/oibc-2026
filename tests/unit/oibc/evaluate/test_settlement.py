import pandas as pd
import pytest

from src.oibc.config import CompetitionSpec
from src.oibc.evaluate.settlement import (
    SettlementError,
    SettlementResult,
    aggregate_groups,
    breakdown,
    daily_totals,
    settle,
    skill_score,
)
from src.oibc.timeaxis import k_weight

KST = "Asia/Seoul"
DT = "datetime64[us, Asia/Seoul]"


def _ts(vals: list[str]) -> pd.Series:
    return pd.Series(pd.DatetimeIndex(pd.to_datetime(vals).tz_localize(KST)), dtype=DT)


def _assign(rows: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "site_id": pd.Series([r[0] for r in rows], dtype="string"),
            "group": pd.Series([r[1] for r in rows], dtype="string"),
        }
    )


def _site(rows: list[tuple[str, str, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "site_id": pd.Series([r[0] for r in rows], dtype="string"),
            "ts": _ts([r[1] for r in rows]),
            "pred_kwh": pd.Series([r[2] for r in rows], dtype="float64"),
            "actual_kwh": pd.Series([r[3] for r in rows], dtype="float64"),
        }
    )


def _smp(rows: list[tuple[str, float]], spec: CompetitionSpec | None = None) -> pd.DataFrame:
    spec = spec or CompetitionSpec()
    ts = _ts([r[0] for r in rows])
    k = k_weight(ts, spec)
    return pd.DataFrame(
        {
            "ts": ts,
            "smp": pd.Series([r[1] for r in rows], dtype="float64"),
            "k": k.astype("int8"),
        }
    )


def _group(rows: list[tuple[str, str, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "group": pd.Series([r[0] for r in rows], dtype="string"),
            "ts": _ts([r[1] for r in rows]),
            "forecast_kwh": pd.Series([r[2] for r in rows], dtype="float64"),
            "actual_kwh": pd.Series([r[3] for r in rows], dtype="float64"),
        }
    )


def test_manual_settlement_matches_hand_calculation() -> None:
    gf = _group(
        [
            ("G1", "2025-06-01 10:00", 10.0, 8.0),
            ("G1", "2025-06-01 12:00", 5.0, 9.0),
            ("G1", "2025-06-01 15:00", 7.0, 7.0),
            ("G2", "2025-06-01 10:00", 0.0, 0.0),
            ("G2", "2025-06-01 12:00", 3.0, 1.0),
            ("G2", "2025-06-01 15:00", 4.0, 6.0),
        ]
    )
    smp = _smp([("2025-06-01 10:00", 100.0), ("2025-06-01 12:00", 200.0), ("2025-06-01 15:00", 0.0)])
    res = settle(gf, smp)
    assert res.revenue == pytest.approx(2800.0)
    assert res.penalty == pytest.approx(380.0)
    assert res.settlement == pytest.approx(2420.0)
    assert res.settlement == res.revenue - res.penalty
    zero = res.hourly[res.hourly["smp"] == 0.0]
    assert (zero["penalty"] == 0.0).all()
    assert str(res.hourly["ts"].dtype) == DT
    assert str(res.hourly["k"].dtype) == "int8"


def test_offsetting_errors_cancel_within_group() -> None:
    site = _site(
        [
            ("PV001", "2025-06-01 12:00", 10.0, 5.0),
            ("PV002", "2025-06-01 12:00", 5.0, 10.0),
        ]
    )
    same = aggregate_groups(site, _assign([("PV001", "G1"), ("PV002", "G1")]))
    res = settle(same, _smp([("2025-06-01 12:00", 100.0)]))
    assert res.penalty == pytest.approx(0.0)
    split = aggregate_groups(site, _assign([("PV001", "G1"), ("PV002", "G2")]))
    res2 = settle(split, _smp([("2025-06-01 12:00", 100.0)]))
    assert res2.penalty > 0


def test_aggregation_preserves_totals() -> None:
    site = _site(
        [
            ("PV001", "2025-06-01 10:00", 3.0, 4.0),
            ("PV002", "2025-06-01 10:00", 5.0, 1.0),
            ("PV003", "2025-06-01 11:00", 2.0, 2.0),
            ("PV004", "2025-06-01 11:00", 1.0, 0.0),
        ]
    )
    out = aggregate_groups(site, _assign([("PV001", "G1"), ("PV002", "G1"), ("PV003", "G2"), ("PV004", "G2")]))
    assert out["forecast_kwh"].sum() == pytest.approx(site["pred_kwh"].sum(), rel=1e-12)
    assert out["actual_kwh"].sum() == pytest.approx(site["actual_kwh"].sum(), rel=1e-12)
    assert int(out["n_sites"].sum()) == len(site)
    assert list(out.columns) == ["group", "ts", "forecast_kwh", "actual_kwh", "n_sites"]
    assert str(out["n_sites"].dtype) == "int32"


def test_negative_actual_kept_as_is() -> None:
    gf = _group([("G1", "2025-06-01 02:00", 0.0, -0.3)])
    res = settle(gf, _smp([("2025-06-01 02:00", 100.0)]))
    row = res.hourly.iloc[0]
    assert row["abs_err"] == pytest.approx(0.3)
    assert row["revenue"] == pytest.approx(-30.0)
    assert row["penalty"] == pytest.approx(3.0)


def test_k_boundary_hours() -> None:
    gf = _group(
        [
            ("G1", "2025-06-01 11:00", 1.0, 0.0),
            ("G1", "2025-06-01 15:00", 1.0, 0.0),
            ("G1", "2025-06-01 16:00", 1.0, 0.0),
            ("G1", "2025-06-02 00:00", 1.0, 0.0),
        ]
    )
    smp = _smp(
        [
            ("2025-06-01 11:00", 10.0),
            ("2025-06-01 15:00", 10.0),
            ("2025-06-01 16:00", 10.0),
            ("2025-06-02 00:00", 10.0),
        ]
    )
    res = settle(gf, smp, spec=CompetitionSpec())
    got = dict(zip(res.hourly["ts"].dt.hour.tolist(), res.hourly["k"].tolist(), strict=True))
    assert got[11] == 3
    assert got[15] == 3
    assert got[16] == 1
    assert got[0] == 1


def test_missing_smp_timestamp_rejected() -> None:
    gf = _group([("G1", "2025-06-01 10:00", 1.0, 1.0), ("G1", "2025-06-01 11:00", 1.0, 1.0)])
    smp = _smp([("2025-06-01 10:00", 100.0)])
    with pytest.raises(SettlementError):
        settle(gf, smp)


def test_mismatched_k_rejected() -> None:
    smp = _smp([("2025-06-01 12:00", 100.0)])
    smp.loc[0, "k"] = 1
    with pytest.raises(SettlementError):
        settle(_group([("G1", "2025-06-01 12:00", 1.0, 1.0)]), smp)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0])
def test_invalid_forecast_rejected(bad: float) -> None:
    with pytest.raises(SettlementError):
        settle(_group([("G1", "2025-06-01 10:00", bad, 1.0)]), _smp([("2025-06-01 10:00", 100.0)]))


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_invalid_actual_rejected(bad: float) -> None:
    with pytest.raises(SettlementError):
        settle(_group([("G1", "2025-06-01 10:00", 1.0, bad)]), _smp([("2025-06-01 10:00", 100.0)]))
    with pytest.raises(SettlementError):
        aggregate_groups(
            _site([("PV001", "2025-06-01 10:00", 1.0, bad)]), _assign([("PV001", "G1")])
        )


def test_unassigned_site_rejected() -> None:
    with pytest.raises(SettlementError):
        aggregate_groups(_site([("PV009", "2025-06-01 10:00", 1.0, 1.0)]), _assign([("PV001", "G1")]))


def test_duplicate_site_key_rejected() -> None:
    site = _site(
        [
            ("PV001", "2025-06-01 10:00", 1.0, 1.0),
            ("PV001", "2025-06-01 10:00", 2.0, 2.0),
        ]
    )
    with pytest.raises(SettlementError):
        aggregate_groups(site, _assign([("PV001", "G1")]))


def test_breakdown_preserves_totals_all_axes() -> None:
    gf = _group(
        [
            ("G1", "2025-06-30 10:00", 5.0, 4.0),
            ("G1", "2025-07-01 12:00", 2.0, 3.0),
            ("G2", "2025-07-01 12:00", 1.0, 1.0),
            ("G2", "2025-10-01 12:00", 4.0, 0.0),
        ]
    )
    smp = _smp(
        [
            ("2025-06-30 10:00", 100.0),
            ("2025-07-01 12:00", 200.0),
            ("2025-10-01 12:00", 50.0),
        ]
    )
    res = settle(gf, smp)
    assert isinstance(res, SettlementResult)
    for by in ("group", "hour", "op_date", "month", "quarter"):
        part = breakdown(res, by)  # type: ignore[arg-type]
        assert part["penalty"].sum() == pytest.approx(res.penalty, rel=1e-12)
        assert part["revenue"].sum() == pytest.approx(res.revenue, rel=1e-12)


def test_midnight_belongs_to_previous_day() -> None:
    gf = _group([("G1", "2024-07-01 00:00", 1.0, 2.0)])
    res = settle(gf, _smp([("2024-07-01 00:00", 100.0)]))
    month = breakdown(res, "month")
    assert month["month"].tolist() == ["2024-06"]
    daily = daily_totals(res)
    assert daily.index[0] == pd.Timestamp("2024-06-30 00:00", tz=KST)


def test_skill_score_value_and_invalid_reference() -> None:
    assert skill_score(20.0, 80.0) == pytest.approx(75.0)
    with pytest.raises(SettlementError):
        skill_score(20.0, 0.0)
    with pytest.raises(SettlementError):
        skill_score(20.0, float("inf"))


def test_inputs_not_mutated() -> None:
    site = _site([("PV001", "2025-06-01 10:00", 3.0, 2.0)])
    assign = _assign([("PV001", "G1")])
    before_site = site.copy(deep=True)
    before_assign = assign.copy(deep=True)
    grouped = aggregate_groups(site, assign)
    before_grouped = grouped.copy(deep=True)
    settle(grouped, _smp([("2025-06-01 10:00", 100.0)]))
    pd.testing.assert_frame_equal(site, before_site)
    pd.testing.assert_frame_equal(assign, before_assign)
    pd.testing.assert_frame_equal(grouped, before_grouped)


def test_contract_violations_rejected() -> None:
    site = _site([("PV001", "2025-06-01 10:00", 1.0, 1.0)])
    with pytest.raises(SettlementError):
        aggregate_groups(site.drop(columns=["pred_kwh"]), _assign([("PV001", "G1")]))
    with pytest.raises(SettlementError):
        aggregate_groups(site, _assign([("PV001", "G1"), ("PV001", "G2")]))
    with pytest.raises(SettlementError):
        aggregate_groups(site, _assign([("PV001", "GX")]))
    bad_ts = site.copy()
    bad_ts["ts"] = pd.Series(pd.to_datetime(["2025-06-01 10:30"]).tz_localize(KST))
    with pytest.raises(SettlementError):
        aggregate_groups(bad_ts, _assign([("PV001", "G1")]))
    gf = _group([("G1", "2025-06-01 10:00", 1.0, 1.0)])
    with pytest.raises(SettlementError):
        settle(gf.iloc[:0], _smp([("2025-06-01 10:00", 100.0)]))
    with pytest.raises(SettlementError):
        settle(_group([("GX", "2025-06-01 10:00", 1.0, 1.0)]), _smp([("2025-06-01 10:00", 100.0)]))
    dup = pd.concat([gf, gf], ignore_index=True)
    with pytest.raises(SettlementError):
        settle(dup, _smp([("2025-06-01 10:00", 100.0)]))
    smp = _smp([("2025-06-01 10:00", 100.0), ("2025-06-01 11:00", 100.0)])
    dup_smp = pd.concat([smp, smp.iloc[[0]]], ignore_index=True)
    with pytest.raises(SettlementError):
        settle(gf, dup_smp)
    with pytest.raises(SettlementError):
        settle(gf, _smp([("2025-06-01 10:00", -5.0)]))
    with pytest.raises(SettlementError):
        breakdown(settle(gf, _smp([("2025-06-01 10:00", 100.0)])), "year")  # type: ignore[arg-type]


def test_daily_totals_format() -> None:
    gf = _group(
        [
            ("G1", "2025-06-01 10:00", 2.0, 1.0),
            ("G2", "2025-06-02 10:00", 2.0, 1.0),
        ]
    )
    smp = _smp([("2025-06-01 10:00", 100.0), ("2025-06-02 10:00", 100.0)])
    res = settle(gf, smp)
    daily = daily_totals(res)
    assert str(daily.index.dtype) == DT
    assert list(daily.columns) == ["penalty", "revenue"]
    assert daily["penalty"].sum() == pytest.approx(res.penalty)
    assert daily["revenue"].sum() == pytest.approx(res.revenue)


def test_fractional_k_rejected() -> None:
    smp = _smp([("2025-06-01 12:00", 100.0)])
    smp["k"] = smp["k"].astype("float64")
    smp.loc[0, "k"] = 3.5
    with pytest.raises(SettlementError):
        settle(_group([("G1", "2025-06-01 12:00", 1.0, 1.0)]), smp)


def test_null_site_id_rejected() -> None:
    site = _site([("PV001", "2025-06-01 10:00", 1.0, 1.0)])
    site.loc[0, "site_id"] = pd.NA
    with pytest.raises(SettlementError):
        aggregate_groups(site, _assign([("PV001", "G1")]))
