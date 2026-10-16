"""기준 배정 불변식 가드."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.oibc.compose.baselines import capacity_balanced_assignment, random_assignment
from src.oibc.compose.constraints import CompositionError, validate_assignment
from src.oibc.config import CompetitionSpec

KST = "Asia/Seoul"
SITES = ["PV001", "PV002", "PV003", "PV004"]


def _spec() -> CompetitionSpec:
    return CompetitionSpec(
        n_sites=4,
        min_group_capacity_kw=200.0,
        train_end=pd.Timestamp("2025-06-01 00:00", tz=KST),
        eval_start=pd.Timestamp("2025-06-01 01:00", tz=KST),
        eval_end=pd.Timestamp("2025-06-02 00:00", tz=KST),
        n_eval_hours=24,
    )


def _sites(cap: float = 100.0) -> pd.DataFrame:
    return pd.DataFrame({"site_id": SITES, "capacity_kw": [cap] * 4})


def test_random_assignment_valid_partition() -> None:
    spec = _spec()
    for seed in range(5):
        out = random_assignment(_sites(), np.random.default_rng(seed), spec=spec)
        assert list(out.columns) == ["site_id", "group"]
        validate_assignment(out, _sites(), spec=spec)


def test_random_assignment_reproducible_and_order_invariant() -> None:
    spec = _spec()
    first = random_assignment(_sites(), np.random.default_rng(7), spec=spec)
    second = random_assignment(_sites(), np.random.default_rng(7), spec=spec)
    pd.testing.assert_frame_equal(first, second)
    shuffled = _sites().sample(frac=1.0, random_state=3).reset_index(drop=True)
    third = random_assignment(shuffled, np.random.default_rng(7), spec=spec)
    pd.testing.assert_frame_equal(first, third)


def test_random_assignment_fails_closed_when_infeasible() -> None:
    with pytest.raises(CompositionError):
        random_assignment(
            _sites(cap=10.0), np.random.default_rng(0), spec=_spec(), max_tries=5
        )


def test_random_assignment_varies_across_seeds() -> None:
    seen = set()
    for seed in range(20):
        out = random_assignment(_sites(), np.random.default_rng(seed), spec=_spec())
        seen.add(tuple(out.sort_values("site_id")["group"].tolist()))
    assert len(seen) >= 2


def test_capacity_balanced_gap_bounded() -> None:
    from src.oibc.config import CompetitionSpec as _Spec

    ids = [f"PV{i:03d}" for i in range(1, 7)]
    caps = [50.0, 80.0, 120.0, 150.0, 100.0, 100.0]
    sites = pd.DataFrame({"site_id": ids, "capacity_kw": caps})
    spec = _Spec(
        n_sites=6,
        min_group_capacity_kw=200.0,
        train_end=pd.Timestamp("2025-06-01 00:00", tz=KST),
        eval_start=pd.Timestamp("2025-06-01 01:00", tz=KST),
        eval_end=pd.Timestamp("2025-06-02 00:00", tz=KST),
        n_eval_hours=24,
    )
    out = capacity_balanced_assignment(sites, np.random.default_rng(0), spec=spec)
    validate_assignment(out, sites, spec=spec)
    got = out.merge(sites, on="site_id").groupby("group")["capacity_kw"].sum()
    assert abs(float(got.iloc[0] - got.iloc[1])) <= 150.0


def test_capacity_balanced_reproducible() -> None:
    first = capacity_balanced_assignment(_sites(), np.random.default_rng(11), spec=_spec())
    second = capacity_balanced_assignment(_sites(), np.random.default_rng(11), spec=_spec())
    pd.testing.assert_frame_equal(first, second)


def test_baselines_reject_bad_sites() -> None:
    duplicate_columns = pd.concat([_sites(), _sites()[["capacity_kw"]]], axis=1)
    with pytest.raises(CompositionError):
        random_assignment(duplicate_columns, np.random.default_rng(0), spec=_spec())
    with pytest.raises(CompositionError):
        random_assignment(_sites().drop(columns=["capacity_kw"]), np.random.default_rng(0), spec=_spec())
    null_sites = _sites()
    null_sites.loc[0, "site_id"] = pd.NA
    with pytest.raises(CompositionError):
        random_assignment(null_sites, np.random.default_rng(0), spec=_spec())
    dup_sites = pd.concat([_sites(), _sites().iloc[:1]], ignore_index=True)
    with pytest.raises(CompositionError):
        capacity_balanced_assignment(dup_sites, np.random.default_rng(0), spec=_spec())
    bad_cap = _sites()
    bad_cap.loc[0, "capacity_kw"] = -5.0
    with pytest.raises(CompositionError):
        random_assignment(bad_cap, np.random.default_rng(0), spec=_spec())
    with pytest.raises(CompositionError):
        random_assignment(_sites(), np.random.default_rng(0), spec=_spec(), max_tries=0)
    with pytest.raises(CompositionError):
        capacity_balanced_assignment(_sites(cap=10.0), np.random.default_rng(0), spec=_spec())


def test_capacity_balanced_order_invariant() -> None:
    first = capacity_balanced_assignment(_sites(), np.random.default_rng(11), spec=_spec())
    second = capacity_balanced_assignment(_sites().iloc[::-1], np.random.default_rng(11), spec=_spec())
    pd.testing.assert_frame_equal(first, second)
