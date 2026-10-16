import numpy as np
import pandas as pd
import pytest

from src.oibc.compose.anticluster import balanced_split, build_composition, spatial_strata
from src.oibc.compose.constraints import CompositionError, validate_assignment
from src.oibc.config import CompetitionSpec, ComposeConfig


def _regional_sites() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "site_id": [f"S{i:03d}" for i in range(1, 9)],
            "capacity_kw": [100.0] * 8,
            "lat": [36.0, 36.1, 36.2, 36.05, 35.9, 35.85, 36.0, 35.95],
            "lon": [126.0, 126.1, 126.2, 126.05, 129.0, 129.1, 129.2, 129.05],
        }
    )


def _cfg(**kwargs: object) -> ComposeConfig:
    base: dict[str, object] = {"n_strata": 2, "n_init": 2, "seed": 0}
    base.update(kwargs)
    return ComposeConfig(**base)  # type: ignore[arg-type]


def _spec(n: int, floor: float) -> CompetitionSpec:
    return CompetitionSpec(n_sites=n, min_group_capacity_kw=floor)


def test_spatial_strata_deterministic_and_order_invariant() -> None:
    sites = _regional_sites()
    cfg = _cfg()
    first = spatial_strata(sites, cfg)
    shuffled = sites.sample(frac=1.0, random_state=0).reset_index(drop=True)
    second = spatial_strata(shuffled, cfg)
    assert dict(zip(sites["site_id"], first.tolist(), strict=True)) == dict(
        zip(shuffled["site_id"], second.tolist(), strict=True)
    )


def test_spatial_strata_labels_canonical() -> None:
    sites = _regional_sites()
    labels = spatial_strata(sites, _cfg())
    assert set(labels.tolist()) == {0, 1}
    frame = pd.DataFrame({"site_id": sites["site_id"].astype("string"), "label": labels})
    mins = frame.groupby("label")["site_id"].min().sort_values()
    assert list(mins.index) == [0, 1]


def test_spatial_strata_rejects_nonfinite_coords() -> None:
    sites = _regional_sites()
    sites.loc[0, "lat"] = float("nan")
    with pytest.raises(CompositionError):
        spatial_strata(sites, _cfg())


def test_spatial_strata_rejects_too_few_sites() -> None:
    sites = _regional_sites().iloc[:2].reset_index(drop=True)
    with pytest.raises(CompositionError):
        spatial_strata(sites, _cfg(n_strata=4))


def test_balanced_split_splits_each_stratum() -> None:
    sites = pd.DataFrame(
        {
            "site_id": [f"S{i:03d}" for i in range(1, 9)],
            "capacity_kw": [100.0] * 4 + [200.0] * 4,
        }
    )
    strata = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype="int64")
    out = balanced_split(sites, strata, spec=_spec(8, 100.0))
    cap_of = dict(zip(sites["site_id"], sites["capacity_kw"], strict=True))
    for s in (0, 1):
        members = sites.loc[strata == s, "site_id"].tolist()
        groups = set(out.loc[out["site_id"].isin(members), "group"])
        assert groups == {"G1", "G2"}
        diff = abs(
            sum(cap_of[m] for m in members if out.loc[out["site_id"] == m, "group"].iloc[0] == "G1")
            - sum(cap_of[m] for m in members if out.loc[out["site_id"] == m, "group"].iloc[0] == "G2")
        )
        assert diff <= max(cap_of[m] for m in members)


def test_balanced_split_tiebreak_by_site_id() -> None:
    sites = pd.DataFrame(
        {"site_id": [f"S{i:03d}" for i in range(1, 5)], "capacity_kw": [100.0] * 4}
    )
    strata = np.zeros(4, dtype="int64")
    first = balanced_split(sites, strata, spec=_spec(4, 1.0))
    second = balanced_split(sites, strata, spec=_spec(4, 1.0))
    pd.testing.assert_frame_equal(first, second)
    got = dict(zip(first["site_id"], first["group"], strict=True))
    assert got == {"S001": "G1", "S002": "G2", "S003": "G1", "S004": "G2"}


def test_balanced_split_repairs_capacity_floor() -> None:
    caps = [495.0, 738.0, 421.0, 309.0, 222.0, 488.0, 988.0]
    sites = pd.DataFrame(
        {"site_id": [f"S{i:03d}" for i in range(1, 8)], "capacity_kw": caps}
    )
    strata = np.array([0, 1, 1, 0, 1, 0, 1], dtype="int64")
    greedy = balanced_split(sites, strata, spec=_spec(7, 1.0))
    cap_of = dict(zip(sites["site_id"], sites["capacity_kw"], strict=True))
    greedy_total = {
        g: sum(cap_of[s] for s in greedy.loc[greedy["group"] == g, "site_id"]) for g in ("G1", "G2")
    }
    assert min(greedy_total.values()) < 1655.0
    out = balanced_split(sites, strata, spec=_spec(7, 1655.0))
    total = {g: sum(cap_of[s] for s in out.loc[out["group"] == g, "site_id"]) for g in ("G1", "G2")}
    assert total["G1"] >= 1655.0
    assert total["G2"] >= 1655.0
    moved = [s for s in cap_of if greedy.loc[greedy["site_id"] == s, "group"].iloc[0] != out.loc[out["site_id"] == s, "group"].iloc[0]]
    assert moved == ["S005"]


def test_balanced_split_fails_closed_when_infeasible() -> None:
    sites = pd.DataFrame(
        {"site_id": [f"S{i:03d}" for i in range(1, 5)], "capacity_kw": [100.0] * 4}
    )
    strata = np.zeros(4, dtype="int64")
    with pytest.raises(CompositionError):
        balanced_split(sites, strata, spec=_spec(4, 250.0))


def test_balanced_split_rejects_bad_strata_length() -> None:
    sites = pd.DataFrame(
        {"site_id": [f"S{i:03d}" for i in range(1, 5)], "capacity_kw": [100.0] * 4}
    )
    with pytest.raises(CompositionError):
        balanced_split(sites, np.zeros(3, dtype="int64"), spec=_spec(4, 1.0))
    with pytest.raises(CompositionError):
        balanced_split(sites, np.zeros(4, dtype="float64"), spec=_spec(4, 1.0))


def test_build_composition_is_partition() -> None:
    sites = _regional_sites()
    spec = _spec(8, 100.0)
    out = build_composition(sites, _cfg(), spec=spec)
    validate_assignment(out, sites, spec=spec)
    assert out["site_id"].tolist() == sorted(out["site_id"].tolist())
    assert len(out) == 8
    assert list(out.columns) == ["site_id", "group"]


def test_build_composition_mixes_regions() -> None:
    sites = _regional_sites()
    out = build_composition(sites, _cfg(), spec=_spec(8, 100.0))
    west = {"S001", "S002", "S003", "S004"}
    for group in ("G1", "G2"):
        members = set(out.loc[out["group"] == group, "site_id"])
        assert members & west
        assert members - west


def test_build_composition_ignores_extra_columns() -> None:
    sites = _regional_sites()
    plain = build_composition(sites, _cfg(), spec=_spec(8, 100.0))
    decorated = sites.copy()
    decorated["current_group"] = ["A", "B", "A", "B", "A", "B", "A", "B"]
    decorated["wb_cloud"] = [10.0, 90.0, 20.0, 80.0, 30.0, 70.0, 40.0, 60.0]
    again = build_composition(decorated, _cfg(), spec=_spec(8, 100.0))
    pd.testing.assert_frame_equal(plain, again)


def test_balanced_split_rejects_bad_site_frame() -> None:
    good = pd.DataFrame(
        {"site_id": [f"S{i:03d}" for i in range(1, 5)], "capacity_kw": [100.0] * 4}
    )
    strata = np.zeros(4, dtype="int64")
    with pytest.raises(CompositionError):
        balanced_split(good.drop(columns=["capacity_kw"]), strata, spec=_spec(4, 1.0))
    null_ids = good.copy()
    null_ids.loc[0, "site_id"] = None
    with pytest.raises(CompositionError):
        balanced_split(null_ids, strata, spec=_spec(4, 1.0))
    dup_ids = good.copy()
    dup_ids.loc[3, "site_id"] = dup_ids.loc[0, "site_id"]
    with pytest.raises(CompositionError):
        balanced_split(dup_ids, strata, spec=_spec(4, 1.0))
    bad_cap = good.copy()
    bad_cap.loc[0, "capacity_kw"] = -5.0
    with pytest.raises(CompositionError):
        balanced_split(bad_cap, strata, spec=_spec(4, 1.0))


def test_spatial_strata_rejects_empty_stratum() -> None:
    from sklearn.exceptions import ConvergenceWarning

    sites = _regional_sites()
    sites["lat"] = 36.0
    sites["lon"] = 127.0
    with pytest.warns(ConvergenceWarning), pytest.raises(CompositionError, match="empty stratum"):
        spatial_strata(sites, _cfg())


def test_balanced_split_prefers_globally_smaller_group_on_local_tie() -> None:
    sites = pd.DataFrame(
        {
            "site_id": ["S001", "S002", "S003", "S004"],
            "capacity_kw": [100.0, 99.0, 99.0, 50.0],
        }
    )
    strata = np.array([0, 0, 0, 1], dtype="int64")
    out = balanced_split(sites, strata, spec=_spec(4, 1.0))
    got = dict(zip(out["site_id"], out["group"], strict=True))
    assert got["S004"] == "G1"


def test_balanced_split_fails_closed_on_empty_sites() -> None:
    sites = pd.DataFrame({"site_id": pd.Series(dtype="string"), "capacity_kw": pd.Series(dtype="float64")})
    with pytest.raises(CompositionError):
        balanced_split(sites, np.zeros(0, dtype="int64"), spec=_spec(0, 1000.0))


def test_balanced_split_preserves_order_invariance_with_fractional_stratum_totals() -> None:
    sites = pd.DataFrame({"site_id": list("abcdef"), "capacity_kw": [0.1, 0.2, 0.3] * 2})
    strata = np.array([0, 0, 0, 1, 1, 1], dtype="int64")
    expected = balanced_split(sites, strata, spec=_spec(6, 0.1))
    order = [2, 0, 1, 3, 4, 5]
    actual = balanced_split(sites.iloc[order], strata[order], spec=_spec(6, 0.1))
    pd.testing.assert_frame_equal(actual, expected)


def test_build_composition_byte_identical_after_shuffle() -> None:
    sites = _regional_sites()
    expected = build_composition(sites, _cfg(), spec=_spec(8, 100.0))
    actual = build_composition(sites.sample(frac=1, random_state=5), _cfg(), spec=_spec(8, 100.0))
    assert actual.to_csv(index=False) == expected.to_csv(index=False)


def test_spatial_strata_rejects_ambiguous_schema_and_overflow() -> None:
    sites = _regional_sites()
    with pytest.raises(CompositionError, match="duplicate columns"):
        spatial_strata(pd.concat([sites, sites[["lat"]]], axis=1), _cfg())
    sites["capacity_kw"] = 1e308
    with pytest.raises(CompositionError, match="total capacity"):
        spatial_strata(sites, _cfg())
    sites = _regional_sites()
    sites["lon"] = 1e308
    with pytest.raises(CompositionError, match="projected coordinates"):
        spatial_strata(sites, _cfg())
