import pandas as pd
import pytest

from src.oibc.compose.constraints import CompositionError, validate_assignment
from src.oibc.config import CompetitionSpec


def _sites() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "site_id": [f"S{i:03d}" for i in range(1, 7)],
            "capacity_kw": [500.0, 500.0, 500.0, 500.0, 500.0, 500.0],
        }
    )


def _spec(**kwargs: object) -> CompetitionSpec:
    base: dict[str, object] = {"n_sites": 6, "min_group_capacity_kw": 1000.0}
    base.update(kwargs)
    return CompetitionSpec(**base)  # type: ignore[arg-type]


def test_validate_assignment_accepts_valid_partition() -> None:
    sites = _sites()
    assignment = pd.DataFrame(
        {"site_id": sites["site_id"].tolist(), "group": ["G1", "G1", "G1", "G2", "G2", "G2"]}
    )
    spec = _spec()
    caps = validate_assignment(assignment, sites, spec=spec)
    assert list(caps.keys()) == list(spec.groups)
    assert caps["G1"] == pytest.approx(1500.0)
    assert caps["G2"] == pytest.approx(1500.0)
    assert sum(caps.values()) == pytest.approx(3000.0)


def test_validate_assignment_rejects_missing_site() -> None:
    sites = _sites()
    assignment = pd.DataFrame(
        {"site_id": sites["site_id"].tolist()[:5], "group": ["G1", "G1", "G1", "G2", "G2"]}
    )
    with pytest.raises(CompositionError):
        validate_assignment(assignment, sites, spec=_spec())


def test_validate_assignment_rejects_extra_site() -> None:
    sites = _sites()
    assignment = pd.DataFrame(
        {"site_id": [*sites["site_id"].tolist()[:5], "S999"], "group": ["G1", "G1", "G1", "G2", "G2", "G2"]}
    )
    with pytest.raises(CompositionError):
        validate_assignment(assignment, sites, spec=_spec())


def test_validate_assignment_rejects_duplicate_site() -> None:
    sites = _sites()
    ids = sites["site_id"].tolist()
    ids[5] = ids[0]
    assignment = pd.DataFrame({"site_id": ids, "group": ["G1", "G1", "G1", "G2", "G2", "G2"]})
    with pytest.raises(CompositionError):
        validate_assignment(assignment, sites, spec=_spec())


def test_validate_assignment_rejects_unknown_group() -> None:
    sites = _sites()
    assignment = pd.DataFrame(
        {"site_id": sites["site_id"].tolist(), "group": ["G1", "G1", "G1", "G2", "G2", "G3"]}
    )
    with pytest.raises(CompositionError):
        validate_assignment(assignment, sites, spec=_spec())


def test_validate_assignment_rejects_empty_group() -> None:
    sites = _sites()
    assignment = pd.DataFrame({"site_id": sites["site_id"].tolist(), "group": ["G1"] * 6})
    with pytest.raises(CompositionError):
        validate_assignment(assignment, sites, spec=_spec())


def test_validate_assignment_rejects_capacity_below_min() -> None:
    sites = pd.DataFrame(
        {
            "site_id": [f"S{i:03d}" for i in range(1, 7)],
            "capacity_kw": [1000.5, 200.0, 200.0, 200.0, 200.0, 200.0],
        }
    )
    assignment = pd.DataFrame(
        {"site_id": sites["site_id"].tolist(), "group": ["G1", "G2", "G2", "G2", "G2", "G2"]}
    )
    with pytest.raises(CompositionError, match="G2"):
        validate_assignment(assignment, sites, spec=_spec(min_group_capacity_kw=1000.001))


def test_validate_assignment_accepts_capacity_at_min() -> None:
    sites = pd.DataFrame(
        {
            "site_id": [f"S{i:03d}" for i in range(1, 7)],
            "capacity_kw": [1000.0, 200.0, 200.0, 200.0, 200.0, 200.0],
        }
    )
    assignment = pd.DataFrame(
        {"site_id": sites["site_id"].tolist(), "group": ["G1", "G2", "G2", "G2", "G2", "G2"]}
    )
    caps = validate_assignment(assignment, sites, spec=_spec(min_group_capacity_kw=1000.0))
    assert caps["G1"] == pytest.approx(1000.0)
    assert caps["G2"] == pytest.approx(1000.0)


def test_validate_assignment_rejects_bad_columns() -> None:
    sites = _sites()
    swapped = pd.DataFrame(
        {"group": ["G1", "G1", "G1", "G2", "G2", "G2"], "site_id": sites["site_id"].tolist()}
    )
    with pytest.raises(CompositionError):
        validate_assignment(swapped, sites, spec=_spec())
    extra = pd.DataFrame(
        {
            "site_id": sites["site_id"].tolist(),
            "group": ["G1", "G1", "G1", "G2", "G2", "G2"],
            "extra": [0] * 6,
        }
    )
    with pytest.raises(CompositionError):
        validate_assignment(extra, sites, spec=_spec())


def test_validate_assignment_does_not_mutate_input() -> None:
    sites = _sites()
    assignment = pd.DataFrame(
        {"site_id": sites["site_id"].tolist(), "group": ["G1", "G1", "G1", "G2", "G2", "G2"]}
    )
    before_assignment = assignment.copy(deep=True)
    before_sites = sites.copy(deep=True)
    validate_assignment(assignment, sites, spec=_spec())
    pd.testing.assert_frame_equal(assignment, before_assignment)
    pd.testing.assert_frame_equal(sites, before_sites)


def test_validate_assignment_rejects_sites_without_capacity() -> None:
    sites = _sites().drop(columns=["capacity_kw"])
    assignment = pd.DataFrame(
        {"site_id": [f"S{i:03d}" for i in range(1, 7)], "group": ["G1", "G1", "G1", "G2", "G2", "G2"]}
    )
    with pytest.raises(CompositionError):
        validate_assignment(assignment, sites, spec=_spec())


def test_validate_assignment_rejects_sites_count_mismatch() -> None:
    sites = _sites().iloc[:5].reset_index(drop=True)
    assignment = pd.DataFrame(
        {"site_id": sites["site_id"].tolist(), "group": ["G1", "G1", "G1", "G2", "G2"]}
    )
    with pytest.raises(CompositionError):
        validate_assignment(assignment, sites, spec=_spec())


def test_validate_assignment_rejects_null_site_id() -> None:
    sites = _sites()
    sites.loc[0, "site_id"] = None
    assignment = pd.DataFrame(
        {"site_id": [f"S{i:03d}" for i in range(1, 7)], "group": ["G1", "G1", "G1", "G2", "G2", "G2"]}
    )
    with pytest.raises(CompositionError):
        validate_assignment(assignment, sites, spec=_spec())


def test_validate_assignment_rejects_duplicate_site_in_sites() -> None:
    sites = _sites()
    sites.loc[5, "site_id"] = sites.loc[0, "site_id"]
    assignment = pd.DataFrame(
        {"site_id": [f"S{i:03d}" for i in range(1, 7)], "group": ["G1", "G1", "G1", "G2", "G2", "G2"]}
    )
    with pytest.raises(CompositionError):
        validate_assignment(assignment, sites, spec=_spec())


@pytest.mark.parametrize("bad", [0.0, -10.0, float("nan"), float("inf")])
def test_validate_assignment_rejects_bad_capacity(bad: float) -> None:
    sites = _sites()
    sites.loc[0, "capacity_kw"] = bad
    assignment = pd.DataFrame(
        {"site_id": [f"S{i:03d}" for i in range(1, 7)], "group": ["G1", "G1", "G1", "G2", "G2", "G2"]}
    )
    with pytest.raises(CompositionError):
        validate_assignment(assignment, sites, spec=_spec())


def test_validate_assignment_rejects_null_assignment() -> None:
    sites = _sites()
    assignment = pd.DataFrame(
        {"site_id": sites["site_id"].tolist(), "group": ["G1", "G1", "G1", "G2", "G2", None]}
    )
    with pytest.raises(CompositionError):
        validate_assignment(assignment, sites, spec=_spec())


def test_validate_assignment_rejects_string_id_collision() -> None:
    sites = _sites()
    sites["site_id"] = sites["site_id"].astype(object)
    sites.loc[0, "site_id"] = 1
    sites.loc[1, "site_id"] = "1"
    assignment = pd.DataFrame({"site_id": sites.site_id, "group": ["G1"] * 3 + ["G2"] * 3})
    with pytest.raises(CompositionError, match="duplicate"):
        validate_assignment(assignment, sites, spec=_spec())
    sites = _sites()
    assignment["site_id"] = assignment["site_id"].astype(object)
    assignment.loc[0, "site_id"] = 1
    with pytest.raises(CompositionError, match="duplicate"):
        validate_assignment(assignment, sites, spec=_spec())


def test_validate_assignment_rejects_duplicate_columns_and_overflow() -> None:
    sites = _sites()
    assignment = pd.DataFrame({"site_id": sites.site_id, "group": ["G1"] * 3 + ["G2"] * 3})
    with pytest.raises(CompositionError, match="duplicate columns"):
        validate_assignment(assignment, pd.concat([sites, sites[["capacity_kw"]]], axis=1), spec=_spec())
    sites["capacity_kw"] = 1e308
    with pytest.raises(CompositionError, match="non-finite"):
        validate_assignment(assignment, sites, spec=_spec())
