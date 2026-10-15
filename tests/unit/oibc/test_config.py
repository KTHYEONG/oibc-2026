import pandas as pd
import pytest
from pathlib import Path
from pydantic import ValidationError

from src.oibc.config import CompetitionSpec, DataPaths, repo_root


def test_default_spec_self_consistent() -> None:
    s = CompetitionSpec()
    assert (s.eval_end - s.eval_start) == pd.Timedelta(hours=8759)
    assert s.train_end + pd.Timedelta(hours=1) == s.eval_start
    for v in (s.train_end, s.eval_start, s.eval_end):
        assert str(v.tzinfo) == "Asia/Seoul"


def test_inconsistent_eval_window_rejected() -> None:
    s = CompetitionSpec()
    with pytest.raises(ValidationError, match="eval window"):
        CompetitionSpec(eval_end=s.eval_end - pd.Timedelta(hours=1))


def test_spec_immutable() -> None:
    s = CompetitionSpec()
    with pytest.raises(ValidationError, match=r"[Ff]rozen"):
        s.penalty_rate = 0.5  # type: ignore[misc]


def test_paths_under_repo_root() -> None:
    p = DataPaths()
    root = repo_root()
    for d in (p.raw_dir, p.canonical_dir, p.derived_dir):
        assert root in d.parents or d == root


def test_naive_timestamp_rejected() -> None:
    with pytest.raises(ValidationError, match="tz-aware"):
        CompetitionSpec(train_end="2025-06-01 00:00")  # type: ignore[arg-type]


def test_train_eval_boundary_rejected() -> None:
    s = CompetitionSpec()
    with pytest.raises(ValidationError, match="train_end"):
        CompetitionSpec(train_end=s.train_end - pd.Timedelta(hours=1))


def test_paths_allow_explicit_overrides(tmp_path: Path) -> None:
    assert DataPaths(raw_dir=tmp_path).raw_dir == tmp_path


@pytest.mark.parametrize("value", [pd.NaT, pd.Timestamp("2025-06-01", tz="UTC")])
def test_invalid_timestamp_rejected(value: pd.Timestamp) -> None:
    with pytest.raises(ValidationError, match="Asia/Seoul"):
        CompetitionSpec(train_end=value)


def test_subhour_boundary_rejected() -> None:
    s = CompetitionSpec()
    with pytest.raises(ValidationError, match="hourly"):
        CompetitionSpec(train_end=s.train_end + pd.Timedelta(nanoseconds=1))


def test_rounds_cannot_exceed_cutoff() -> None:
    with pytest.raises(ValidationError, match="rounds"):
        CompetitionSpec(rounds=(10, 18))


def test_peak_hours_are_clock_hours() -> None:
    with pytest.raises(ValidationError, match="k_peak_hours"):
        CompetitionSpec(k_peak_hours=(24,))
