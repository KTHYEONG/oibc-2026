import pandas as pd
import pytest
from pathlib import Path
from pydantic import ValidationError

from src.oibc.config import CompetitionSpec, DataPaths, FeatureConfig, FoldSpec, ValidationConfig, repo_root


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


def test_default_validation_folds_chronological() -> None:
    cfg = ValidationConfig()
    assert [f.name for f in cfg.folds] == ["A", "B"]
    assert cfg.folds[0].test_start < cfg.folds[1].test_start
    assert cfg.folds[1].test_end == pd.Timestamp("2025-05-31 00:00", tz="Asia/Seoul")


def test_overlapping_folds_rejected() -> None:
    folds = (
        FoldSpec(
            name="A",
            test_start=pd.Timestamp("2023-06-01", tz="Asia/Seoul"),
            test_end=pd.Timestamp("2024-05-31", tz="Asia/Seoul"),
        ),
        FoldSpec(
            name="B",
            test_start=pd.Timestamp("2024-05-31", tz="Asia/Seoul"),
            test_end=pd.Timestamp("2025-05-31", tz="Asia/Seoul"),
        ),
    )
    with pytest.raises(ValidationError, match="overlap"):
        ValidationConfig(folds=folds)


def test_eval_period_fold_rejected() -> None:
    folds = (
        FoldSpec(
            name="A",
            test_start=pd.Timestamp("2024-06-01", tz="Asia/Seoul"),
            test_end=pd.Timestamp("2025-06-01", tz="Asia/Seoul"),
        ),
    )
    with pytest.raises(ValidationError, match="evaluation"):
        ValidationConfig(folds=folds)


def test_non_midnight_fold_boundary_rejected() -> None:
    with pytest.raises(ValidationError, match="midnight"):
        FoldSpec(
            name="B",
            test_start=pd.Timestamp("2024-06-01 01:00", tz="Asia/Seoul"),
            test_end=pd.Timestamp("2025-05-31", tz="Asia/Seoul"),
        )


def test_duplicate_seeds_rejected() -> None:
    with pytest.raises(ValidationError, match="seeds"):
        ValidationConfig(seeds=(0, 0))


def test_fold_name_required() -> None:
    with pytest.raises(ValidationError, match="non-empty"):
        FoldSpec(
            name="",
            test_start=pd.Timestamp("2024-06-01", tz="Asia/Seoul"),
            test_end=pd.Timestamp("2025-05-31", tz="Asia/Seoul"),
        )


def test_fold_end_before_start_rejected() -> None:
    with pytest.raises(ValidationError, match="test_end"):
        FoldSpec(
            name="C",
            test_start=pd.Timestamp("2025-05-31", tz="Asia/Seoul"),
            test_end=pd.Timestamp("2025-05-30", tz="Asia/Seoul"),
        )


def test_naive_fold_boundary_rejected() -> None:
    with pytest.raises(ValidationError, match="tz-aware"):
        FoldSpec(name="C", test_start="2024-06-01", test_end="2025-05-31")  # type: ignore[arg-type]


def test_empty_folds_rejected() -> None:
    with pytest.raises(ValidationError, match="at least one"):
        ValidationConfig(folds=())


def test_duplicate_fold_names_rejected() -> None:
    cfg = ValidationConfig()
    with pytest.raises(ValidationError, match="distinct"):
        ValidationConfig(folds=(cfg.folds[0], cfg.folds[0]))


def test_unordered_folds_rejected() -> None:
    cfg = ValidationConfig()
    with pytest.raises(ValidationError, match="ordered"):
        ValidationConfig(folds=(cfg.folds[1], cfg.folds[0]))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"embargo_days": -1},
        {"min_warm_days": 0},
        {"bootstrap_block_days": 0},
        {"bootstrap_n": 99},
        {"ci_level": 1.0},
        {"min_rel_improvement": 1.0},
        {"min_quarter_win_ratio": 0.0},
    ],
)
def test_validation_bounds_rejected(kwargs: object) -> None:
    with pytest.raises(ValidationError):
        ValidationConfig(**kwargs)  # type: ignore[arg-type]


def test_features_dir_under_repo_root(tmp_path: Path) -> None:
    p = DataPaths()
    assert p.features_dir == repo_root() / "data" / "features"
    assert DataPaths(features_dir=tmp_path).features_dir == tmp_path


def test_feature_config_defaults_valid() -> None:
    cfg = FeatureConfig()
    assert cfg.groups == {"cs", "rd", "day", "fleet"}
    assert cfg.day_round in CompetitionSpec().rounds


def test_feature_config_subset_violation_rejected() -> None:
    with pytest.raises(ValidationError, match="rd_vars"):
        FeatureConfig(rd_vars=("wa_cloud", "wa_unknown_var"))


def test_feature_config_even_window_rejected() -> None:
    with pytest.raises(ValidationError, match="day_roll_window"):
        FeatureConfig(day_roll_window=4)


def test_feature_config_zero_offset_rejected() -> None:
    with pytest.raises(ValidationError, match="neighbor_offsets"):
        FeatureConfig(neighbor_offsets=(0, 1))


def test_feature_config_prefix_violation_rejected() -> None:
    with pytest.raises(ValidationError, match="prefix"):
        FeatureConfig(wa_vars=("xx_cloud", "wa_temp"))


def test_feature_config_duplicate_vars_rejected() -> None:
    with pytest.raises(ValidationError, match="duplicates"):
        FeatureConfig(wa_vars=("wa_cloud", "wa_cloud"))


def test_feature_config_wb_subset_violation_rejected() -> None:
    with pytest.raises(ValidationError, match="neighbor_vars"):
        FeatureConfig(neighbor_vars=("wb_unknown_var",))


def test_feature_config_bare_subset_violation_rejected() -> None:
    with pytest.raises(ValidationError, match="neighbor_vars"):
        FeatureConfig(neighbor_vars=("cloud",))


def test_feature_config_vsfleet_violation_rejected() -> None:
    with pytest.raises(ValidationError, match="vsfleet_vars"):
        FeatureConfig(vsfleet_vars=("wa_rain",))


def test_feature_config_day_round_rejected() -> None:
    with pytest.raises(ValidationError, match="day_round"):
        FeatureConfig(day_round=9)


def test_feature_config_duplicate_offsets_rejected() -> None:
    with pytest.raises(ValidationError, match="duplicates"):
        FeatureConfig(day_shift_offsets=(-2, -2))


def test_feature_config_day_hours_rejected() -> None:
    with pytest.raises(ValidationError, match="day_hours"):
        FeatureConfig(day_hours=(16, 9))


@pytest.mark.parametrize("kwargs", [
    {"wa_vars": ("wb_cloud",)},
    {"wb_vars": ("wb_cloud",)},
    {"day_min_elev_deg": float("nan")},
    {"day_min_elev_deg": float("inf")},
])
def test_feature_config_unbuildable_inputs_rejected(kwargs: dict) -> None:
    with pytest.raises(ValidationError):
        FeatureConfig(**kwargs)
