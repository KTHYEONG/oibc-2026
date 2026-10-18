import pandas as pd
import pytest
from pathlib import Path
from pydantic import ValidationError

from src.oibc.config import ArtifactPaths, CompetitionSpec, ComposeConfig, DataPaths, FeatureConfig, FoldSpec, ModelConfig, RegionalConfig, RunConfig, ValidationConfig, repo_root


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


def test_model_config_defaults_valid() -> None:
    cfg = ModelConfig()
    assert cfg.seeds == (0, 1, 2)
    assert cfg.weight == "smp_k_cap"


def test_model_config_duplicate_seeds_rejected() -> None:
    with pytest.raises(ValueError, match="seeds"):
        ModelConfig(seeds=(0, 0))


@pytest.mark.parametrize("kwargs", [
    {"learning_rate": 0.0},
    {"learning_rate": 1.5},
    {"num_leaves": 1},
    {"n_rounds": 0},
    {"n_jobs": 0},
    {"subsample": 0.0},
    {"colsample_bytree": 0.0},
    {"seeds": ()},
    {"target_clip_max": 0.9},
    {"min_weight": 0.0},
])
def test_model_config_bounds_rejected(kwargs: dict) -> None:
    with pytest.raises(ValueError, match=r"must"):
        ModelConfig(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("field", ["learning_rate", "subsample", "colsample_bytree", "reg_lambda", "target_clip_max", "min_weight"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_model_config_nonfinite_rejected(field: str, value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        ModelConfig(**{field: value})


@pytest.mark.parametrize("kwargs", [{"min_child_samples": 0}, {"subsample_freq": -1}, {"reg_lambda": -1.0}])
def test_model_config_invalid_lightgbm_parameters(kwargs: dict) -> None:
    with pytest.raises(ValueError, match="must"):
        ModelConfig(**kwargs)


@pytest.mark.parametrize("kwargs", [
    {"noise_z": 0.0}, {"noise_z": -1.0}, {"noise_z": float("inf")},
    {"noise_z": float("nan")}, {"min_noise_df": 0},
    {"ci_level": 1e-300}, {"ci_level": float.fromhex("0x1.fffffffffffffp-1")},
])
def test_validation_noise_params_rejected(kwargs: dict) -> None:
    with pytest.raises(ValidationError):
        ValidationConfig(**kwargs)


def test_runconfig_commit_drift_default_denied() -> None:
    cfg = RunConfig(run_id="r1", hypothesis="h")
    assert cfg.allow_commit_drift is False


def test_compose_config_defaults_preregistered() -> None:
    cfg = ComposeConfig()
    assert cfg.n_strata == 64
    assert cfg.seed == 0
    assert cfg.n_init == 10
    assert cfg.km_per_deg == 111.0
    assert cfg.ref_lat_deg == 36.5


@pytest.mark.parametrize("kwargs", [
    {"n_strata": 1},
    {"n_init": 0},
    {"km_per_deg": 0.0},
    {"km_per_deg": float("nan")},
    {"km_per_deg": float("inf")},
    {"ref_lat_deg": float("nan")},
    {"ref_lat_deg": 90.0},
    {"ref_lat_deg": -90.0},
])
def test_compose_config_rejects_invalid(kwargs: dict) -> None:
    with pytest.raises(ValidationError):
        ComposeConfig(**kwargs)


def test_artifact_paths_compositions_under_root() -> None:
    p = ArtifactPaths()
    assert p.compositions == p.root / "compositions"


def test_run_config_composition_default_reference() -> None:
    assert RunConfig(run_id="r1", hypothesis="h").composition_id == "current_group"


@pytest.mark.parametrize("value", ["../x", ""])
def test_run_config_composition_rejects_unsafe(value: str) -> None:
    with pytest.raises(ValidationError):
        RunConfig(run_id="r1", hypothesis="h", composition_id=value)


def test_run_config_legacy_record_loads() -> None:
    data = RunConfig(run_id="r1", hypothesis="h").model_dump(mode="json")
    data.pop("composition_id")
    assert RunConfig.model_validate(data).composition_id == "current_group"


def test_model_config_seed_workers_lower_bound() -> None:
    with pytest.raises(ValidationError, match="seed_workers"):
        ModelConfig(seed_workers=0)


def test_model_config_seed_workers_defaults() -> None:
    cfg = ModelConfig()
    assert cfg.seed_workers == 3
    assert cfg.n_jobs == 6


def test_regional_default_disabled() -> None:
    regional = ModelConfig().regional
    assert regional.weight == 0.0
    assert regional.units == ((2, 0), (4, 0), (4, 1), (8, 0), (8, 1), (8, 2), (12, 0), (16, 0), (16, 1))


@pytest.mark.parametrize("weight", [-0.1, 1.1, float("nan")])
def test_regional_weight_bounds_rejected(weight: float) -> None:
    with pytest.raises(ValidationError, match="weight"):
        RegionalConfig(weight=weight)


@pytest.mark.parametrize("units", [[(0, 0)], [(4, -1)], [(4, 0), (4, 0)]])
def test_regional_units_validation_rejected(units: list) -> None:
    with pytest.raises(ValidationError, match="units"):
        RegionalConfig(weight=0.5, units=units)  # type: ignore[arg-type]


def test_regional_empty_units_with_positive_weight_rejected() -> None:
    with pytest.raises(ValidationError, match="units"):
        RegionalConfig(weight=0.5, units=())


def test_regional_legacy_record_roundtrip() -> None:
    data = RunConfig(run_id="r1", hypothesis="h").model_dump(mode="json")
    data["model"].pop("regional")
    assert RunConfig.model_validate(data).model.regional == RegionalConfig()


@pytest.mark.parametrize("kwargs", [
    {"sd_vars": ("a", "a")},
    {"excluded_vars": ("lat", "lat")},
    {"ref_lat_deg": float("nan")},
    {"ref_lat_deg": 90.0},
    {"kmeans_n_init": 0},
    {"learning_rate": 0.0},
    {"learning_rate": 1.5},
    {"num_leaves": 1},
    {"min_child_samples": 0},
    {"subsample_freq": -1},
    {"reg_lambda": -1.0},
    {"subsample": 0.0},
    {"colsample_bytree": 0.0},
    {"n_rounds": 0},
])
def test_regional_bounds_rejected(kwargs: dict) -> None:
    with pytest.raises(ValidationError, match="must"):
        RegionalConfig(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("field", ["learning_rate", "subsample", "colsample_bytree", "reg_lambda"])
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_regional_nonfinite_rejected(field: str, value: float) -> None:
    with pytest.raises(ValidationError, match="finite"):
        RegionalConfig(**{field: value})


def test_calibration_defaults_contain_identity() -> None:
    from src.oibc.config import CalibrationConfig

    cfg = CalibrationConfig()
    n_a = round((cfg.a_max - cfg.a_min) / cfg.step) + 1
    n_b = round((cfg.b_max - cfg.b_min) / cfg.step) + 1
    assert (n_a, n_b) == (61, 81)
    a_vals = {round(cfg.a_min + i * cfg.step, 10) for i in range(n_a)}
    b_vals = {round(cfg.b_min + j * cfg.step, 10) for j in range(n_b)}
    assert 1.0 in a_vals
    assert 0.0 in b_vals


@pytest.mark.parametrize(
    "kwargs",
    [
        {"a_min": 1.01},
        {"b_max": -0.01},
        {"step": 0.0},
        {"step": 0.003},
        {"cloud_vars": ()},
        {"cloud_vars": ("wa_cloud",)},
        {"a_min": float("nan")},
        {"step": float("inf")},
        {"a_min": 1.05, "a_max": 0.90},
        {"b_min": 0.15, "b_max": -0.05},
        {"step": 0.0001},
        {"a_min": 0.91, "a_max": 1.03, "step": 0.04},
    ],
)
def test_calibration_bounds_rejected(kwargs: dict) -> None:
    from src.oibc.config import CalibrationConfig

    with pytest.raises(ValueError, match=r"."):
        CalibrationConfig(**kwargs)


def test_family_config_defaults_inactive() -> None:
    from src.oibc.config import FamilyConfig

    cfg = FamilyConfig()
    assert cfg.cat_weight == cfg.mlp_weight == 0.0
    assert cfg.mlp_units == ((4, 0), (8, 0), (16, 0))
    assert len(cfg.mlp_seeds) == 5
    assert cfg.cat_units == ((8, 0),)


def test_family_weights_sum_below_one() -> None:
    from src.oibc.config import FamilyConfig

    with pytest.raises(ValueError, match=r"1\.0"):
        FamilyConfig(cat_weight=0.5, mlp_weight=0.5)
    FamilyConfig(cat_weight=0.2, mlp_weight=0.2)


@pytest.mark.parametrize("kwargs", [
    {"cat_weight": -0.1},
    {"cat_weight": float("nan")},
    {"cat_weight": 1.5},
])
def test_family_weight_bounds(kwargs: dict) -> None:
    from src.oibc.config import FamilyConfig

    with pytest.raises(ValueError, match="weight"):
        FamilyConfig(**kwargs)


def test_active_family_requires_units() -> None:
    from src.oibc.config import FamilyConfig

    with pytest.raises(ValueError, match="units"):
        FamilyConfig(mlp_weight=0.2, mlp_units=())
    FamilyConfig(mlp_weight=0.0, mlp_units=())


def test_family_duplicate_units_rejected() -> None:
    from src.oibc.config import FamilyConfig

    with pytest.raises(ValueError, match="duplicates"):
        FamilyConfig(cat_units=((8, 0), (8, 0)))


@pytest.mark.parametrize("kwargs", [
    {"mlp_validation_fraction": 1.0},
    {"mlp_hidden": ()},
    {"mlp_hidden": (0,)},
])
def test_family_mlp_hyperparameter_bounds(kwargs: dict) -> None:
    from src.oibc.config import FamilyConfig

    with pytest.raises(ValueError, match=r"must"):
        FamilyConfig(**kwargs)  # type: ignore[arg-type]


def test_model_config_carries_families() -> None:
    assert ModelConfig().model_dump(mode="json")["families"]["cat_weight"] == 0.0


@pytest.mark.parametrize("kwargs", [
    {"cat_units": ((0, 0),)},
    {"cat_units": ((8, -1),)},
    {"mlp_units": ((0, 1),)},
])
def test_family_bad_unit_entries_rejected(kwargs: dict) -> None:
    from src.oibc.config import FamilyConfig

    with pytest.raises(ValueError, match="n_units"):
        FamilyConfig(**kwargs)


@pytest.mark.parametrize("kwargs", [
    {"cat_seeds": (0, 0)},
    {"mlp_seeds": (2, 2)},
])
def test_family_duplicate_seeds_rejected(kwargs: dict) -> None:
    from src.oibc.config import FamilyConfig

    with pytest.raises(ValueError, match="seeds"):
        FamilyConfig(**kwargs)


def test_family_negative_seeds_rejected() -> None:
    from src.oibc.config import FamilyConfig

    with pytest.raises(ValueError, match="non-negative"):
        FamilyConfig(cat_seeds=(-1,))


@pytest.mark.parametrize("kwargs", [
    {"cat_weight": 0.3, "cat_seeds": ()},
    {"mlp_weight": 0.3, "mlp_seeds": ()},
])
def test_family_active_member_requires_seeds(kwargs: dict) -> None:
    from src.oibc.config import FamilyConfig

    with pytest.raises(ValueError, match="seeds"):
        FamilyConfig(**kwargs)


@pytest.mark.parametrize("kwargs", [
    {"cat_iterations": 0},
    {"cat_depth": 0},
    {"cat_depth": 17},
    {"cat_learning_rate": 0.0},
    {"cat_learning_rate": 1.5},
    {"mlp_alpha": -0.5},
    {"mlp_batch_size": 0},
    {"mlp_learning_rate_init": 0.0},
    {"mlp_max_iter": 0},
    {"mlp_n_iter_no_change": 0},
])
def test_family_hyperparameter_bounds_rejected(kwargs: dict) -> None:
    from src.oibc.config import FamilyConfig

    with pytest.raises(ValueError, match=r"must"):
        FamilyConfig(**kwargs)  # type: ignore[arg-type]
