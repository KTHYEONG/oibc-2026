"""대회 공식 상수 및 경로."""

from __future__ import annotations

import re
from itertools import pairwise
from math import isfinite
from pathlib import Path
from typing import Final, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def repo_root() -> Path:
    """저장소 루트 경로를 반환한다."""
    return Path(__file__).resolve().parents[2]


_KST = "Asia/Seoul"


class CompetitionSpec(BaseModel):
    """대회 공식 상수."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    timezone: Literal["Asia/Seoul"] = "Asia/Seoul"
    n_sites: int = 253
    min_group_capacity_kw: float = 20_000.0
    groups: tuple[str, str] = ("G1", "G2")
    rounds: tuple[int, int] = (10, 17)
    cutoff_hour: int = Field(default=17, ge=0, le=23)
    penalty_rate: float = 0.10
    k_peak_hours: tuple[int, ...] = (11, 12, 13, 14, 15)
    k_peak: int = Field(default=3, ge=0, le=127)
    k_base: int = Field(default=1, ge=0, le=127)
    train_end: pd.Timestamp = pd.Timestamp("2025-06-01 00:00", tz="Asia/Seoul")
    eval_start: pd.Timestamp = pd.Timestamp("2025-06-01 01:00", tz="Asia/Seoul")
    eval_end: pd.Timestamp = pd.Timestamp("2026-06-01 00:00", tz="Asia/Seoul")
    n_eval_hours: int = 8760

    @field_validator("train_end", "eval_start", "eval_end", mode="before")
    @classmethod
    def _coerce_ts(cls, v: object) -> pd.Timestamp:
        ts = v if isinstance(v, pd.Timestamp) else pd.Timestamp(v)
        if pd.isna(ts) or str(ts.tzinfo) != _KST:
            raise ValueError("tz-aware Asia/Seoul required")
        if ts != ts.floor("h"):
            raise ValueError("timestamp must be hourly on the hour")
        return ts

    @model_validator(mode="after")
    def _check_window(self) -> CompetitionSpec:
        if any(r < 0 or r > self.cutoff_hour for r in self.rounds):
            raise ValueError("rounds must be valid hours at or before cutoff")
        if any(h < 0 or h > 23 for h in self.k_peak_hours):
            raise ValueError("k_peak_hours must be valid hours")
        if self.train_end + pd.Timedelta(hours=1) != self.eval_start:
            raise ValueError("train_end + 1h must equal eval_start")
        span = self.eval_end - self.eval_start
        if span + pd.Timedelta(hours=1) != pd.Timedelta(hours=self.n_eval_hours):
            raise ValueError("eval window inconsistent with n_eval_hours")
        return self


class QualityConfig(BaseModel):
    """품질 플래그 임계값."""

    model_config = ConfigDict(frozen=True)

    over_cap_ratio: float = 1.05
    night_elev_deg: float = -5.0
    night_gen_ratio: float = 0.01
    zero_day_min_elev_deg: float = 10.0
    zero_day_ratio: float = 0.001
    observed_bounds: dict[str, tuple[float | None, float | None]] = {
        "wa_wind_speed": (None, 50.0),
        "wb_wind_speed": (None, 55.6),
        "wb_cloud": (None, 100.0),
        "wb_uv_idx": (None, 15.0),
        "wb_vis": (None, 50.0),
        "wb_pressure": (900.0, None),
    }


class DataPaths(BaseModel):
    """데이터 레이어 경로."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    raw_dir: Path = repo_root() / "data" / "raw" / "01_participant_dataset"
    canonical_dir: Path = repo_root() / "data" / "canonical"
    derived_dir: Path = repo_root() / "data" / "derived"
    features_dir: Path = repo_root() / "data" / "features"


class FeatureConfig(BaseModel):
    """특성 묶음·변수 목록·발행 내 변환 파라미터."""

    model_config = ConfigDict(frozen=True)

    groups: frozenset[Literal["cs", "rd", "day", "fleet"]] = frozenset({"cs", "rd", "day", "fleet"})
    day_min_elev_deg: float = 0.0
    wa_vars: tuple[str, ...] = ("wa_cloud", "wa_temp", "wa_humidity", "wa_rain", "wa_snow", "wa_wind_speed", "wa_ground_press")
    wb_vars: tuple[str, ...] = ("wb_cloud", "wb_temp", "wb_rel_hum", "wb_uv_idx", "wb_precip_prob", "wb_total_liq", "wb_vis", "wb_wind_speed", "wb_dew_point", "wb_ceiling", "wb_snow")
    neighbor_vars: tuple[str, ...] = ("wa_cloud", "wb_cloud", "wb_uv_idx")
    neighbor_offsets: tuple[int, ...] = (-1, 1)
    rd_vars: tuple[str, ...] = ("wa_cloud", "wb_cloud", "wb_uv_idx", "wb_precip_prob", "wa_rain")
    day_round: int = 17
    day_shift_offsets: tuple[int, ...] = (-2, 2)
    day_roll_window: int = 3
    day_hours: tuple[int, int] = (9, 16)
    day_mean_vars: tuple[str, ...] = ("wa_cloud", "wb_cloud", "wb_precip_prob")
    day_max_vars: tuple[str, ...] = ("wb_uv_idx",)
    day_sum_vars: tuple[str, ...] = ("wa_rain", "wb_total_liq")
    fleet_vars: tuple[str, ...] = ("wa_cloud", "wb_cloud", "wb_precip_prob", "wb_uv_idx")
    vsfleet_vars: tuple[str, ...] = ("wa_cloud", "wb_cloud", "wb_uv_idx")

    @model_validator(mode="after")
    def _check_features(self) -> FeatureConfig:
        pools = (*self.wa_vars, *self.wb_vars)
        for name in pools:
            if not (name.startswith("wa_") or name.startswith("wb_")):
                raise ValueError(f"variable without provider prefix: {name!r}")
        if not all(v.startswith("wa_") for v in self.wa_vars) or not all(v.startswith("wb_") for v in self.wb_vars):
            raise ValueError("wa_vars/wb_vars: provider prefix mismatch")
        wa = set(self.wa_vars)
        wb = set(self.wb_vars)
        if len(wa) != len(self.wa_vars) or len(wb) != len(self.wb_vars):
            raise ValueError("wa_vars/wb_vars must not contain duplicates")
        if "wa_cloud" not in wa or not {"wb_cloud", "wb_uv_idx"} <= wb:
            raise ValueError("wa_vars/wb_vars: clear-sky inputs required")
        if not isfinite(self.day_min_elev_deg):
            raise ValueError("day_min_elev_deg must be finite")
        for label, subset in (
            ("neighbor_vars", self.neighbor_vars),
            ("rd_vars", self.rd_vars),
            ("day_mean_vars", self.day_mean_vars),
            ("day_max_vars", self.day_max_vars),
            ("day_sum_vars", self.day_sum_vars),
            ("fleet_vars", self.fleet_vars),
        ):
            for name in subset:
                if name.startswith("wa_"):
                    if name not in wa:
                        raise ValueError(f"{label}: {name!r} not in wa_vars")
                elif name.startswith("wb_"):
                    if name not in wb:
                        raise ValueError(f"{label}: {name!r} not in wb_vars")
                else:
                    raise ValueError(f"{label}: {name!r} without provider prefix")
        for name in self.vsfleet_vars:
            if name not in set(self.fleet_vars):
                raise ValueError(f"vsfleet_vars: {name!r} not in fleet_vars")
        if self.day_round not in CompetitionSpec().rounds:
            raise ValueError("day_round must be a valid issuance round")
        for label, offsets in (("neighbor_offsets", self.neighbor_offsets), ("day_shift_offsets", self.day_shift_offsets)):
            if 0 in offsets:
                raise ValueError(f"{label} must not contain 0")
            if len(set(offsets)) != len(offsets):
                raise ValueError(f"{label} must not contain duplicates")
        if self.day_roll_window < 3 or self.day_roll_window % 2 == 0:
            raise ValueError("day_roll_window must be an odd integer >= 3")
        lo, hi = self.day_hours
        if not (0 <= lo <= hi <= 23):
            raise ValueError("day_hours must satisfy 0 <= start <= end <= 23")
        return self


class FoldSpec(BaseModel):
    """검증 폴드: 시험 운영일 구간 [test_start, test_end]."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    name: str
    test_start: pd.Timestamp = Field()
    test_end: pd.Timestamp = Field()

    @field_validator("test_start", "test_end", mode="before")
    @classmethod
    def _coerce_op_date(cls, v: object) -> pd.Timestamp:
        ts = v if isinstance(v, pd.Timestamp) else pd.Timestamp(v)
        if pd.isna(ts) or str(ts.tzinfo) != _KST:
            raise ValueError("tz-aware Asia/Seoul required")
        if ts != ts.floor("D"):
            raise ValueError("fold boundary must be midnight")
        return ts

    @model_validator(mode="after")
    def _check_fold(self) -> FoldSpec:
        if not self.name:
            raise ValueError("fold name must be non-empty")
        if self.test_end < self.test_start:
            raise ValueError("test_end must be on or after test_start")
        return self


class ValidationConfig(BaseModel):
    """다중 기간 검증·시드 반복·채택 규칙 설정."""

    model_config = ConfigDict(frozen=True)

    folds: tuple[FoldSpec, ...] = (
        FoldSpec(
            name="A",
            test_start=pd.Timestamp("2023-06-01 00:00", tz="Asia/Seoul"),
            test_end=pd.Timestamp("2024-05-31 00:00", tz="Asia/Seoul"),
        ),
        FoldSpec(
            name="B",
            test_start=pd.Timestamp("2024-06-01 00:00", tz="Asia/Seoul"),
            test_end=pd.Timestamp("2025-05-31 00:00", tz="Asia/Seoul"),
        ),
    )
    embargo_days: int = 1
    min_warm_days: int = 1
    seeds: tuple[int, ...] = (0, 1, 2)
    bootstrap_block_days: int = 7
    bootstrap_n: int = 1000
    bootstrap_seed: int = 0
    ci_level: float = 0.95
    min_rel_improvement: float = 0.01
    min_quarter_win_ratio: float = 0.7
    noise_z: float = 2.0
    min_noise_df: int = 4

    @model_validator(mode="after")
    def _check_validation(self) -> ValidationConfig:
        if len(self.folds) < 1:
            raise ValueError("folds must contain at least one fold")
        names = [f.name for f in self.folds]
        if len(set(names)) != len(names):
            raise ValueError("fold names must be distinct")
        starts = [f.test_start for f in self.folds]
        if any(b < a for a, b in pairwise(starts)):
            raise ValueError("folds must be ordered by test_start")
        for prev, nxt in pairwise(self.folds):
            if nxt.test_start <= prev.test_end:
                raise ValueError("fold test windows must not overlap")
        last_op = (CompetitionSpec().train_end - pd.Timedelta(days=1)).floor("D")
        if any(f.test_end > last_op for f in self.folds):
            raise ValueError("fold test_end must not enter the evaluation period")
        if self.embargo_days < 0:
            raise ValueError("embargo_days must be >= 0")
        if self.min_warm_days < 1:
            raise ValueError("min_warm_days must be >= 1")
        if len(self.seeds) < 1 or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be non-empty and distinct")
        if self.bootstrap_block_days < 1:
            raise ValueError("bootstrap_block_days must be >= 1")
        if self.bootstrap_n < 100:
            raise ValueError("bootstrap_n must be >= 100")
        if not 0.0 < self.ci_level < 1.0 or not 0.5 < (1.0 + self.ci_level) / 2.0 < 1.0:
            raise ValueError("ci_level must be in (0, 1)")
        if not 0.0 <= self.min_rel_improvement < 1.0:
            raise ValueError("min_rel_improvement must be in [0, 1)")
        if not 0.0 < self.min_quarter_win_ratio <= 1.0:
            raise ValueError("min_quarter_win_ratio must be in (0, 1]")
        if not isfinite(self.noise_z) or not self.noise_z > 0:
            raise ValueError("noise_z must be finite and > 0")
        if self.min_noise_df < 1:
            raise ValueError("min_noise_df must be >= 1")
        return self


class ArtifactPaths(BaseModel):
    """실험 산출물 경로."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    root: Path = repo_root() / "artifacts"
    registry: Path = repo_root() / "artifacts" / "registry.jsonl"
    submissions: Path = repo_root() / "artifacts" / "submissions"
    runs: Path = repo_root() / "artifacts" / "runs"
    compositions: Path = repo_root() / "artifacts" / "compositions"


REFERENCE_COMP_ID: Final = "current_group"


class ComposeConfig(BaseModel):
    """지리 층 반클러스터링 구성 파라미터."""

    model_config = ConfigDict(frozen=True)

    n_strata: int = 64
    seed: int = 0
    n_init: int = 10
    km_per_deg: float = 111.0
    ref_lat_deg: float = 36.5

    @model_validator(mode="after")
    def _check_compose(self) -> ComposeConfig:
        if self.n_strata < 2:
            raise ValueError("n_strata must be >= 2")
        if self.n_init < 1:
            raise ValueError("n_init must be >= 1")
        if not isfinite(self.km_per_deg) or not self.km_per_deg > 0:
            raise ValueError("km_per_deg must be finite and > 0")
        if not isfinite(self.ref_lat_deg) or not -90.0 < self.ref_lat_deg < 90.0:
            raise ValueError("ref_lat_deg must be finite and in (-90, 90)")
        return self


class ModelConfig(BaseModel):
    """GBM 학습 파라미터·시드·가중치·타깃 설정."""

    model_config = ConfigDict(frozen=True)

    objective: Literal["l1"] = "l1"
    learning_rate: float = 0.05
    num_leaves: int = 127
    min_child_samples: int = 200
    subsample: float = 0.8
    subsample_freq: int = 1
    colsample_bytree: float = 0.8
    reg_lambda: float = 0.0
    n_rounds: int = 500
    n_jobs: int = 6
    seeds: tuple[int, ...] = (0, 1, 2)
    weight: Literal["smp_k_cap", "k_cap", "none"] = "smp_k_cap"
    target_clip_max: float = 1.2
    min_weight: float = 1e-3

    @model_validator(mode="after")
    def _check_model(self) -> ModelConfig:
        for name in ("learning_rate", "subsample", "colsample_bytree", "reg_lambda", "target_clip_max", "min_weight"):
            if not isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.min_child_samples < 1 or self.subsample_freq < 0 or self.reg_lambda < 0:
            raise ValueError("min_child_samples must be positive; subsample_freq and reg_lambda must be non-negative")
        if not 0.0 < self.learning_rate <= 1.0:
            raise ValueError("learning_rate must be in (0, 1]")
        if self.num_leaves < 2:
            raise ValueError("num_leaves must be >= 2")
        if self.n_rounds < 1:
            raise ValueError("n_rounds must be >= 1")
        if self.n_jobs < 1:
            raise ValueError("n_jobs must be >= 1")
        if not 0.0 < self.subsample <= 1.0:
            raise ValueError("subsample must be in (0, 1]")
        if not 0.0 < self.colsample_bytree <= 1.0:
            raise ValueError("colsample_bytree must be in (0, 1]")
        if len(self.seeds) < 1 or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be non-empty and distinct")
        if self.target_clip_max < 1.0:
            raise ValueError("target_clip_max must be >= 1")
        if self.min_weight <= 0.0:
            raise ValueError("min_weight must be > 0")
        return self


_RUN_ID = re.compile(r"[A-Za-z0-9_.-]+")


class RunConfig(BaseModel):
    """실험 1회의 식별자·가설·특성·모델 설정."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    hypothesis: str
    features: FeatureConfig = FeatureConfig()
    model: ModelConfig = ModelConfig()
    baseline_run_id: str | None = None
    composition_id: str = REFERENCE_COMP_ID
    note: str = ""
    allow_commit_drift: bool = False

    @field_validator("run_id", "baseline_run_id", "composition_id")
    @classmethod
    def _check_run_id(cls, v: str | None) -> str | None:
        if v is not None and (not v or v in {".", ".."} or _RUN_ID.fullmatch(v) is None):
            raise ValueError("run_id must match [A-Za-z0-9_.-]+")
        return v

    @field_validator("hypothesis")
    @classmethod
    def _check_hypothesis(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("hypothesis must be non-empty")
        return v
