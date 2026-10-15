"""대회 공식 상수 및 경로."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

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
