"""원본 데이터셋을 L1 canonical으로 변환한다."""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from src.oibc.config import CompetitionSpec, DataPaths, QualityConfig
from src.oibc.data import schemas
from src.oibc.quality import rules
from src.oibc.timeaxis import cutoff, issued_at, k_weight, op_date

logger = logging.getLogger(__name__)

_KST = "Asia/Seoul"
_DT = "datetime64[us, Asia/Seoul]"

_WB_WIND_COLS = ("wb_wind_speed", "wb_wind_gust_spd")


class IngestError(ValueError):
    """입력 계약 위반."""


@dataclass(frozen=True)
class TableReport:
    """테이블별 입출력 행 수와 제외·플래그 집계."""

    rows_in: int
    rows_out: int
    excluded: Mapping[str, int]
    flags: Mapping[str, int]


def _as_kst_ts(raw: pd.Series, name: str) -> pd.Series:
    parsed = pd.to_datetime(raw, errors="raise")
    dti = pd.DatetimeIndex(parsed)
    if dti.tz is None:
        raise IngestError(f"{name}: tz-naive timestamp")
    if str(dti.tz) not in {_KST, "UTC+09:00"}:
        raise IngestError(f"{name}: must be KST")
    if dti.hasnans:
        raise IngestError(f"{name}: missing timestamp")
    return pd.Series(dti.tz_convert(_KST), index=raw.index)


def _drop_non_hourly(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    mask = df["ts"] == df["ts"].dt.floor("h")
    return df[mask].copy(), int((~mask).sum())


def _drop_all_dupes(df: pd.DataFrame, keys: list[str]) -> tuple[pd.DataFrame, int]:
    if df[keys].isna().any().any():
        raise IngestError(f"missing key: {keys}")
    mask = df.duplicated(subset=keys, keep=False)
    return df[~mask].copy(), int(mask.sum())


def _period(ts: pd.Series, spec: CompetitionSpec) -> pd.Series:
    return pd.Series(np.where(ts <= spec.train_end, "train", "eval"), index=ts.index, dtype="string")


def _flag_counts(df: pd.DataFrame, cols: list[str]) -> dict[str, int]:
    return {c: int(df[c].sum()) for c in cols if c in df.columns}


def _cast_like_schema(df: pd.DataFrame, schema: schemas.TableSchema) -> pd.DataFrame:
    out = df.copy()
    for col, want in schema.columns.items():
        if col in out.columns and str(out[col].dtype) != want:
            out[col] = out[col].astype(want)
    return out


def _report(rows_in: int, df: pd.DataFrame, excluded: dict[str, int], flags: dict[str, int]) -> TableReport:
    return TableReport(
        rows_in=rows_in,
        rows_out=len(df),
        excluded={k: v for k, v in excluded.items() if v},
        flags=flags,
    )


def canonical_smp(
    train: pd.DataFrame, eval_: pd.DataFrame, spec: CompetitionSpec
) -> tuple[pd.DataFrame, TableReport]:
    """학습·평가 SMP를 합쳐 canonical SMP를 반환한다."""
    rows_in = len(train) + len(eval_)
    excluded: dict[str, int] = {}
    left = train.copy()
    right = eval_.copy()
    left["ts"] = _as_kst_ts(left["ts"], "ts")
    right["ts"] = _as_kst_ts(right["ts"], "ts")
    left, n = _drop_non_hourly(left)
    excluded["non_hourly"] = excluded.get("non_hourly", 0) + n
    right, n = _drop_non_hourly(right)
    excluded["non_hourly"] = excluded.get("non_hourly", 0) + n
    if "k" not in right.columns:
        raise IngestError("smp: missing eval k")
    expect = k_weight(right["ts"], spec)
    if not bool(right["k"].eq(expect).fillna(False).all()):
        raise IngestError("smp: eval k mismatch")
    if len(set(left["ts"].to_numpy()) & set(right["ts"].to_numpy())):
        raise IngestError("smp: train/eval ts overlap")
    combined = pd.concat([left, right], ignore_index=True)
    combined, n_dup = _drop_all_dupes(combined, ["ts"])
    excluded["duplicate_key"] = n_dup
    combined = combined.sort_values("ts").reset_index(drop=True)
    combined["op_date"] = op_date(combined["ts"])
    combined["k"] = k_weight(combined["ts"], spec)
    combined["period"] = _period(combined["ts"], spec)
    out = _cast_like_schema(combined[["ts", "op_date", "smp", "k", "period"]], schemas.SMP)
    eval_rows = out[out["period"] == "eval"].reset_index(drop=True)
    want_ts = pd.Series(
        pd.date_range(spec.eval_start, spec.eval_end, freq="h", tz=_KST).as_unit("us"),
        dtype=_DT,
    )
    eval_source, _ = _drop_all_dupes(right, ["ts"])
    source_ts = eval_source["ts"].sort_values().reset_index(drop=True).astype(_DT)
    if (
        len(eval_rows) != spec.n_eval_hours
        or not eval_rows["ts"].reset_index(drop=True).equals(want_ts)
        or not source_ts.equals(want_ts)
    ):
        raise IngestError("smp: eval grid incomplete")
    schemas.validate_frame(out, schemas.SMP)
    return out, _report(rows_in, out, excluded, {})


def canonical_generation(
    raw: pd.DataFrame, sites: pd.DataFrame, qcfg: QualityConfig
) -> tuple[pd.DataFrame, TableReport]:
    """원본 발전량을 정규화하고 품질 플래그를 붙인다."""
    rows_in = len(raw)
    df = raw.copy()
    known = set(sites["site_id"].unique())
    if not df["site_id"].isin(known).all():
        raise IngestError("generation: unknown site_id")
    df["ts"] = _as_kst_ts(df["ts"], "ts")
    df, n_hour = _drop_non_hourly(df)
    df, n_dup = _drop_all_dupes(df, ["site_id", "ts"])
    df["energy"] = df["energy"].astype("float32")
    df["op_date"] = op_date(df["ts"])
    df = rules.generation_flags(df, sites, qcfg)
    out = _cast_like_schema(df, schemas.GENERATION)
    out = out.sort_values(["site_id", "ts"]).reset_index(drop=True)
    schemas.validate_frame(out, schemas.GENERATION)
    flags = _flag_counts(out, ["dq_negative", "dq_over_cap", "dq_night_gen", "dq_zero_day"])
    return out, _report(rows_in, out, {"non_hourly": n_hour, "duplicate_key": n_dup}, flags)


def canonical_sites(
    raw: pd.DataFrame, generation: pd.DataFrame, spec: CompetitionSpec
) -> tuple[pd.DataFrame, TableReport]:
    """원본 sites에 발전량 최초 시각을 붙인다."""
    rows_in = len(raw)
    if len(raw) != spec.n_sites:
        raise IngestError(f"sites: rows {len(raw)} != {spec.n_sites}")
    if raw["site_id"].duplicated().any():
        raise IngestError("sites: duplicate site_id")
    capacity = raw["capacity_kw"].to_numpy(dtype="float64")
    if not np.isfinite(capacity).all() or bool((capacity <= 0).any()):
        raise IngestError("sites: non-positive capacity_kw")
    gen = generation[["site_id", "ts"]].copy()
    gen["ts"] = _as_kst_ts(gen["ts"], "ts")
    gen, _ = _drop_non_hourly(gen)
    gen, _ = _drop_all_dupes(gen, ["site_id", "ts"])
    first = gen.groupby("site_id")["ts"].min()
    out = raw.copy()
    mapped = out["site_id"].map(first)
    out["first_ts"] = pd.Series(
        pd.to_datetime(mapped, utc=True).dt.tz_convert(_KST), index=out.index, dtype=_DT
    )
    out = _cast_like_schema(out, schemas.SITES)
    out = out.sort_values("site_id").reset_index(drop=True)
    schemas.validate_frame(out, schemas.SITES)
    return out, _report(rows_in, out, {}, {})


def canonical_forecast(
    train: pd.DataFrame, eval_: pd.DataFrame, provider: Literal["wa", "wb"], spec: CompetitionSpec
) -> tuple[pd.DataFrame, TableReport]:
    """학습·평가 예보를 합쳐 canonical 예보를 반환한다."""
    station = "wa_station" if provider == "wa" else "wb_station"
    schema = schemas.FORECAST_WA if provider == "wa" else schemas.FORECAST_WB
    rows_in = len(train) + len(eval_)
    left = train.copy()
    right = eval_.copy()
    left["ts"] = _as_kst_ts(left["ts"], "ts")
    right["ts"] = _as_kst_ts(right["ts"], "ts")
    keys = [station, "round", "ts"]
    allowed = set(spec.rounds)
    if bool((~left["round"].isin(allowed)).any()) or bool((~right["round"].isin(allowed)).any()):
        raise IngestError(f"forecast_{provider}: round outside {sorted(allowed)}")
    left, n_left = _drop_non_hourly(left)
    right, n_right = _drop_non_hourly(right)
    n_hour = n_left + n_right
    left_keys = pd.MultiIndex.from_frame(left[keys])
    right_keys = pd.MultiIndex.from_frame(right[keys])
    if left_keys.isin(right_keys).any():
        raise IngestError(f"forecast_{provider}: train/eval key overlap")
    combined = pd.concat([left, right], ignore_index=True)
    combined, n_dup = _drop_all_dupes(combined, keys)
    combined["round"] = combined["round"].astype("int8")
    combined["op_date"] = op_date(combined["ts"])
    combined["issued_at"] = issued_at(combined["op_date"], combined["round"], spec)
    cut = cutoff(combined["op_date"], spec)
    if bool((combined["issued_at"] > cut).any()):
        raise IngestError(f"forecast_{provider}: issued_at after cutoff")
    lead = (combined["ts"] - combined["issued_at"]).dt.total_seconds() // 3600
    combined["lead_h"] = lead.astype("int16")
    combined["period"] = _period(combined["ts"], spec)
    if provider == "wb":
        for col in _WB_WIND_COLS:
            if col in combined.columns:
                combined[col] = combined[col].to_numpy(dtype="float64") / 3.6
    out = _cast_like_schema(combined, schema)
    out = out.sort_values([station, "round", "ts"]).reset_index(drop=True)
    schemas.validate_frame(out, schema)
    return out, _report(rows_in, out, {"non_hourly": n_hour, "duplicate_key": n_dup}, {})


def canonical_observed(
    raw: pd.DataFrame, provider: Literal["wa", "wb"], qcfg: QualityConfig
) -> tuple[pd.DataFrame, TableReport]:
    """실측 기상을 정규화하고 sentinel 플래그를 붙인다."""
    station = "wa_station" if provider == "wa" else "wb_station"
    schema = schemas.OBSERVED_WA if provider == "wa" else schemas.OBSERVED_WB
    rows_in = len(raw)
    df = raw.copy()
    df["ts"] = _as_kst_ts(df["ts"], "ts")
    df, n_hour = _drop_non_hourly(df)
    df, n_dup = _drop_all_dupes(df, [station, "ts"])
    df["op_date"] = op_date(df["ts"])
    if provider == "wb" and "wb_wind_speed" in df.columns:
        df["wb_wind_speed"] = df["wb_wind_speed"].to_numpy(dtype="float64") / 3.6
    df["dq_sentinel"] = rules.observed_sentinel(df, qcfg).astype("bool")
    out = _cast_like_schema(df, schema)
    out = out.sort_values([station, "ts"]).reset_index(drop=True)
    schemas.validate_frame(out, schema)
    return out, _report(rows_in, out, {"non_hourly": n_hour, "duplicate_key": n_dup}, _flag_counts(out, ["dq_sentinel"]))


_RAW_FILES = (
    "sites.csv",
    "smp.csv",
    "train/smp.csv",
    "train/generation.parquet",
    "train/forecast_wa.parquet",
    "train/forecast_wb.parquet",
    "train/observed_wa.parquet",
    "train/observed_wb.parquet",
    "test/forecast_wa.parquet",
    "test/forecast_wb.parquet",
)

_TABLE_FILES = (
    "sites",
    "smp",
    "generation",
    "forecast_wa",
    "forecast_wb",
    "observed_wa",
    "observed_wb",
)


def _sha256(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _git_rev() -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    rev = proc.stdout.strip()
    return rev if rev else None


def _write_table(df: pd.DataFrame, path: Path, report: TableReport, name: str) -> None:
    df.to_parquet(path, compression="zstd", index=False)
    logger.info(
        "[DATA] canonical table=%s rows_in=%d rows_out=%d excluded=%s",
        name,
        report.rows_in,
        report.rows_out,
        dict(report.excluded),
    )


def build_canonical(
    paths: DataPaths, spec: CompetitionSpec, qcfg: QualityConfig
) -> dict[str, TableReport]:
    """원본 전체를 canonical Parquet과 manifest로 변환한다."""
    raw_dir = paths.raw_dir
    out_dir = paths.canonical_dir
    raw_root = raw_dir.resolve()
    output_root = out_dir.resolve()
    outputs = [out_dir / f"{name}.parquet" for name in _TABLE_FILES] + [out_dir / "manifest.json"]
    if output_root.is_relative_to(raw_root) or any(p.resolve().is_relative_to(raw_root) for p in outputs):
        raise IngestError("canonical_dir: output must not overwrite raw data")
    out_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict[str, object]] = {}
    for rel in _RAW_FILES:
        digest, size = _sha256(raw_dir / rel)
        files[rel] = {"sha256": digest, "bytes": size}
    reports: dict[str, TableReport] = {}
    gen_raw = pd.read_parquet(raw_dir / "train/generation.parquet")
    sites_raw = pd.read_csv(raw_dir / "sites.csv")
    sites_df, reports["sites"] = canonical_sites(sites_raw, gen_raw, spec)
    _write_table(sites_df, out_dir / "sites.parquet", reports["sites"], "sites")
    gen_df, reports["generation"] = canonical_generation(gen_raw, sites_df, qcfg)
    _write_table(gen_df, out_dir / "generation.parquet", reports["generation"], "generation")
    del gen_raw, gen_df
    train_smp = pd.read_csv(raw_dir / "train/smp.csv")
    eval_smp = pd.read_csv(raw_dir / "smp.csv")
    smp_df, reports["smp"] = canonical_smp(train_smp, eval_smp, spec)
    _write_table(smp_df, out_dir / "smp.parquet", reports["smp"], "smp")
    del train_smp, eval_smp, smp_df
    provider: Literal["wa", "wb"]
    for provider in ("wa", "wb"):
        fc_train = pd.read_parquet(raw_dir / f"train/forecast_{provider}.parquet")
        fc_test = pd.read_parquet(raw_dir / f"test/forecast_{provider}.parquet")
        fc_df, reports[f"forecast_{provider}"] = canonical_forecast(
            fc_train, fc_test, provider, spec
        )
        _write_table(
            fc_df,
            out_dir / f"forecast_{provider}.parquet",
            reports[f"forecast_{provider}"],
            f"forecast_{provider}",
        )
        del fc_train, fc_test, fc_df
    for provider in ("wa", "wb"):
        obs_raw = pd.read_parquet(raw_dir / f"train/observed_{provider}.parquet")
        obs_df, reports[f"observed_{provider}"] = canonical_observed(
            obs_raw, provider, qcfg
        )
        _write_table(
            obs_df,
            out_dir / f"observed_{provider}.parquet",
            reports[f"observed_{provider}"],
            f"observed_{provider}",
        )
        del obs_raw, obs_df
    manifest = {
        "files": files,
        "git_rev": _git_rev(),
        "quality": qcfg.model_dump(mode="json"),
        "spec": spec.model_dump(mode="json"),
        "tables": {
            name: {
                "rows_in": reports[name].rows_in,
                "rows_out": reports[name].rows_out,
                "excluded": dict(reports[name].excluded),
                "flags": dict(reports[name].flags),
            }
            for name in _TABLE_FILES
        },
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, sort_keys=True, ensure_ascii=False, indent=2)
    return reports
