"""Canonical 테이블 계약."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

_DT = "datetime64[us, Asia/Seoul]"

_WA_VARS = (
    "wa_cloud",
    "wa_temp",
    "wa_temp_max",
    "wa_temp_min",
    "wa_humidity",
    "wa_ground_press",
    "wa_wind_speed",
    "wa_wind_dir",
    "wa_rain",
    "wa_snow",
)

_WB_FC_VARS = (
    "wb_temp",
    "wb_feel_temp",
    "wb_wet_bulb_temp",
    "wb_dew_point",
    "wb_wind_speed",
    "wb_wind_dir",
    "wb_wind_gust_spd",
    "wb_rel_hum",
    "wb_vis",
    "wb_ceiling",
    "wb_uv_idx",
    "wb_precip_prob",
    "wb_rain_prob",
    "wb_snow_prob",
    "wb_ice_prob",
    "wb_total_liq",
    "wb_rain",
    "wb_snow",
    "wb_ice",
    "wb_cloud",
)

_WB_OB_VARS = (
    "wb_temp",
    "wb_feel_temp",
    "wb_feel_temp_shade",
    "wb_rel_hum",
    "wb_dew_point",
    "wb_wind_dir",
    "wb_wind_speed",
    "wb_uv_idx",
    "wb_vis",
    "wb_cloud",
    "wb_ceiling",
    "wb_pressure",
    "wb_appr_temp",
    "wb_wind_chill_temp",
    "wb_precip_1h",
)


@dataclass(frozen=True)
class TableSchema:
    """Canonical 테이블 계약."""

    name: str
    key: tuple[str, ...]
    columns: Mapping[str, str]


class SchemaError(ValueError):
    """스키마 계약 위반."""


def _forecast_columns(station: str) -> dict[str, str]:
    base: dict[str, str] = {
        station: "string",
        "round": "int8",
        "ts": _DT,
        "op_date": _DT,
        "issued_at": _DT,
        "lead_h": "int16",
        "period": "string",
    }
    variables = _WA_VARS if station == "wa_station" else _WB_FC_VARS
    base.update(dict.fromkeys(variables, "float32"))
    return base


def _observed_columns(station: str) -> dict[str, str]:
    base: dict[str, str] = {station: "string", "ts": _DT, "op_date": _DT}
    variables = _WA_VARS if station == "wa_station" else _WB_OB_VARS
    base.update(dict.fromkeys(variables, "float32"))
    base["dq_sentinel"] = "bool"
    return base


SITES = TableSchema(
    name="sites",
    key=("site_id",),
    columns={
        "site_id": "string",
        "current_group": "string",
        "capacity_kw": "float32",
        "lat": "float64",
        "lon": "float64",
        "wa_station": "string",
        "wb_station": "string",
        "wa_dist": "float64",
        "wb_dist": "float64",
        "first_ts": _DT,
    },
)

SMP = TableSchema(
    name="smp",
    key=("ts",),
    columns={"ts": _DT, "op_date": _DT, "smp": "float32", "k": "int8", "period": "string"},
)

GENERATION = TableSchema(
    name="generation",
    key=("site_id", "ts"),
    columns={
        "site_id": "string",
        "ts": _DT,
        "op_date": _DT,
        "energy": "float32",
        "dq_negative": "bool",
        "dq_over_cap": "bool",
        "dq_night_gen": "bool",
        "dq_zero_day": "bool",
    },
)

FORECAST_WA = TableSchema(
    name="forecast_wa", key=("wa_station", "round", "ts"), columns=_forecast_columns("wa_station")
)

FORECAST_WB = TableSchema(
    name="forecast_wb", key=("wb_station", "round", "ts"), columns=_forecast_columns("wb_station")
)

OBSERVED_WA = TableSchema(
    name="observed_wa", key=("wa_station", "ts"), columns=_observed_columns("wa_station")
)

OBSERVED_WB = TableSchema(
    name="observed_wb", key=("wb_station", "ts"), columns=_observed_columns("wb_station")
)


def validate_frame(df: pd.DataFrame, schema: TableSchema) -> None:
    """필수 컬럼·dtype·키 유일성·키 정렬을 검사한다."""
    for col, want in schema.columns.items():
        if col not in df.columns:
            raise SchemaError(f"{schema.name}: missing column {col!r}")
        if str(df[col].dtype) != want:
            raise SchemaError(
                f"{schema.name}: column {col!r} dtype {df[col].dtype} != {want}"
            )
    keys = list(schema.key)
    for col in keys:
        if df[col].isna().any():
            raise SchemaError(f"{schema.name}: missing key {col!r}")
    if "period" in schema.columns and not df["period"].isin(["train", "eval"]).all():
        raise SchemaError(f"{schema.name}: invalid period")
    if df.duplicated(subset=keys).any():
        raise SchemaError(f"{schema.name}: duplicate key {schema.key}")
    if len(df) > 1:
        ordered = df[keys].sort_values(keys, kind="mergesort").reset_index(drop=True)
        current = df[keys].reset_index(drop=True)
        if not current.equals(ordered[keys]):
            raise SchemaError(f"{schema.name}: key not sorted by {schema.key}")
