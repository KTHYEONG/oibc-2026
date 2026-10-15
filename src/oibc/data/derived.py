"""L2 파생 테이블 구축."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.oibc.config import CompetitionSpec, DataPaths
from src.oibc.data import gate
from src.oibc.solar import solar_position
from src.oibc.timeaxis import op_date, require_hourly_kst

logger = logging.getLogger(__name__)

_KST = "Asia/Seoul"
_DT = "datetime64[us, Asia/Seoul]"

_META_COLS = ("op_date", "issued_at", "lead_h", "period")


class DerivedError(ValueError):
    """파생 격자 계약 위반."""


def build_grid(
    sites: pd.DataFrame, smp: pd.DataFrame, *, spec: CompetitionSpec | None = None
) -> pd.DataFrame:
    """발전소와 시각의 완전 격자를 반환한다."""
    spec = spec or CompetitionSpec()
    if sites["site_id"].isna().any() or sites["site_id"].duplicated().any():
        raise DerivedError("sites: missing or duplicate site_id")
    capacities = sites["capacity_kw"].to_numpy(dtype="float64")
    if not np.isfinite(capacities).all() or (capacities <= 0).any():
        raise DerivedError("sites: capacity_kw must be finite and positive")
    if sites[["wa_station", "wb_station"]].isna().any().any():
        raise DerivedError("sites: missing station")
    require_hourly_kst(smp["ts"], "smp.ts")
    expected_period = np.where(smp["ts"] <= spec.train_end, "train", "eval")
    if smp["ts"].duplicated().any() or not smp["period"].eq(expected_period).fillna(False).all():
        raise DerivedError("smp: duplicate timestamp or invalid period")
    eval_ts = pd.DatetimeIndex(smp.loc[smp["period"] == "eval", "ts"]).sort_values()
    expected_ts = pd.date_range(spec.eval_start, spec.eval_end, freq="h")
    if not eval_ts.equals(expected_ts):
        raise DerivedError("grid: eval window incomplete or misaligned")
    sites = sites.sort_values("site_id").reset_index(drop=True)
    smp = smp.sort_values("ts").reset_index(drop=True)
    grid = sites[["site_id", "capacity_kw", "wa_station", "wb_station"]].merge(
        smp[["ts", "period"]], how="cross"
    )
    grid["site_id"] = grid["site_id"].astype("string")
    grid["ts"] = grid["ts"].astype(_DT)
    grid["op_date"] = op_date(grid["ts"])
    grid["period"] = grid["period"].astype("string")
    grid["capacity_kw"] = grid["capacity_kw"].astype("float32")
    grid["wa_station"] = grid["wa_station"].astype("string")
    grid["wb_station"] = grid["wb_station"].astype("string")
    return grid[["site_id", "ts", "op_date", "period", "capacity_kw", "wa_station", "wb_station"]]


def build_solar(grid: pd.DataFrame, sites: pd.DataFrame) -> pd.DataFrame:
    """격자 시각의 태양 위치를 반환한다."""
    lookup = sites.set_index("site_id")
    if not lookup.index.is_unique:
        raise DerivedError("sites: duplicate site_id")
    rows = lookup.reindex(pd.Index(grid["site_id"]))
    lat = rows["lat"].to_numpy(dtype="float64")
    lon = rows["lon"].to_numpy(dtype="float64")
    elev, azim, cosz = solar_position(grid["ts"], lat, lon)
    return pd.DataFrame(
        {
            "site_id": grid["site_id"].astype("string").reset_index(drop=True),
            "ts": pd.Series(pd.DatetimeIndex(grid["ts"]), dtype=_DT).reset_index(drop=True),
            "elev": elev,
            "azim": azim,
            "cosz": cosz,
        }
    )


def fill_forecast_rounds(
    forecast: pd.DataFrame, station_col: str, spec: CompetitionSpec
) -> pd.DataFrame:
    """라운드 완전 격자로 확장하고 다른 라운드 값으로 채운다."""
    gate._validate_forecast(forecast, station_col, spec)
    if forecast.empty:
        raise DerivedError("forecast: empty input has no time range")
    stations = pd.Index(forecast[station_col].unique())
    full_ts = pd.date_range(
        forecast["ts"].min(), forecast["ts"].max(), freq="h", tz=_KST
    ).as_unit("us")
    full = pd.MultiIndex.from_product([stations, spec.rounds, full_ts], names=[station_col, "round", "ts"]).to_frame(index=False)
    full[station_col] = full[station_col].astype("string")
    full["round"] = full["round"].astype("int8")
    full["ts"] = full["ts"].astype(_DT)
    var_cols = [c for c in forecast.columns if c not in {station_col, "round", "ts", *_META_COLS}]
    keys = [station_col, "round", "ts"]
    merged = full.merge(forecast, on=keys, how="left", indicator=True, validate="one_to_one")
    merged["fc_source"] = np.where(merged["_merge"] == "both", "orig", "other_round")
    missing = (merged["_merge"] != "both").to_numpy()
    if bool(missing.any()):
        donor = forecast[[*keys, "issued_at", *var_cols]].copy()
        donor["round"] = donor["round"].map(dict(zip(spec.rounds, spec.rounds[::-1], strict=True)))
        pos = np.flatnonzero(missing)
        cand = merged.loc[missing, keys].merge(donor, on=keys, how="left", indicator=True, validate="one_to_one")
        has = (cand["_merge"] == "both").to_numpy()
        for name in [*var_cols, "issued_at"]:
            merged.loc[pos, name] = cand[name].array
        merged.loc[pos[~has], "fc_source"] = "missing"
    merged["fc_source"] = merged["fc_source"].astype("string")
    merged["op_date"] = op_date(merged["ts"])
    merged["period"] = pd.Series(
        np.where(merged["ts"] <= spec.train_end, "train", "eval"), dtype="string"
    )
    lead = (merged["ts"] - merged["issued_at"]).dt.total_seconds() // 3600
    merged["lead_h"] = lead.astype("Int16")
    keep = [*keys, *var_cols, *_META_COLS, "fc_source"]
    filled = merged[keep].sort_values(keys).reset_index(drop=True)
    gate.assert_causal(filled["ts"], filled["issued_at"], spec)
    counts = filled["fc_source"].value_counts().to_dict()
    logger.info("[DATA] derived table=forecast_filled_%s rows=%d fc_source=%s", station_col.split("_")[0], len(filled), counts)
    return filled


def build_derived(paths: DataPaths, spec: CompetitionSpec) -> dict[str, int]:
    """canonical에서 L2 파생 테이블을 구축한다."""
    out_dir = paths.derived_dir
    outputs = [out_dir / f"{name}.parquet" for name in ("grid", "solar", "forecast_filled_wa", "forecast_filled_wb")]
    for protected in (paths.raw_dir.resolve(), paths.canonical_dir.resolve()):
        if out_dir.resolve().is_relative_to(protected) or any(p.resolve().is_relative_to(protected) for p in outputs):
            raise DerivedError("derived_dir: output must not overwrite raw or canonical data")
    out_dir.mkdir(parents=True, exist_ok=True)
    sites = pd.read_parquet(paths.canonical_dir / "sites.parquet")
    smp = pd.read_parquet(paths.canonical_dir / "smp.parquet")
    if len(sites) != spec.n_sites:
        raise DerivedError(f"sites: rows {len(sites)} != {spec.n_sites}")
    grid = build_grid(sites, smp, spec=spec)
    grid.to_parquet(out_dir / "grid.parquet", compression="zstd", index=False)
    logger.info("[DATA] derived table=grid rows=%d", len(grid))
    solar = build_solar(grid, sites)
    solar.to_parquet(out_dir / "solar.parquet", compression="zstd", index=False)
    logger.info("[DATA] derived table=solar rows=%d", len(solar))
    counts = {"grid": len(grid), "solar": len(solar)}
    del solar, grid, smp
    for provider, station in (("wa", "wa_station"), ("wb", "wb_station")):
        forecast = pd.read_parquet(paths.canonical_dir / f"forecast_{provider}.parquet")
        filled = fill_forecast_rounds(forecast, station, spec)
        filled.to_parquet(out_dir / f"forecast_filled_{provider}.parquet", compression="zstd", index=False)
        counts[f"forecast_filled_{provider}"] = len(filled)
    return counts
