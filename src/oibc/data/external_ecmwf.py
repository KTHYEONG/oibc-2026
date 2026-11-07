"""ECMWF Open Data IFS ssrd 광역 요약과 그룹 예측 2단 보정: 수집, 공개 시각 감사, 블록 특성, β 적합·적용.

고정 상자 요약만 쓰고(발전소 좌표 미사용) 모든 그룹에 같은 x를 곱한다. 계약 위반은 ExternalError로 중단한다.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.oibc.config import CompetitionSpec, DataPaths, EcmwfConfig, EcmwfRunSpec, ExternalGfsConfig
from src.oibc.data.external_gfs import ExternalError, crop_box

logger = logging.getLogger(__name__)

_KST = "Asia/Seoul"
_AUDIT_COLUMNS = ["target_date", "init", "published_utc"]
_SECONDS_PER_HOUR = 3600.0


def run_init(target: pd.Timestamp, run: EcmwfRunSpec) -> pd.Timestamp:
    """대상일(KST 날짜, tz-naive 자정)에 쓰는 실행의 init 시각(UTC)을 반환한다."""
    day = pd.Timestamp(target)
    if day.tzinfo is not None or day != day.normalize():
        raise ExternalError("run_init: target must be a tz-naive date")
    return (day - pd.Timedelta(days=run.days_back) + pd.Timedelta(hours=run.init_hour_utc)).tz_localize("UTC")


def block_leads(run: EcmwfRunSpec, cfg: EcmwfConfig) -> list[tuple[int, int]]:
    """KST 블록마다 (시작 리드, 끝 리드) 시간을 반환한다. 대상일 KST 시각 h의 리드는 24·days_back + h - 9 - init_hour."""
    kst_offset = 9
    base = 24 * run.days_back - kst_offset - run.init_hour_utc
    return [(base + end - cfg.block_hours, base + end) for end in cfg.block_end_hours]


def needed_leads(run: EcmwfRunSpec, cfg: EcmwfConfig) -> list[int]:
    """블록 계산에 필요한 누적 리드 목록(오름차순, 중복 없음). 실행에 리드가 지정돼 있으면 그대로 쓴다."""
    if run.leads is not None:
        return sorted(set(run.leads))
    return sorted({lead for pair in block_leads(run, cfg) for lead in pair})


def run_dir(paths: DataPaths, run: EcmwfRunSpec) -> Path:
    """실행별 대상일 npz 저장 폴더."""
    return paths.external_dir / "ecmwf" / run.name


def audit_path(paths: DataPaths, run: EcmwfRunSpec) -> Path:
    """실행별 공개 시각 감사 CSV 경로."""
    return paths.external_dir / "ecmwf" / f"audit_{run.name}.csv"


def target_dates(cfg: EcmwfConfig, run: EcmwfRunSpec | None = None) -> pd.DatetimeIndex:
    """수집·감사 대상일(KST 날짜, tz-naive). AIFS 실행은 aifs_target_start부터다."""
    start = cfg.aifs_target_start if run is not None and run.product == "aifs-single" else cfg.target_start
    return pd.date_range(start, cfg.target_end, freq="D")


def all_runs(cfg: EcmwfConfig) -> tuple[EcmwfRunSpec, ...]:
    """수집·감사·동등성 대조 대상 실행(IFS 우선순위 실행 + 사용 시 AIFS)."""
    return (*cfg.runs, cfg.aifs) if cfg.use_aifs else cfg.runs


def is_before_cutoff(target: pd.Timestamp, published_utc: pd.Timestamp, spec: CompetitionSpec) -> bool:
    """공개 시각이 대상일 D의 예측 마감(D-1 cutoff_hour KST) 이하인지 반환한다."""
    cutoff = (pd.Timestamp(target) - pd.Timedelta(days=1) + pd.Timedelta(hours=spec.cutoff_hour)).tz_localize(_KST)
    return bool(pd.Timestamp(published_utc) <= cutoff)


def read_audit(path: Path) -> pd.DataFrame:
    """감사 CSV를 target_date(naive)·published_utc(UTC, 결측 가능) 표로 읽는다. 형식 위반 시 ExternalError."""
    if not path.exists():
        raise ExternalError(f"ecmwf audit missing: {path}")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if list(frame.columns) != _AUDIT_COLUMNS:
        raise ExternalError("ecmwf audit: unexpected columns")
    out = pd.DataFrame(
        {
            "target_date": pd.to_datetime(frame["target_date"], format="%Y-%m-%d"),
            "published_utc": pd.to_datetime(frame["published_utc"].replace("", None), utc=True),
        }
    )
    if out["target_date"].duplicated().any():
        raise ExternalError("ecmwf audit: duplicate target_date")
    return out


def block_means(
    arrays: Mapping[str, np.ndarray],
    lat: np.ndarray,
    lon: np.ndarray,
    run: EcmwfRunSpec,
    cfg: EcmwfConfig,
    *,
    whole_box: bool = False,
) -> np.ndarray | None:
    """누적 ssrd(J/m²)에서 블록별 내부 상자(whole_box면 저장 상자 전체) 평균 일사(W/m²). 리드가 하나라도 없으면 None."""
    keys = [f"{cfg.param}_{lead}" for lead in needed_leads(run, cfg)]
    if any(k not in arrays for k in keys):
        return None
    lat = np.asarray(lat, dtype="float64")
    lon = np.asarray(lon, dtype="float64")
    inner = ((lat >= cfg.inner_lat[0]) & (lat <= cfg.inner_lat[1]))[:, None] & (
        (lon >= cfg.inner_lon[0]) & (lon <= cfg.inner_lon[1])
    )[None, :]
    if not inner.any():
        raise ExternalError("block_means: empty inner box")
    out = np.empty(len(cfg.block_end_hours), dtype="float64")
    for b, (s0, s1) in enumerate(block_leads(run, cfg)):
        acc1 = np.asarray(arrays[f"{cfg.param}_{s1}"], dtype="float64")
        acc0 = np.asarray(arrays[f"{cfg.param}_{s0}"], dtype="float64")
        if acc1.shape != inner.shape or acc0.shape != inner.shape:
            raise ExternalError("block_means: array shape does not match lat/lon")
        rate = (acc1 - acc0) / ((s1 - s0) * _SECONDS_PER_HOUR)
        out[b] = rate.mean() if whole_box else rate[inner].mean()
    return out


def ec_blocks(paths: DataPaths, cfg: EcmwfConfig, spec: CompetitionSpec) -> pd.DataFrame:
    """대상일·블록마다 마감 전에 공개된 우선순위 첫 실행의 내부 상자 일사를 고른다.

    반환 열: op(naive 날짜), blk(int), ec_in·ec_box(내부·저장 상자 W/m²), src(실행 이름). 쓸 실행이 없는 블록은 행이 없다.
    """
    chosen: dict[tuple[pd.Timestamp, int], tuple[float, float, str]] = {}
    for run in cfg.runs:
        audit = read_audit(audit_path(paths, run)).set_index("target_date")["published_utc"]
        folder = run_dir(paths, run)
        n_used = 0
        for day in target_dates(cfg):
            if any((day, b) in chosen for b in range(len(cfg.block_end_hours))):
                continue
            pub = audit.get(day)
            if pub is None or pd.isna(pub) or not is_before_cutoff(day, pub, spec):
                continue
            path = folder / f"{day:%Y%m%d}.npz"
            if not path.exists():
                continue
            with np.load(path) as z:
                arrays = {k: z[k] for k in z.files if k.startswith(cfg.param)}
                means = block_means(arrays, z["lat"], z["lon"], run, cfg)
                box = block_means(arrays, z["lat"], z["lon"], run, cfg, whole_box=True)
            if means is None:
                continue
            for b, v in enumerate(means):
                chosen[(day, b)] = (float(v), float(box[b]), run.name)  # type: ignore[index]
            n_used += 1
        logger.info("[DATA] ecmwf run=%s days_used=%d", run.name, n_used)
    rows = [(op, b, v, vb, src) for (op, b), (v, vb, src) in sorted(chosen.items())]
    out = pd.DataFrame(rows, columns=["op", "blk", "ec_in", "ec_box", "src"])
    out["blk"] = out["blk"].astype("int64")
    return out


def block_index(ts: pd.Series, cfg: EcmwfConfig) -> pd.Series:
    """KST right-closed 시각을 블록 번호(Int64)로 바꾼다. 블록 밖은 <NA>."""
    hour = pd.Series(ts.dt.hour.to_numpy(), index=ts.index).where(lambda h: h != 0, 24)
    out = pd.Series(pd.NA, index=ts.index, dtype="Int64")
    for b, end in enumerate(cfg.block_end_hours):
        out[(hour > end - cfg.block_hours) & (hour <= end)] = b
    return out


def op_day(ts: pd.Series) -> pd.Series:
    """right-closed 시각의 운영일(tz-naive 자정)."""
    return (ts - pd.Timedelta(hours=1)).dt.tz_localize(None).dt.normalize()


def gfs_blocks(gfs_hourly: pd.DataFrame, cfg: EcmwfConfig) -> pd.DataFrame:
    """GFS 시간별 표에서 운영일·블록 평균 일사(g_in)를 만든다."""
    if "ts" not in gfs_hourly.columns or cfg.gfs_column not in gfs_hourly.columns:
        raise ExternalError("gfs_blocks: missing columns")
    g = pd.DataFrame(
        {"op": op_day(gfs_hourly["ts"]), "blk": block_index(gfs_hourly["ts"], cfg), "g": gfs_hourly[cfg.gfs_column]}
    )
    g = g.dropna(subset=["blk"]).astype({"blk": "int64"})
    return g.groupby(["op", "blk"]).agg(g_in=("g", "mean")).reset_index()


def x_table(ec: pd.DataFrame, gfs: pd.DataFrame, cfg: EcmwfConfig) -> pd.DataFrame:
    """x = (EC - GFS)/(GFS + offset). 두 쪽이 모두 있는 블록만 남긴다."""
    t = ec.merge(gfs, on=["op", "blk"], how="inner")
    t["x"] = (t["ec_in"] - t["g_in"]) / (t["g_in"] + cfg.x_offset_wm2)
    return t.dropna(subset=["x"])[["op", "blk", "x", "src"]].reset_index(drop=True)


# ---- AIFS ----


def aifs_windows(cfg: EcmwfConfig) -> list[tuple[int, int]]:
    """AIFS 누적 창(연속 리드 쌍) 목록."""
    leads = needed_leads(cfg.aifs, cfg)
    return list(pairwise(leads))


def aifs_window_ts_hours(cfg: EcmwfConfig) -> list[range]:
    """창마다 대상일 right-closed ts 시각(1~24) 범위. 대상일 밖으로 넘으면 ExternalError."""
    base = 24 * cfg.aifs.days_back - 9 - cfg.aifs.init_hour_utc
    out = []
    for a, b in aifs_windows(cfg):
        lo, hi = a - base + 1, b - base
        if lo < 1 or hi > 23:
            raise ExternalError("aifs window outside the target day")
        out.append(range(lo, hi + 1))
    return out


def aifs_block_means(
    arrays: Mapping[str, np.ndarray], lat: np.ndarray, lon: np.ndarray, cfg: EcmwfConfig, *, whole_box: bool = False
) -> np.ndarray | None:
    """AIFS 누적 ssrd에서 블록별(창 대응) 내부 상자(whole_box면 저장 상자 전체) 평균 일사(W/m²). 리드가 없으면 None."""
    if any(f"{cfg.param}_{lead}" not in arrays for lead in needed_leads(cfg.aifs, cfg)):
        return None
    lat = np.asarray(lat, dtype="float64")
    lon = np.asarray(lon, dtype="float64")
    inner = ((lat >= cfg.inner_lat[0]) & (lat <= cfg.inner_lat[1]))[:, None] & (
        (lon >= cfg.inner_lon[0]) & (lon <= cfg.inner_lon[1])
    )[None, :]
    if not inner.any():
        raise ExternalError("aifs_block_means: empty inner box")
    win = []
    for a, b in aifs_windows(cfg):
        acc1 = np.asarray(arrays[f"{cfg.param}_{b}"], dtype="float64")
        acc0 = np.asarray(arrays[f"{cfg.param}_{a}"], dtype="float64")
        if acc1.shape != inner.shape or acc0.shape != inner.shape:
            raise ExternalError("aifs_block_means: array shape does not match lat/lon")
        rate = (acc1 - acc0) / ((b - a) * _SECONDS_PER_HOUR)
        win.append(rate.mean() if whole_box else rate[inner].mean())
    return np.array([win[w] for w in cfg.aifs_block_window], dtype="float64")


def aifs_blocks(paths: DataPaths, cfg: EcmwfConfig, spec: CompetitionSpec) -> pd.DataFrame:
    """마감 전에 공개된 AIFS 실행의 블록별 일사. 반환 열: op, blk, a_in(내부 상자), a_box(저장 상자)."""
    audit = read_audit(audit_path(paths, cfg.aifs)).set_index("target_date")["published_utc"]
    folder = run_dir(paths, cfg.aifs)
    rows: list[tuple[pd.Timestamp, int, float, float]] = []
    for day in target_dates(cfg, cfg.aifs):
        pub = audit.get(day)
        path = folder / f"{day:%Y%m%d}.npz"
        if pub is None or pd.isna(pub) or not is_before_cutoff(day, pub, spec) or not path.exists():
            continue
        with np.load(path) as z:
            arrays = {k: z[k] for k in z.files if k.startswith(cfg.param)}
            means = aifs_block_means(arrays, z["lat"], z["lon"], cfg)
            box = aifs_block_means(arrays, z["lat"], z["lon"], cfg, whole_box=True)
        if means is not None:
            rows.extend((day, b, float(v), float(box[b])) for b, v in enumerate(means))  # type: ignore[index]
    logger.info("[DATA] ecmwf run=%s days_used=%d", cfg.aifs.name, len(rows) // max(len(cfg.block_end_hours), 1))
    out = pd.DataFrame(rows, columns=["op", "blk", "a_in", "a_box"])
    out["blk"] = out["blk"].astype("int64")
    return out


def gfs_aifs_blocks(gfs_hourly: pd.DataFrame, cfg: EcmwfConfig) -> pd.DataFrame:
    """AIFS 창과 같은 ts 시각의 GFS 평균 일사를 블록에 대응시킨다. 반환 열: op, blk, g_a."""
    if "ts" not in gfs_hourly.columns or cfg.gfs_column not in gfs_hourly.columns:
        raise ExternalError("gfs_aifs_blocks: missing columns")
    op = op_day(gfs_hourly["ts"])
    hour = gfs_hourly["ts"].dt.hour
    win_means = []
    for hours in aifs_window_ts_hours(cfg):
        sel = hour.isin(list(hours))
        win_means.append(gfs_hourly.loc[sel, cfg.gfs_column].groupby(op[sel]).mean())
    parts = [
        pd.DataFrame({"op": win_means[w].index, "blk": b, "g_a": win_means[w].to_numpy()})
        for b, w in enumerate(cfg.aifs_block_window)
    ]
    out = pd.concat(parts, ignore_index=True)
    out["blk"] = out["blk"].astype("int64")
    return out


def aifs_x(aifs: pd.DataFrame, gfs: pd.DataFrame, cfg: EcmwfConfig) -> pd.DataFrame:
    """x_a = (AIFS - GFS 같은 창)/(GFS + offset). 두 쪽이 모두 있는 블록만 남긴다."""
    t = aifs.merge(gfs, on=["op", "blk"], how="inner")
    t["x_a"] = (t["a_in"] - t["g_a"]) / (t["g_a"] + cfg.x_offset_wm2)
    return t.dropna(subset=["x_a"])[["op", "blk", "x_a"]].reset_index(drop=True)


def split_aifs_x(
    aifs: pd.DataFrame, ec: pd.DataFrame, gfs_window: pd.DataFrame, gfs_block: pd.DataFrame, cfg: EcmwfConfig
) -> pd.DataFrame:
    """AIFS x를 만들되, 한 창을 공유하는 블록은 HRES 블록 일사 비율로 나눠 그 블록 GFS와 비교한다.

    HRES 값이 그 창의 블록 전부에 없거나 평균이 1 W/m² 이하인 날은 창 값을 그대로 쓴다. 반환 열: op, blk, x_a.
    """
    shared = {b for b in range(len(cfg.aifs_block_window)) if cfg.aifs_block_window.count(cfg.aifs_block_window[b]) > 1}
    t = aifs.merge(gfs_window, on=["op", "blk"], how="inner")
    t = t.merge(ec[["op", "blk", "ec_in"]], on=["op", "blk"], how="left").merge(gfs_block, on=["op", "blk"], how="left")
    t["win"] = t["blk"].map(dict(enumerate(cfg.aifs_block_window)))
    grp = t[t["blk"].isin(shared)].groupby(["op", "win"])["ec_in"]
    t["hm"] = grp.transform("mean").reindex(t.index)
    t["hn"] = grp.transform("count").reindex(t.index)
    need = t["win"].map(pd.Series(cfg.aifs_block_window).value_counts())
    split = t["blk"].isin(shared) & (t["hn"] == need) & (t["hm"] > 1.0) & t["g_in"].notna()
    t["x_a"] = (t["a_in"] - t["g_a"]) / (t["g_a"] + cfg.x_offset_wm2)
    a_split = t.loc[split, "a_in"] * t.loc[split, "ec_in"] / t.loc[split, "hm"]
    t.loc[split, "x_a"] = (a_split - t.loc[split, "g_in"]) / (t.loc[split, "g_in"] + cfg.x_offset_wm2)
    return t.dropna(subset=["x_a"])[["op", "blk", "x_a"]].reset_index(drop=True)


def combine_aifs(xt: pd.DataFrame, xa: pd.DataFrame, cfg: EcmwfConfig) -> pd.DataFrame:
    """HRES x에 AIFS x를 결합한다. 00z 블록은 (1-w)·x + w·x_a, 그 밖(18z·12z·HRES 없음)은 x_a로 바꾼다."""
    w = cfg.aifs_weight_00z
    t = xt[["op", "blk", "x", "src"]].merge(xa, on=["op", "blk"], how="outer")
    has_a = t["x_a"].notna()
    blend = has_a & (t["src"] == "00z")
    swap = has_a & ~blend
    t.loc[blend, "x"] = (1.0 - w) * t.loc[blend, "x"] + w * t.loc[blend, "x_a"]
    t.loc[blend, "src"] = "00z+aifs"
    t.loc[swap, "x"] = t.loc[swap, "x_a"]
    t.loc[swap, "src"] = cfg.aifs.name
    out = t.dropna(subset=["x"])[["op", "blk", "x", "src"]].sort_values(["op", "blk"]).reset_index(drop=True)
    out["blk"] = out["blk"].astype("int64")
    return out


def beta_grid(cfg: EcmwfConfig) -> np.ndarray:
    """β 탐색 격자(끝 포함)."""
    start, end, step = cfg.beta_grid
    if step <= 0 or end < start:
        raise ExternalError("beta_grid: invalid range")
    return np.round(np.arange(start, end + step / 5, step), 10)


def fit_beta(forecast: np.ndarray, actual: np.ndarray, weight: np.ndarray, x: np.ndarray, cfg: EcmwfConfig) -> float:
    """sum w·|F·(1+βx) - A|를 최소화하는 격자 β. 입력 길이가 다르거나 비었거나 비유한 값이면 ExternalError."""
    arrays = [np.asarray(a, dtype="float64") for a in (forecast, actual, weight, x)]
    if len({a.shape for a in arrays}) != 1 or arrays[0].size == 0:
        raise ExternalError("fit_beta: inputs must be non-empty and the same length")
    if not all(np.isfinite(a).all() for a in arrays):
        raise ExternalError("fit_beta: non-finite input")
    f, a, w, xx = arrays
    grid = beta_grid(cfg)
    loss = np.array([np.sum(w * np.abs(f * (1.0 + g * xx) - a)) for g in grid])
    return float(grid[int(np.argmin(loss))])


def correction_multiplier(ts: pd.Series, xt: pd.DataFrame, beta: float, cfg: EcmwfConfig) -> np.ndarray:
    """각 시각의 보정 배율 1+βx. x가 없는 시각(블록 밖·자료 없음)은 1. 배율이 0 이하이면 ExternalError."""
    keys = pd.DataFrame({"op": op_day(ts).to_numpy(), "blk": block_index(ts, cfg).to_numpy()})
    keys = keys.merge(xt[["op", "blk", "x"]].astype({"blk": "Int64"}), on=["op", "blk"], how="left")
    if len(keys) != len(ts):
        raise ExternalError("correction_multiplier: duplicate x keys")
    x = keys["x"].to_numpy(dtype="float64", na_value=np.nan)
    mult = np.where(np.isnan(x), 1.0, 1.0 + beta * np.nan_to_num(x))
    if not (np.isfinite(mult).all() and (mult > 0).all()):
        raise ExternalError("correction_multiplier: non-positive or non-finite multiplier")
    return mult


def fit_frame(group_oof: pd.DataFrame, weights: pd.Series, xt: pd.DataFrame, cfg: EcmwfConfig) -> pd.DataFrame:
    """그룹 OOF(group, ts, actual_kwh, pred_bag)에 가중치와 x를 붙여 β 적합 행을 만든다(x 있는 블록, target_start 이후)."""
    need = {"group", "ts", "actual_kwh", "pred_bag"}
    if not need <= set(group_oof.columns):
        raise ExternalError("fit_frame: missing columns")
    h = group_oof.assign(
        w=group_oof["ts"].map(weights), op=op_day(group_oof["ts"]), blk=block_index(group_oof["ts"], cfg)
    )
    if h["w"].isna().any():
        raise ExternalError("fit_frame: missing weight")
    h = h.dropna(subset=["blk"]).astype({"blk": "int64"}).merge(xt[["op", "blk", "x"]], on=["op", "blk"], how="inner")
    return h[h["op"] >= pd.Timestamp(cfg.target_start)].reset_index(drop=True)


# ---- 미러 동등성 ----

_EQUIV_COLUMNS = [
    "run",
    "target_date",
    "key",
    "origin_lastmod",
    "origin_md5",
    "origin_size",
    "mirror_lastmod",
    "mirror_md5",
    "mirror_size",
    "match",
]
_FIELD = {name: re.compile(rf"<{name}>([^<]*)</{name}>") for name in ("Key", "LastModified", "ETag", "Size")}


def mirror_equiv_path(paths: DataPaths) -> Path:
    """원본·미러 파일 동등성 대조 CSV 경로."""
    return paths.external_dir / "ecmwf" / "mirror_equiv.csv"


def run_prefix(run: EcmwfRunSpec, day: pd.Timestamp, cfg: EcmwfConfig) -> str:
    """대상일 실행 파일들의 버킷 공통 접두어."""
    init = run_init(day, run)
    return f"{init:%Y%m%d}/{init:%H}z/{run.product}/{cfg.resolution}/{run.stream}/{init:%Y%m%d%H}0000-"


def mirror_keys(run: EcmwfRunSpec, day: pd.Timestamp, cfg: EcmwfConfig) -> list[str]:
    """대상일 보정에 쓰는 리드별 grib2·index 파일 키."""
    prefix = run_prefix(run, day, cfg)
    return [f"{prefix}{lead}h-{run.stream}-fc.{ext}" for lead in needed_leads(run, cfg) for ext in ("grib2", "index")]


def parse_listing(xml: str) -> dict[str, tuple[str, str, int]]:
    """S3·GCS 목록 XML에서 키별 (LastModified, MD5 ETag, 크기)를 읽는다. 필드가 빠진 항목은 ExternalError."""
    import html

    out: dict[str, tuple[str, str, int]] = {}
    for block in xml.split("<Contents>")[1:]:
        found = {name: pat.search(block) for name, pat in _FIELD.items()}
        missing = [name for name, m in found.items() if m is None]
        if missing:
            raise ExternalError(f"listing entry missing {missing}")
        key, lastmod, etag, size = (found[n].group(1) for n in ("Key", "LastModified", "ETag", "Size"))  # type: ignore[union-attr]
        out[html.unescape(key)] = (lastmod, html.unescape(etag).strip('"'), int(size))
    return out


def compare_mirror(
    run: EcmwfRunSpec,
    day: pd.Timestamp,
    keys: Sequence[str],
    origin: Mapping[str, tuple[str, str, int]],
    mirror: Mapping[str, tuple[str, str, int]],
) -> list[dict[str, Any]]:
    """키마다 원본·미러 메타데이터를 나란히 두고, 둘 다 있고 MD5·크기가 같을 때만 match=True."""
    rows = []
    for key in keys:
        a, b = origin.get(key), mirror.get(key)
        rows.append(
            {
                "run": run.name,
                "target_date": f"{day:%Y-%m-%d}",
                "key": key,
                "origin_lastmod": a[0] if a else "",
                "origin_md5": a[1] if a else "",
                "origin_size": a[2] if a else None,
                "mirror_lastmod": b[0] if b else "",
                "mirror_md5": b[1] if b else "",
                "mirror_size": b[2] if b else None,
                "match": bool(a and b and a[1] == b[1] and a[2] == b[2]),
            }
        )
    return rows


# ---- 수집·감사(네트워크) ----


class _NotFoundError(ExternalError):
    """원격 파일 없음(404)."""


def _get(url: str, byte_range: tuple[int, int] | None = None, retries: int = 40) -> bytes:  # pragma: no cover
    import urllib.error
    import urllib.request

    if not url.startswith("https://"):
        raise ExternalError("only https sources are allowed")
    headers = {"Range": f"bytes={byte_range[0]}-{byte_range[1]}"} if byte_range else {}
    err: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)  # noqa: S310 - 위에서 https만 허용
            with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 - https만 허용
                body: bytes = resp.read()
            if byte_range and len(body) != byte_range[1] - byte_range[0] + 1:
                raise ExternalError(f"short range read for {url}")
            return body
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise _NotFoundError(f"not found: {url}") from exc
            err = exc
        except (OSError, ExternalError) as exc:
            err = exc
        # S3는 GET에 503(SlowDown)을 자주 돌려주므로 짧게 여러 번 재시도한다
        time.sleep(min(8.0, 1.0 + attempt))
    raise ExternalError(f"request failed after {retries} tries: {url}: {err!r}")


def _stem(base: str, init: pd.Timestamp, run: EcmwfRunSpec, lead: int, cfg: EcmwfConfig) -> str:  # pragma: no cover
    return (
        f"{base}/{init:%Y%m%d}/{init:%H}z/{run.product}/{cfg.resolution}/{run.stream}/{init:%Y%m%d%H}0000-{lead}h-{run.stream}-fc"
    )


def fetch_target(
    paths: DataPaths, cfg: EcmwfConfig, run: EcmwfRunSpec, day: pd.Timestamp
) -> str | None:  # pragma: no cover
    """대상일 하나의 실행 ssrd를 상자로 잘라 npz로 저장한다(있으면 건너뜀). 실패 시 오류 문자열, 성공 시 None."""
    from src.oibc.data.external_gfs import _decode_grib

    path = run_dir(paths, run) / f"{day:%Y%m%d}.npz"
    if path.exists():
        return None
    init = run_init(day, run)
    box = ExternalGfsConfig(box_lat=cfg.box_lat, box_lon=cfg.box_lon)
    out: dict[str, np.ndarray] = {}
    lat = lon = np.empty(0)
    try:
        for lead in needed_leads(run, cfg):
            try:
                stem = _stem(cfg.mirror_url, init, run, lead, cfg)
                index = _get(stem + ".index")
            except _NotFoundError:
                stem = _stem(cfg.s3_url, init, run, lead, cfg)
                index = _get(stem + ".index")
            rows = [json.loads(line) for line in index.decode().splitlines() if line.strip()]
            hit = [r for r in rows if r.get("param") == cfg.param]
            if len(hit) != 1:
                raise ExternalError(f"{cfg.param} messages={len(hit)} lead={lead}")
            off, length = int(hit[0]["_offset"]), int(hit[0]["_length"])
            values, glat, glon = _decode_grib(_get(stem + ".grib2", (off, off + length - 1)))
            cropped, lat, lon = crop_box(values, glat, glon, box)
            out[f"{cfg.param}_{lead}"] = cropped.astype("float32")
    except ExternalError as exc:
        return f"{run.name} {day:%Y-%m-%d}: {exc}"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    payload: dict[str, Any] = {"lat": lat, "lon": lon, **out}
    np.savez_compressed(tmp, **payload)
    tmp.replace(path)
    return None


def fetch_ecmwf(
    paths: DataPaths, cfg: EcmwfConfig, runs: Sequence[str] | None = None, workers: int = 8
) -> int:  # pragma: no cover
    """설정 기간의 모든 실행·대상일을 받는다(이미 있는 파일은 유지). 실패 건수를 반환한다."""
    selected = [r for r in all_runs(cfg) if runs is None or r.name in runs]
    jobs = [(r, d) for r in selected for d in target_dates(cfg, r)]
    fails = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for k, err in enumerate(pool.map(lambda j: fetch_target(paths, cfg, j[0], j[1]), jobs), 1):
            if err:
                fails += 1
                logger.warning("[DATA] ecmwf fetch fail %s", err)
            if k % 100 == 0 or k == len(jobs):
                logger.info("[DATA] ecmwf fetch progress=%d/%d fails=%d", k, len(jobs), fails)
    return fails


def published_time(cfg: EcmwfConfig, run: EcmwfRunSpec, day: pd.Timestamp) -> str:  # pragma: no cover
    """S3 원본 목록의 LastModified로 필요한 리드 파일 중 가장 늦은 공개 시각(ISO, UTC)을 반환한다. 하나라도 없으면 빈 문자열."""
    init = run_init(day, run)
    prefix = f"{init:%Y%m%d}/{init:%H}z/{run.product}/{cfg.resolution}/{run.stream}/"
    text = _get(f"{cfg.s3_url}/?list-type=2&max-keys=1000&prefix={prefix}").decode()
    if "<IsTruncated>true</IsTruncated>" in text:
        raise ExternalError(f"ecmwf audit: truncated listing {prefix}")
    stamps = dict(re.findall(r"<Key>([^<]*)</Key><LastModified>([^<]*)</LastModified>", text))
    want = [f"{prefix}{init:%Y%m%d%H}0000-{lead}h-{run.stream}-fc.grib2" for lead in needed_leads(run, cfg)]
    if any(k not in stamps for k in want):
        return ""
    latest: pd.Timestamp = max(pd.Timestamp(stamps[k]) for k in want)
    return str(latest.isoformat())


def audit_ecmwf(paths: DataPaths, cfg: EcmwfConfig, workers: int = 8) -> list[Path]:  # pragma: no cover
    """실행별 공개 시각 감사 CSV를 만든다(덮어씀)."""
    out = []
    for run in all_runs(cfg):
        days = list(target_dates(cfg, run))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pubs = list(pool.map(lambda d, r=run: published_time(cfg, r, d), days))
        frame = pd.DataFrame(
            {
                "target_date": [f"{d:%Y-%m-%d}" for d in days],
                "init": [f"{run_init(d, run):%Y%m%d%H}" for d in days],
                "published_utc": pubs,
            }
        )
        path = audit_path(paths, run)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        logger.info(
            "[DATA] ecmwf audit run=%s days=%d listed=%d",
            run.name,
            len(days),
            int((frame["published_utc"] != "").sum()),
        )
        out.append(path)
    return out


def _list_files(base: str, prefix: str) -> dict[str, tuple[str, str, int]]:  # pragma: no cover
    out: dict[str, tuple[str, str, int]] = {}
    token = ""
    while True:
        if "amazonaws" in base:
            url = f"{base}/?list-type=2&max-keys=1000&prefix={prefix}" + (f"&continuation-token={token}" if token else "")
        else:
            url = f"{base}?prefix={prefix}" + (f"&marker={token}" if token else "")
        text = _get(url).decode()
        page = parse_listing(text)
        out.update(page)
        if "<IsTruncated>true</IsTruncated>" not in text:
            return out
        nxt = re.search(r"<Next(?:ContinuationToken|Marker)>([^<]*)</Next", text)
        token = nxt.group(1) if nxt else max(page)


def audit_mirror(paths: DataPaths, cfg: EcmwfConfig, workers: int = 6) -> Path:  # pragma: no cover
    """실행·대상일마다 사용 파일의 원본(S3)·미러(GCS) MD5·크기·LastModified를 목록 API로 대조해 CSV로 남긴다(덮어씀)."""

    def one(job: tuple[EcmwfRunSpec, pd.Timestamp]) -> list[dict[str, Any]]:
        run, day = job
        prefix = run_prefix(run, day, cfg)
        origin, mirror = _list_files(cfg.s3_url, prefix), _list_files(cfg.mirror_url, prefix)
        return compare_mirror(run, day, mirror_keys(run, day, cfg), origin, mirror)

    jobs = [(run, day) for run in all_runs(cfg) for day in target_dates(cfg, run)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        frame = pd.DataFrame([row for rows in pool.map(one, jobs) for row in rows], columns=_EQUIV_COLUMNS)
    path = mirror_equiv_path(paths)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    both = (frame["origin_md5"] != "") & (frame["mirror_md5"] != "")
    logger.info(
        "[DATA] ecmwf mirror equiv files=%d both=%d match=%d mismatch=%d origin_only=%d mirror_only=%d",
        len(frame),
        int(both.sum()),
        int(frame["match"].sum()),
        int((both & ~frame["match"]).sum()),
        int(((frame["origin_md5"] != "") & (frame["mirror_md5"] == "")).sum()),
        int(((frame["origin_md5"] == "") & (frame["mirror_md5"] != "")).sum()),
    )
    return path
