"""외부 GFS 예보의 한국 광역 요약: 수집, 공개 시각 감사, 대상일 시간별 표 구축.

모든 발전소에 같은 값으로 붙는 고정 상자 요약만 만든다(발전소 좌표 미사용). 위반 시 ExternalError로 중단한다.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from src.oibc.config import CompetitionSpec, DataPaths, ExternalGfsConfig, FeatureConfig, NoaaGfsConfig, NoaaStat
from src.oibc.timeaxis import op_date

logger = logging.getLogger(__name__)

_KST = "Asia/Seoul"
_AUDIT_COLUMNS = ["init", "last_lead_h", "published_utc"]
_GRID_DEG = 0.25  # NOAA `pgrb2.0p25` 제품 격자 간격
_INIT_ID = re.compile(r"^\d{10}$")
_GRIB_LOCK = threading.Lock()  # ecCodes 빌드가 스레드 안전하다는 보장이 없어 해독만 직렬화한다


class ExternalError(ValueError):
    """외부 GFS 입력 계약 위반."""


def gfs_dir(paths: DataPaths, run_hour: int) -> Path:
    """실행 시각별 init 파일 디렉터리."""
    return paths.external_dir / "gfs" / f"{run_hour:02d}z"


def audit_path(paths: DataPaths) -> Path:
    """NOAA 원본 공개 시각 감사 CSV 경로."""
    return paths.external_dir / "gfs" / "pubtime_audit.csv"


def hourly_path(paths: DataPaths) -> Path:
    """대상일 시간별 GFS 광역 요약 표 경로."""
    return paths.derived_dir / "gfs_hourly.parquet"


def target_window(init: pd.Timestamp) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """init(UTC)이 발행된 KST 날짜를 D-1로 보고, 대상일 D의 KST 01:00~익일 00:00 시각과 리드(h)를 반환한다."""
    if init.tzinfo is None or str(init.tz) != "UTC" or init != init.floor("h"):
        raise ExternalError("init must be tz-aware UTC on the hour")
    d_minus_1 = init.tz_convert(_KST).normalize()
    ts = pd.date_range(d_minus_1 + pd.Timedelta(hours=25), periods=24, freq="h")
    leads = ((ts - init) / pd.Timedelta(hours=1)).to_numpy().astype("int64")
    return ts, leads


def last_used_lead(run_hour: int) -> int:
    """해당 실행에서 대상일에 쓰이는 마지막 리드(h)."""
    init = pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(hours=run_hour)
    return int(target_window(init)[1][-1])


def deaverage(values: np.ndarray, leads: np.ndarray, init_hour: int) -> np.ndarray:
    """6시간마다 초기화되는 '직전 1~6시간 평균' 배열(리드 축 0)에서 리드별 직전 1시간 평균을 복원한다."""
    if init_hour % 6 != 0:
        raise ExternalError("avg6 de-averaging needs a 6-hourly init")
    if (leads < 1).any() or (leads >= values.shape[0]).any():
        raise ExternalError("leads out of range for de-averaging")
    n = ((leads - 1) % 6 + 1).astype("float64").reshape((-1,) + (1,) * (values.ndim - 1))
    cur = values[leads].astype("float64")
    prev = values[leads - 1].astype("float64")
    return np.asarray(n * cur - (n - 1.0) * prev)


def summarize_init(
    arrays: Mapping[str, np.ndarray], lat: np.ndarray, lon: np.ndarray, init: pd.Timestamp, cfg: ExternalGfsConfig
) -> pd.DataFrame:
    """init 하나의 (리드, 위도, 경도) 배열을 대상일 24시간의 광역 요약 행으로 바꾼다."""
    ts, leads = target_window(init)
    in_mask = ((lat >= cfg.inner_lat[0]) & (lat <= cfg.inner_lat[1]))[:, None] & (
        (lon >= cfg.inner_lon[0]) & (lon <= cfg.inner_lon[1])
    )[None, :]
    if not in_mask.any():
        raise ExternalError("inner box selects no grid cell")
    out: dict[str, object] = {"ts": ts, "init": init}
    prefix = f"gfs{init.hour:02d}"
    for name, (source, kind) in cfg.variables.items():
        if source not in arrays:
            raise ExternalError(f"init {init:%Y%m%d%H}: missing variable {source!r}")
        values = np.asarray(arrays[source])
        if values.ndim != 3 or values.shape[1:] != (lat.size, lon.size):
            raise ExternalError(f"init {init:%Y%m%d%H}: bad shape for {source!r}")
        hourly = deaverage(values, leads, init.hour) if kind == "avg6" else values[leads].astype("float64")
        inner = hourly[:, in_mask]
        out[f"{prefix}_{name}_box"] = hourly.mean(axis=(1, 2))
        out[f"{prefix}_{name}_in"] = inner.mean(axis=1)
        out[f"{prefix}_{name}_insd"] = inner.std(axis=1)
    return pd.DataFrame(out)


def read_audit(path: Path) -> pd.Series:
    """감사 CSV를 init(UTC) → 공개 시각(UTC) Series로 읽는다."""
    if not path.exists():
        raise ExternalError(f"publication audit missing: {path}")
    frame = pd.read_csv(path, dtype={"init": str})
    if list(frame.columns) != _AUDIT_COLUMNS:
        raise ExternalError("publication audit: unexpected columns")
    init = pd.to_datetime(frame["init"], format="%Y%m%d%H").dt.tz_localize("UTC")
    published = pd.to_datetime(frame["published_utc"], utc=True)
    if init.duplicated().any():
        raise ExternalError("publication audit: duplicate init")
    return pd.Series(published.to_numpy(), index=pd.DatetimeIndex(init), name="published_utc")


def is_published_before_cutoff(init: pd.Timestamp, published: pd.Timestamp, spec: CompetitionSpec) -> bool:
    """init 결과가 대상일 D의 예측시점(D-1 17:00 KST) 이전에 공개되었는지 반환한다."""
    ts, _ = target_window(init)
    cutoff = ts[0].normalize() - pd.Timedelta(days=1) + pd.Timedelta(hours=spec.cutoff_hour)
    return bool(published <= cutoff)


def add_derived(table: pd.DataFrame, cfg: ExternalGfsConfig, day_hours: tuple[int, int]) -> pd.DataFrame:
    """같은 운영일 안에서만 이웃 시각(±1h)과 주간 평균 특성을 붙인다."""
    table = table.sort_values("ts").reset_index(drop=True)
    day = op_date(table["ts"])
    hour = table["ts"].dt.hour
    daytime = (hour >= day_hours[0]) & (hour <= day_hours[1])
    for h in cfg.run_hours:
        for name in cfg.neighbor_vars:
            col = f"gfs{h:02d}_{name}_in"
            for off in (-1, 1):
                shifted = table[col].shift(-off)
                same_day = day.shift(-off) == day
                table[f"{col}_n{off:+d}"] = shifted.where(same_day)
        for name in cfg.daymean_vars:
            col = f"gfs{h:02d}_{name}_in"
            table[f"{col}_daymean"] = table[col].where(daytime).groupby(day).transform("mean")
    return table


def build_gfs_hourly(paths: DataPaths, spec: CompetitionSpec, features: FeatureConfig) -> Path:
    """init 파일과 공개 시각 감사로 대상일 시간별 광역 요약 표를 만든다. 마감 이후 공개된 init은 NaN으로 둔다."""
    cfg = features.gfs
    audit = read_audit(audit_path(paths))
    frames: list[pd.DataFrame] = []
    for h in cfg.run_hours:
        inits = _expected_inits(cfg, h)
        late = 0
        rows: list[pd.DataFrame] = []
        for init in inits:
            path = gfs_dir(paths, h) / f"{init:%Y%m%d%H}.npz"
            if not path.exists():
                raise ExternalError(f"missing init file {path.name} ({h:02d}z)")
            if init not in audit.index:
                raise ExternalError(f"publication audit has no row for {init:%Y%m%d%H}")
            with np.load(path) as z:
                arrays = {src: z[src] for src, _ in cfg.variables.values() if src in z}
                lat, lon = z["lat"], z["lon"]
                lead_h = z["lead_h"]
            if not np.array_equal(lead_h, np.arange(lead_h.size)):
                raise ExternalError(f"init {init:%Y%m%d%H}: lead axis must be 0..N hourly")
            summary = summarize_init(arrays, lat, lon, init, cfg)
            if not is_published_before_cutoff(init, audit[init], spec):
                summary.loc[:, [c for c in summary.columns if c not in ("ts", "init")]] = np.nan
                late += 1
            rows.append(summary)
        run = pd.concat(rows, ignore_index=True).drop(columns="init")
        logger.info("[DATA] gfs run=%02dz inits=%d late_nan=%d rows=%d", h, len(inits), late, len(run))
        frames.append(run.set_index("ts"))
    table = pd.concat(frames, axis=1, join="outer").reset_index()
    table["ts"] = table["ts"].dt.tz_convert(_KST).astype(f"datetime64[us, {_KST}]")
    table = add_derived(table, cfg, features.day_hours)
    cols = cfg.feature_columns()
    table = table[["ts", *cols]]
    table[cols] = table[cols].astype("float32")
    _require_eval_coverage(table, spec, cols)
    out = hourly_path(paths)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out, index=False)
    logger.info("[DATA] gfs hourly rows=%d cols=%d -> %s", len(table), len(cols), out)
    return out


def _expected_inits(cfg: ExternalGfsConfig, run_hour: int) -> pd.DatetimeIndex:
    days = pd.date_range(cfg.init_start_date, cfg.init_end_date, freq="D", tz="UTC")
    return days + pd.Timedelta(hours=run_hour)


def _require_eval_coverage(table: pd.DataFrame, spec: CompetitionSpec, cols: list[str]) -> None:
    eval_ts = pd.date_range(spec.eval_start, spec.eval_end, freq="h")
    have = pd.DatetimeIndex(table["ts"])
    if not eval_ts.isin(have).all():
        raise ExternalError("gfs hourly does not cover every evaluation hour")


def noaa_fetch_leads(run_hour: int) -> np.ndarray:
    """NOAA 원본에서 받을 리드: 대상일 리드와 평균 풀기에 필요한 직전 리드(00z는 15..39)."""
    init = pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(hours=run_hour)
    leads = target_window(init)[1]
    return np.arange(int(leads.min()) - 1, int(leads.max()) + 1, dtype="int64")


def parse_idx(text: str) -> list[tuple[int, int | None, str]]:
    """NOAA `.idx`를 (시작 바이트, 다음 메시지 시작 또는 None, "VAR:LEVEL:STEP") 목록으로 바꾼다. 형식 위반 시 ExternalError."""
    starts: list[int] = []
    descs: list[str] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split(":")
        if len(parts) < 6 or not parts[1].isdigit() or not parts[2].startswith("d="):
            raise ExternalError(f"idx: malformed line {line!r}")
        start = int(parts[1])
        if starts and start <= starts[-1]:
            raise ExternalError("idx: byte offsets must increase")
        starts.append(start)
        descs.append(":".join(parts[3:6]))
    if not starts:
        raise ExternalError("idx: empty")
    ends: list[int | None] = [*starts[1:], None]
    return list(zip(starts, ends, descs, strict=True))


def select_message(
    records: Sequence[tuple[int, int | None, str]], short: str, level: str, stat: NoaaStat, lead: int
) -> tuple[int, int]:
    """필드·리드에 맞는 메시지 하나의 포함 바이트 범위. ave는 6시간 리셋 창만 인정한다. 0개·복수·마지막 메시지면 ExternalError."""
    step = f"{((lead - 1) // 6) * 6}-{lead} hour ave fcst" if stat == "ave" else f"{lead} hour fcst"
    want = f"{short}:{level}:{step}"
    hits = [(start, end) for start, end, desc in records if desc == want]
    if len(hits) != 1:
        raise ExternalError(f"idx: {len(hits)} match(es) for {want!r}")
    start, end = hits[0]
    if end is None:
        raise ExternalError(f"idx: {want!r} is the last message (unbounded range)")
    return start, end - 1


def crop_box(
    values: np.ndarray, lat: np.ndarray, lon: np.ndarray, cfg: ExternalGfsConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """전구 2-D 필드를 고정 상자로 자른다(위도 북→남, 경도 서→동, float64). 기대 0.25° 격자와 다르면 ExternalError."""
    lat = np.asarray(lat, dtype="float64")
    lon = (np.asarray(lon, dtype="float64") + 180.0) % 360.0 - 180.0
    values = np.asarray(values, dtype="float64")
    if values.shape != (lat.size, lon.size):
        raise ExternalError("crop_box: values shape must be (lat, lon)")
    yi = np.where((lat >= cfg.box_lat[0]) & (lat <= cfg.box_lat[1]))[0]
    xi = np.where((lon >= cfg.box_lon[0]) & (lon <= cfg.box_lon[1]))[0]
    yi = yi[np.argsort(-lat[yi], kind="stable")]
    xi = xi[np.argsort(lon[xi], kind="stable")]
    exp_lat = np.round(np.arange(cfg.box_lat[1], cfg.box_lat[0] - _GRID_DEG / 2, -_GRID_DEG), 6)
    exp_lon = np.round(np.arange(cfg.box_lon[0], cfg.box_lon[1] + _GRID_DEG / 2, _GRID_DEG), 6)
    if yi.size != exp_lat.size or xi.size != exp_lon.size:
        raise ExternalError("crop_box: grid does not match the expected 0.25 deg box")
    if not (np.allclose(lat[yi], exp_lat) and np.allclose(lon[xi], exp_lon)):
        raise ExternalError("crop_box: grid does not match the expected 0.25 deg box")
    return values[np.ix_(yi, xi)], lat[yi], lon[xi]


def assemble_noaa_arrays(
    raw: Mapping[str, Mapping[int, np.ndarray]], init: pd.Timestamp, ext: ExternalGfsConfig, noaa: NoaaGfsConfig
) -> dict[str, np.ndarray]:
    """리드별 상자 배열을 dynamical npz와 같은 (max_lead_h+1, ny, nx) float32로 조립한다. 받지 않은 리드는 NaN, 누락은 ExternalError."""
    fetch = noaa_fetch_leads(init.hour)
    if int(fetch.max()) > ext.max_lead_h:
        raise ExternalError("max_lead_h is shorter than the fetch leads")
    used = target_window(init)[1]
    shape: tuple[int, ...] | None = None
    out: dict[str, np.ndarray] = {}
    for src, _kind in ext.variables.values():
        if src not in noaa.fields:
            raise ExternalError(f"no NOAA field mapping for {src!r}")
        by_lead = raw.get(src)
        if by_lead is None:
            raise ExternalError(f"init {init:%Y%m%d%H}: missing NOAA variable {src!r}")
        missing = [int(lead) for lead in fetch if int(lead) not in by_lead]
        if missing:
            raise ExternalError(f"init {init:%Y%m%d%H}: {src!r} missing leads {missing}")
        offset = noaa.fields[src][3]
        arr: np.ndarray | None = None
        for lead in fetch:
            grid = np.asarray(by_lead[int(lead)], dtype="float64")
            if shape is None:
                shape = grid.shape
            if grid.ndim != 2 or grid.shape != shape:
                raise ExternalError(f"init {init:%Y%m%d%H}: inconsistent grid shape for {src!r}")
            if arr is None:
                arr = np.full((ext.max_lead_h + 1, *shape), np.nan)
            arr[int(lead)] = grid + offset
        assert arr is not None
        if src in noaa.hourly_from_ave:
            hourly = np.full_like(arr, np.nan)
            hourly[used] = deaverage(arr, used, init.hour)
            arr = hourly
        out[src] = arr.astype("float32")
    return out


def compare_summaries(
    noaa: pd.DataFrame, dyn: pd.DataFrame, ext: ExternalGfsConfig, noaa_cfg: NoaaGfsConfig
) -> pd.DataFrame:
    """두 `summarize_init` 출력의 열별 최대 |차이|와 허용오차 판정(column, max_abs_diff, tol, ok). ts·열 불일치는 ExternalError."""
    if not noaa["ts"].reset_index(drop=True).equals(dyn["ts"].reset_index(drop=True)):
        raise ExternalError("compare: ts mismatch")
    cols = [c for c in dyn.columns if c not in ("ts", "init")]
    if set(cols) != {c for c in noaa.columns if c not in ("ts", "init")}:
        raise ExternalError("compare: column mismatch")
    rows: list[dict[str, object]] = []
    for col in cols:
        head, stat = col.rsplit("_", 1)
        name = head.split("_", 1)[1]
        if name not in noaa_cfg.equiv_tol or name not in ext.variables:
            raise ExternalError(f"compare: no tolerance for {col!r}")
        tol = noaa_cfg.equiv_tol[name] * (noaa_cfg.insd_tol_factor if stat == "insd" else 1.0)
        a = noaa[col].to_numpy(dtype="float64")
        b = dyn[col].to_numpy(dtype="float64")
        na, nb = np.isnan(a), np.isnan(b)
        both = ~na & ~nb
        diff = float(np.abs(a[both] - b[both]).max()) if both.any() else 0.0
        ok = bool(not (na ^ nb).any() and diff <= tol)
        rows.append({"column": col, "max_abs_diff": diff, "tol": tol, "ok": ok})
    return pd.DataFrame(rows, columns=["column", "max_abs_diff", "tol", "ok"])


def noaa_pending_inits(paths: DataPaths, ext: ExternalGfsConfig, run_hour: int) -> list[pd.Timestamp]:
    """npz 파일이 없는 기대 init만 반환한다. 변수가 빠진 기존 파일은 원천 혼합을 막기 위해 ExternalError."""
    need = {src for src, _ in ext.variables.values()}
    pending: list[pd.Timestamp] = []
    for init in _expected_inits(ext, run_hour):
        path = gfs_dir(paths, run_hour) / f"{init:%Y%m%d%H}.npz"
        if not path.exists():
            pending.append(init)
        elif not need <= _npz_variables(path):
            raise ExternalError(f"{path.name}: partial file (missing variables); refusing to mix sources")
    return pending


def parse_init_id(value: str) -> pd.Timestamp:
    """`YYYYMMDDHH`(UTC) 문자열을 tz-aware UTC init으로 바꾼다. 형식 위반 시 ExternalError."""
    if not _INIT_ID.match(value):
        raise ExternalError(f"init id must be YYYYMMDDHH: {value!r}")
    try:
        return pd.Timestamp(f"{value[:8]} {value[8:]}:00", tz="UTC")
    except ValueError as exc:
        raise ExternalError(f"invalid init id {value!r}") from exc


# --- 네트워크 수집(재현 절차). 단위 테스트 대상 아님 ---


def fetch_gfs(
    paths: DataPaths,
    features: FeatureConfig,
    workers: int = 16,
    source: Literal["dynamical", "noaa"] = "dynamical",
    noaa: NoaaGfsConfig | None = None,
) -> int:  # pragma: no cover
    """GFS 상자 npz를 init별로 받는다. dynamical(CC-BY-4.0)은 없는 변수만 병합하고, noaa(원본 GRIB)는 파일이 없는 init만 새로 쓴다."""
    if source == "noaa":
        return _fetch_gfs_noaa(paths, features.gfs, noaa or NoaaGfsConfig(), workers)
    from multiprocessing import Pool

    cfg = features.gfs
    pending: list[tuple[int, pd.Timestamp, tuple[str, ...]]] = []
    for h in cfg.run_hours:
        gfs_dir(paths, h).mkdir(parents=True, exist_ok=True)
        for init in _expected_inits(cfg, h):
            need = tuple(src for src, _ in cfg.variables.values() if src not in _npz_variables(gfs_dir(paths, h) / f"{init:%Y%m%d%H}.npz"))
            if need:
                pending.append((h, init, need))
    logger.info("[DATA] gfs fetch pending=%d workers=%d", len(pending), workers)
    if not pending:
        return 0
    t0 = time.perf_counter()
    fails = 0
    with Pool(workers, initializer=_worker_open, initargs=(cfg.model_dump(mode="json"), str(paths.external_dir))) as pool:
        for k, err in enumerate(pool.imap_unordered(_worker_fetch, pending), start=1):
            if err:
                fails += 1
                logger.error("[DATA] gfs fetch failed %s", err)
            if k % 100 == 0 or k == len(pending):
                logger.info("[DATA] gfs fetched=%d/%d fails=%d sec=%.0f", k, len(pending), fails, time.perf_counter() - t0)
    if fails:
        raise ExternalError(f"gfs fetch: {fails} init(s) failed; rerun to resume")
    return len(pending)


def audit_publication(paths: DataPaths, features: FeatureConfig, workers: int = 6) -> Path:  # pragma: no cover
    """NOAA 원본 S3의 마지막 사용 리드 파일 LastModified를 init별로 기록한다."""
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor

    cfg = features.gfs

    def one(init: pd.Timestamp) -> tuple[str, int, str]:
        lead = last_used_lead(init.hour)
        stem = f"gfs.{init:%Y%m%d}/{init:%H}"
        for sub in ("atmos/", ""):
            key = f"{stem}/{sub}gfs.t{init:%H}z.pgrb2.0p25.f{lead:03d}"
            url = f"{cfg.noaa_bucket_url}/?list-type=2&prefix={key}&max-keys=1"
            for attempt in range(5):
                try:
                    with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310 - 설정에서 https만 허용
                        body = resp.read().decode()
                    break
                except OSError:
                    time.sleep(2 * (attempt + 1))
            else:
                raise ExternalError(f"audit request failed for {key}")
            m = re.search(r"<LastModified>([^<]+)</LastModified>", body)
            if m:
                return f"{init:%Y%m%d%H}", lead, m.group(1)
        raise ExternalError(f"NOAA object not found for {init:%Y%m%d%H}")

    inits = [i for h in cfg.run_hours for i in _expected_inits(cfg, h)]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(one, inits))
    out = audit_path(paths)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=_AUDIT_COLUMNS).sort_values("init").to_csv(out, index=False)
    logger.info("[DATA] gfs audit rows=%d -> %s", len(rows), out)
    return out


def _npz_variables(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with np.load(path) as z:
        return set(z.files)


_BOX = None
_OUT_ROOT: Path | None = None


def _worker_open(cfg_json: dict[str, object], external_dir: str) -> None:  # pragma: no cover
    import icechunk
    import xarray as xr

    global _BOX, _OUT_ROOT
    cfg = ExternalGfsConfig.model_validate(cfg_json)
    storage = icechunk.s3_storage(
        bucket=cfg.icechunk_bucket, prefix=cfg.icechunk_prefix, region=cfg.icechunk_region, anonymous=True
    )
    ds = xr.open_zarr(icechunk.Repository.open(storage).readonly_session("main").store, chunks=None)
    _BOX = ds.sel(latitude=slice(cfg.box_lat[1], cfg.box_lat[0]), longitude=slice(cfg.box_lon[0], cfg.box_lon[1])).isel(
        lead_time=slice(0, cfg.max_lead_h + 1)
    )
    _OUT_ROOT = Path(external_dir) / "gfs"


def _worker_fetch(job: tuple[int, pd.Timestamp, Iterable[str]]) -> str | None:  # pragma: no cover
    h, init, need = job
    if _BOX is None or _OUT_ROOT is None:
        raise ExternalError("fetch worker not initialized")
    path = _OUT_ROOT / f"{h:02d}z" / f"{init:%Y%m%d%H}.npz"
    err: Exception | None = None
    for attempt in range(5):
        try:
            sel = _BOX[list(need)].sel(init_time=init.tz_convert(None)).load()
            merged: dict[str, np.ndarray] = {}
            if path.exists():
                with np.load(path) as z:
                    merged = {k: z[k] for k in z.files}
            merged.update({v: sel[v].to_numpy().astype("float32") for v in need})
            merged["lead_h"] = (sel["lead_time"].to_numpy() / np.timedelta64(1, "h")).astype("int16")
            merged["lat"] = sel["latitude"].to_numpy()
            merged["lon"] = sel["longitude"].to_numpy()
            tmp = path.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, **merged)  # type: ignore[arg-type]  # numpy 스텁이 allow_pickle 키와 충돌
            os.replace(tmp, path)
            return None
        except Exception as exc:  # 네트워크 일시 오류는 재시도
            err = exc
            time.sleep(5 * (attempt + 1))
    return f"{init:%Y%m%d%H}: {err!r}"


def _http_get(url: str, byte_range: tuple[int, int] | None = None, retries: int = 6) -> bytes:  # pragma: no cover
    import urllib.error
    import urllib.request

    if not url.startswith("https://"):
        raise ExternalError("only https sources are allowed")
    headers = {"Range": f"bytes={byte_range[0]}-{byte_range[1]}"} if byte_range else {}
    err: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)  # noqa: S310 - 위에서 https만 허용
            with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310 - https만 허용
                body: bytes = resp.read()
            if byte_range and len(body) != byte_range[1] - byte_range[0] + 1:
                raise ExternalError(f"short range read for {url}")
            return body
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise ExternalError(f"not found: {url}") from exc
            err = exc
        except (OSError, ExternalError) as exc:
            err = exc
        time.sleep(min(60.0, 2.0**attempt))
    raise ExternalError(f"request failed after {retries} tries: {url}: {err!r}")


def _decode_grib(message: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:  # pragma: no cover
    import eccodes

    with _GRIB_LOCK:
        handle = eccodes.codes_new_from_message(message)
        try:
            ni, nj = eccodes.codes_get(handle, "Ni"), eccodes.codes_get(handle, "Nj")
            lat0 = eccodes.codes_get(handle, "latitudeOfFirstGridPointInDegrees")
            lon0 = eccodes.codes_get(handle, "longitudeOfFirstGridPointInDegrees")
            dlat = eccodes.codes_get(handle, "jDirectionIncrementInDegrees")
            dlon = eccodes.codes_get(handle, "iDirectionIncrementInDegrees")
            if eccodes.codes_get(handle, "iScansNegatively") != 0:
                raise ExternalError("unexpected GRIB scan order")
            north_to_south = eccodes.codes_get(handle, "jScansPositively") == 0
            values = np.asarray(eccodes.codes_get_values(handle), dtype="float64").reshape(nj, ni)
        finally:
            eccodes.codes_release(handle)
    lat = lat0 - dlat * np.arange(nj) if north_to_south else lat0 + dlat * np.arange(nj)
    lon = lon0 + dlon * np.arange(ni)
    return values, lat, lon


def noaa_init_arrays(
    init: pd.Timestamp, ext: ExternalGfsConfig, noaa: NoaaGfsConfig
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:  # pragma: no cover
    """NOAA 원본 GRIB에서 init 하나를 받아 dynamical npz와 같은 변수 배열과 상자 위경도를 반환한다."""
    sources = [src for src, _ in ext.variables.values()]
    raw: dict[str, dict[int, np.ndarray]] = {src: {} for src in sources}
    lat_box = lon_box = np.empty(0)
    for lead in noaa_fetch_leads(init.hour):
        stem = f"{ext.noaa_bucket_url}/gfs.{init:%Y%m%d}/{init:%H}/atmos/gfs.t{init:%H}z.pgrb2.0p25.f{int(lead):03d}"
        records = parse_idx(_http_get(stem + ".idx").decode())
        for src in sources:
            short, level, stat, _offset = noaa.fields[src]
            values, lat, lon = _decode_grib(_http_get(stem, select_message(records, short, level, stat, int(lead))))
            raw[src][int(lead)], lat_box, lon_box = crop_box(values, lat, lon, ext)
    return assemble_noaa_arrays(raw, init, ext, noaa), lat_box, lon_box


def _fetch_gfs_noaa(paths: DataPaths, ext: ExternalGfsConfig, noaa: NoaaGfsConfig, workers: int) -> int:  # pragma: no cover
    from concurrent.futures import ThreadPoolExecutor

    pending = [(h, init) for h in ext.run_hours for init in noaa_pending_inits(paths, ext, h)]
    logger.info("[DATA] gfs noaa fetch pending=%d workers=%d", len(pending), workers)
    if not pending:
        return 0

    def one(job: tuple[int, pd.Timestamp]) -> str | None:
        h, init = job
        try:
            arrays, lat, lon = noaa_init_arrays(init, ext, noaa)
            path = gfs_dir(paths, h) / f"{init:%Y%m%d%H}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp.npz")
            payload = {"lead_h": np.arange(ext.max_lead_h + 1, dtype="int16"), "lat": lat, "lon": lon, **arrays}
            np.savez_compressed(tmp, **payload)  # type: ignore[arg-type]  # numpy 스텁이 allow_pickle 키와 충돌
            os.replace(tmp, path)
            return None
        except Exception as exc:  # init 단위 실패는 모아서 보고하고 재실행으로 이어받는다
            return f"{init:%Y%m%d%H}: {exc!r}"

    t0 = time.perf_counter()
    fails = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for k, err in enumerate(ex.map(one, pending), start=1):
            if err:
                fails += 1
                logger.error("[DATA] gfs noaa fetch failed %s", err)
            if k % 50 == 0 or k == len(pending):
                logger.info("[DATA] gfs noaa fetched=%d/%d fails=%d sec=%.0f", k, len(pending), fails, time.perf_counter() - t0)
    if fails:
        raise ExternalError(f"gfs noaa fetch: {fails} init(s) failed; rerun to resume")
    return len(pending)


def compare_gfs_sources(
    paths: DataPaths,
    features: FeatureConfig,
    noaa: NoaaGfsConfig,
    inits: Sequence[pd.Timestamp] | None = None,
    workers: int = 8,
) -> Path:  # pragma: no cover
    """표본 init을 NOAA 원본(메모리)과 기존 npz로 각각 요약해 비교하고 `noaa_equiv.csv`를 쓴다. 허용오차 초과 시 ExternalError."""
    from concurrent.futures import ThreadPoolExecutor

    ext = features.gfs
    if inits is None:
        candidates = [i for h in ext.run_hours for i in _expected_inits(ext, h)]
        rng = np.random.default_rng(noaa.compare_seed)
        pick = np.sort(rng.choice(len(candidates), size=min(noaa.compare_sample, len(candidates)), replace=False))
        inits = [candidates[int(i)] for i in pick]
    for init in inits:
        if not (gfs_dir(paths, init.hour) / f"{init:%Y%m%d%H}.npz").exists():
            raise ExternalError(f"compare: no existing npz for {init:%Y%m%d%H}")
    logger.info("[DATA] gfs compare inits=%d workers=%d", len(inits), workers)

    def one(init: pd.Timestamp) -> pd.DataFrame:
        arrays, lat, lon = noaa_init_arrays(init, ext, noaa)
        with np.load(gfs_dir(paths, init.hour) / f"{init:%Y%m%d%H}.npz") as z:
            dyn_arrays = {src: z[src] for src, _ in ext.variables.values()}
            dyn_lat, dyn_lon = z["lat"], z["lon"]
        if not (np.allclose(lat, dyn_lat) and np.allclose(lon, dyn_lon)):
            raise ExternalError(f"compare: grid mismatch for {init:%Y%m%d%H}")
        table = compare_summaries(
            summarize_init(arrays, lat, lon, init, ext), summarize_init(dyn_arrays, dyn_lat, dyn_lon, init, ext), ext, noaa
        )
        table.insert(0, "init", f"{init:%Y%m%d%H}")
        return table

    with ThreadPoolExecutor(max_workers=workers) as ex:
        report = pd.concat(list(ex.map(one, inits)), ignore_index=True)
    out = paths.external_dir / "gfs" / "noaa_equiv.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(out, index=False)
    worst = report.groupby("column")["max_abs_diff"].max()
    logger.info("[DATA] gfs compare rows=%d fails=%d -> %s", len(report), int((~report["ok"]).sum()), out)
    logger.info("[DATA] gfs compare worst %s", " ".join(f"{c}={v:.3g}" for c, v in worst.items()))
    if not report["ok"].all():
        raise ExternalError(f"gfs compare: {int((~report['ok']).sum())} column(s) exceed tolerance; see {out}")
    return out
