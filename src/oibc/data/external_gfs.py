"""외부 GFS 예보의 한국 광역 요약: 수집, 공개 시각 감사, 대상일 시간별 표 구축.

모든 발전소에 같은 값으로 붙는 고정 상자 요약만 만든다(발전소 좌표 미사용). 위반 시 ExternalError로 중단한다.
"""

from __future__ import annotations

import logging
import os
import re
import time
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from src.oibc.config import CompetitionSpec, DataPaths, ExternalGfsConfig, FeatureConfig
from src.oibc.timeaxis import op_date

logger = logging.getLogger(__name__)

_KST = "Asia/Seoul"
_AUDIT_COLUMNS = ["init", "last_lead_h", "published_utc"]


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


# --- 네트워크 수집(재현 절차). 단위 테스트 대상 아님 ---


def fetch_gfs(paths: DataPaths, features: FeatureConfig, workers: int = 16) -> int:  # pragma: no cover
    """dynamical.org GFS(CC-BY-4.0)에서 상자·리드 0~max_lead_h를 init별 npz로 받는다. 없는 변수만 받아 병합한다."""
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


def _npz_variables(path: Path) -> set[str]:  # pragma: no cover
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
