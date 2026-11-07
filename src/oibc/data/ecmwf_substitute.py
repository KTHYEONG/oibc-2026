"""ECMWF 입력 대체: 평가 기간 GFS 일사 입력을 HRES·AIFS 결합 일사 쪽으로 옮긴 패널을 만든다.

블록마다 GFS 블록 평균 G가 G + λ·(E - G)가 되도록 시간별 값을 같은 비율로 곱한다(e87·e89 검증, LB sub18).
학습 행은 바꾸지 않는다. 고정 상자 요약만 쓰므로 모든 발전소에 같은 값이 들어간다.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.oibc.config import CompetitionSpec, DataPaths, EcmwfConfig, FeatureConfig
from src.oibc.data.external_ecmwf import aifs_window_ts_hours, block_index, op_day
from src.oibc.data.external_gfs import ExternalError, add_derived

logger = logging.getLogger(__name__)

_KST = "Asia/Seoul"


def _cols(features: FeatureConfig) -> tuple[str, str, list[str]]:
    run = f"gfs{features.gfs.run_hours[0]:02d}"
    sw_in, sw_box = f"{run}_sw_in", f"{run}_sw_box"
    derived = [f"{sw_in}_n{off:+d}" for off in (-1, 1)] + [f"{sw_in}_daymean"]
    return sw_in, sw_box, derived


def gfs_references(gfs_hourly: pd.DataFrame, features: FeatureConfig, cfg: EcmwfConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """GFS 내부·전체 상자 일사의 AIFS 창 평균(op, win, gw_in, gw_box)과 블록 평균(op, blk, gb_in, gb_box)."""
    sw_in, sw_box, _ = _cols(features)
    if not {"ts", sw_in, sw_box} <= set(gfs_hourly.columns):
        raise ExternalError("gfs_references: missing columns")
    op = op_day(gfs_hourly["ts"])
    hour = gfs_hourly["ts"].dt.hour
    rename = {sw_in: "gw_in", sw_box: "gw_box"}
    parts = []
    for w, hours in enumerate(aifs_window_ts_hours(cfg)):
        sel = hour.isin(list(hours))
        m = gfs_hourly.loc[sel, [sw_in, sw_box]].groupby(op[sel]).mean().rename(columns=rename)
        parts.append(m.assign(win=w).rename_axis("op").reset_index())
    win = pd.concat(parts, ignore_index=True)
    blk = gfs_hourly.assign(op=op, blk=block_index(gfs_hourly["ts"], cfg)).dropna(subset=["blk"])
    blk = blk.astype({"blk": "int64"}).groupby(["op", "blk"])[[sw_in, sw_box]].mean()
    blk = blk.rename(columns={sw_in: "gb_in", sw_box: "gb_box"}).reset_index()
    return win, blk


def substitution_targets(
    ec: pd.DataFrame, aifs: pd.DataFrame, gfs_win: pd.DataFrame, gfs_blk: pd.DataFrame, cfg: EcmwfConfig
) -> pd.DataFrame:
    """블록별 목표 일사 E(e_in, e_box). HRES 00z와 AIFS가 함께 있으면 평균, AIFS만 있으면 AIFS, 없으면 HRES.

    AIFS 6시간 창은 해 없는 시간이 섞일 수 있어 창을 공유하지 않는 블록은 같은 창 GFS와의 비율로 바꾸고,
    창을 공유하는 블록은 그날 HRES 블록 비율로 나눈다(A6). 반환 열: op, blk, e_in, e_box.
    """
    shared = {b for b, w in enumerate(cfg.aifs_block_window) if cfg.aifs_block_window.count(w) > 1}
    n_shared = pd.Series(cfg.aifs_block_window).value_counts()
    ai = aifs.assign(win=aifs["blk"].map(dict(enumerate(cfg.aifs_block_window))))
    ai = ai.merge(gfs_win, on=["op", "win"], how="left").merge(gfs_blk, on=["op", "blk"], how="left")
    for s in ("in", "box"):
        edge = ~ai["blk"].isin(shared)
        ok = edge & (ai[f"gw_{s}"] > 1)
        ai.loc[ok, f"a_{s}"] = ai.loc[ok, f"gb_{s}"] * ai.loc[ok, f"a_{s}"] / ai.loc[ok, f"gw_{s}"]
        ai.loc[edge & ~ok, f"a_{s}"] = np.nan
    t = ec.merge(ai[["op", "blk", "win", "a_in", "a_box"]], on=["op", "blk"], how="outer")
    t["win"] = t["blk"].map(dict(enumerate(cfg.aifs_block_window)))
    in_shared = t["blk"].isin(shared)
    is00 = t["src"] == "00z"
    for s in ("in", "box"):
        grp = t[in_shared].groupby(["op", "win"])[f"ec_{s}"]
        hm = grp.transform("mean").reindex(t.index)
        hn = grp.transform("count").reindex(t.index)
        split = in_shared & (hn == t["win"].map(n_shared)) & (hm > 1)
        a = t[f"a_{s}"].copy()
        a[split] = t.loc[split, f"a_{s}"] * t.loc[split, f"ec_{s}"] / hm[split]
        has_a = a.notna()
        t[f"e_{s}"] = np.where(has_a & is00, (t[f"ec_{s}"] + a) / 2, np.where(has_a, a, t[f"ec_{s}"]))
    out = t[["op", "blk", "e_in", "e_box"]].dropna(subset=["e_in", "e_box"], how="all").reset_index(drop=True)
    out["blk"] = out["blk"].astype("int64")
    return out


def substitute_gfs_hourly(
    gfs_hourly: pd.DataFrame, targets: pd.DataFrame, features: FeatureConfig, cfg: EcmwfConfig, start_op: pd.Timestamp
) -> pd.DataFrame:
    """운영일 start_op 이후 블록의 sw_in·sw_box를 G → G + λ(E - G)로 옮기고 파생 열을 다시 계산한 표(float32)."""
    sw_in, sw_box, derived = _cols(features)
    lam = cfg.substitute_lambda
    g = gfs_hourly[["ts", sw_in, sw_box]].astype({sw_in: "float64", sw_box: "float64"})
    g["op"], g["blk"] = op_day(g["ts"]), block_index(g["ts"], cfg)
    bm = g.dropna(subset=["blk"]).astype({"blk": "int64"}).groupby(["op", "blk"])[[sw_in, sw_box]].mean()
    bm = bm.rename(columns={sw_in: "G_in", sw_box: "G_box"}).reset_index()
    g = g.merge(bm, on=["op", "blk"], how="left").merge(targets, on=["op", "blk"], how="left")
    for col, s in ((sw_in, "in"), (sw_box, "box")):
        big_g = g[f"G_{s}"]
        ok = (g["op"] >= start_op) & g[f"e_{s}"].notna() & (big_g > 1)
        g.loc[ok, col] = g.loc[ok, col] * ((big_g + lam * (g[f"e_{s}"] - big_g)) / big_g)[ok]
    new = gfs_hourly.copy()
    new[sw_in] = g[sw_in].to_numpy("float32")
    new[sw_box] = g[sw_box].to_numpy("float32")
    tcc_day = [f"gfs{features.gfs.run_hours[0]:02d}_{v}_in_daymean" for v in features.gfs.daymean_vars if v != "sw"]
    return add_derived(new.drop(columns=[*derived, *tcc_day]), features.gfs, features.day_hours)


def substituted_panel(
    panel: pd.DataFrame, table: pd.DataFrame, features: FeatureConfig, train_end: pd.Timestamp, atol: float = 1e-3
) -> pd.DataFrame:
    """패널의 sw 열을 대체 표로 바꾼다. 학습 구간 값이 atol을 넘게 달라지면 ExternalError(학습 행 불변 계약)."""
    sw_in, sw_box, derived = _cols(features)
    cols = [sw_in, sw_box, *derived]
    rep = table[["ts", *cols]].copy()
    rep["ts"] = rep["ts"].astype(panel["ts"].dtype)
    out = panel.drop(columns=cols).merge(rep.astype(dict.fromkeys(cols, "float32")), on="ts", how="left")
    if len(out) != len(panel):
        raise ExternalError("substituted_panel: row count changed")
    old = panel.loc[panel["ts"] <= train_end, cols].to_numpy("float64")
    new = out.loc[out["ts"] <= train_end, cols].to_numpy("float64")
    if old.shape != new.shape or np.nanmax(np.abs(new - old), initial=0.0) > atol:
        raise ExternalError("substituted_panel: training rows changed")
    return out


def build_substituted_panel(
    panel_path: Path, out_path: Path, paths: DataPaths, cfg: EcmwfConfig, features: FeatureConfig, spec: CompetitionSpec
) -> Path:  # pragma: no cover - 파일 입출력 조립(구성 함수는 단위 테스트)
    """기록된 패널과 ECMWF 자료로 평가 기간 입력 대체 패널 파일을 만든다."""
    from src.oibc.data.external_ecmwf import aifs_blocks, ec_blocks
    from src.oibc.data.external_gfs import hourly_path

    gfs_hourly = pd.read_parquet(hourly_path(paths))
    win, blk = gfs_references(gfs_hourly, features, cfg)
    targets = substitution_targets(ec_blocks(paths, cfg, spec), aifs_blocks(paths, cfg, spec), win, blk, cfg)
    start = op_day(pd.Series([spec.eval_start])).iloc[0]
    table = substitute_gfs_hourly(gfs_hourly, targets, features, cfg, start)
    panel = substituted_panel(pd.read_parquet(panel_path), table, features, spec.train_end)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out_path, index=False)
    logger.info("[DATA] ecmwf substitute panel=%s lambda=%s targets=%d", out_path, cfg.substitute_lambda, len(targets))
    return out_path
