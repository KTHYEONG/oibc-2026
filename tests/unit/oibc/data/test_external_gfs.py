from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.oibc.config import CompetitionSpec, DataPaths, ExternalGfsConfig, FeatureConfig, NoaaGfsConfig
from src.oibc.data.external_gfs import (
    ExternalError,
    add_derived,
    assemble_noaa_arrays,
    audit_path,
    build_gfs_hourly,
    compare_summaries,
    crop_box,
    deaverage,
    gfs_dir,
    hourly_path,
    is_published_before_cutoff,
    last_used_lead,
    noaa_fetch_leads,
    noaa_pending_inits,
    parse_idx,
    parse_init_id,
    select_message,
    summarize_init,
    target_window,
)

_KST = "Asia/Seoul"
_LAT = np.array([39.0, 37.0, 35.0, 33.0])
_LON = np.array([125.0, 127.0, 129.0, 131.0])
_SPEC = CompetitionSpec(
    train_end=pd.Timestamp("2025-06-01 00:00", tz=_KST),
    eval_start=pd.Timestamp("2025-06-01 01:00", tz=_KST),
    eval_end=pd.Timestamp("2025-06-02 00:00", tz=_KST),
    n_eval_hours=24,
)


def _cfg(**kw: object) -> ExternalGfsConfig:
    base: dict[str, object] = {
        "init_start_date": "2025-05-29",
        "init_end_date": "2025-05-31",
        "variables": {"sw": ("sw_src", "avg6"), "t2m": ("t_src", "inst")},
        "neighbor_vars": ("sw",),
        "daymean_vars": ("sw",),
    }
    base.update(kw)
    return ExternalGfsConfig.model_validate(base)


def _avg6_from_hourly(hourly: np.ndarray) -> np.ndarray:
    """리드별 1시간 평균에서 6시간마다 초기화되는 누적 평균 배열을 만든다(리드 0은 0)."""
    out = np.zeros_like(hourly)
    for lead in range(1, hourly.shape[0]):
        start = ((lead - 1) // 6) * 6 + 1
        out[lead] = hourly[start : lead + 1].mean(axis=0)
    return out


def _arrays(seed: int, n_lead: int = 49) -> tuple[dict[str, np.ndarray], np.ndarray]:
    rng = np.random.default_rng(seed)
    hourly = rng.uniform(0, 800, (n_lead, _LAT.size, _LON.size))
    temp = rng.uniform(270, 300, (n_lead, _LAT.size, _LON.size))
    return {"sw_src": _avg6_from_hourly(hourly), "t_src": temp}, hourly


def test_target_window_maps_00z_and_18z_to_next_kst_day() -> None:
    ts, leads = target_window(pd.Timestamp("2025-06-01 00:00", tz="UTC"))
    assert ts[0] == pd.Timestamp("2025-06-02 01:00", tz=_KST)
    assert ts[-1] == pd.Timestamp("2025-06-03 00:00", tz=_KST)
    assert leads[0] == 16 and leads[-1] == 39
    ts18, leads18 = target_window(pd.Timestamp("2025-05-31 18:00", tz="UTC"))
    assert ts18[0] == pd.Timestamp("2025-06-02 01:00", tz=_KST)
    assert leads18[0] == 22 and leads18[-1] == 45
    assert last_used_lead(0) == 39 and last_used_lead(18) == 45


def test_target_window_rejects_naive_or_offhour_init() -> None:
    with pytest.raises(ExternalError):
        target_window(pd.Timestamp("2025-06-01 00:00"))
    with pytest.raises(ExternalError):
        target_window(pd.Timestamp("2025-06-01 00:30", tz="UTC"))


def test_deaverage_recovers_hourly_means() -> None:
    arrays, hourly = _arrays(0)
    leads = np.arange(1, 49)
    got = deaverage(arrays["sw_src"], leads, init_hour=0)
    np.testing.assert_allclose(got, hourly[leads], rtol=1e-9, atol=1e-6)


def test_deaverage_rejects_lead_zero_and_offcycle_init() -> None:
    arrays, _ = _arrays(0)
    with pytest.raises(ExternalError):
        deaverage(arrays["sw_src"], np.array([0, 1]), init_hour=0)
    with pytest.raises(ExternalError):
        deaverage(arrays["sw_src"], np.array([1, 2]), init_hour=3)


def test_summarize_init_box_and_inner_stats() -> None:
    cfg = _cfg(inner_lat=(34.0, 38.0), inner_lon=(126.0, 130.0))
    arrays, hourly = _arrays(1)
    init = pd.Timestamp("2025-05-31 00:00", tz="UTC")
    out = summarize_init(arrays, _LAT, _LON, init, cfg)
    _, leads = target_window(init)
    inner = hourly[leads][:, 1:3, 1:3].reshape(24, -1)
    np.testing.assert_allclose(out["gfs00_sw_box"], hourly[leads].mean(axis=(1, 2)), rtol=1e-9)
    np.testing.assert_allclose(out["gfs00_sw_in"], inner.mean(axis=1), rtol=1e-9)
    np.testing.assert_allclose(out["gfs00_sw_insd"], inner.std(axis=1), rtol=1e-9)
    np.testing.assert_allclose(out["gfs00_t2m_in"], arrays["t_src"][leads][:, 1:3, 1:3].reshape(24, -1).mean(axis=1))
    assert len(out) == 24


def test_summarize_init_fails_closed_on_missing_variable_or_empty_inner_box() -> None:
    arrays, _ = _arrays(2)
    init = pd.Timestamp("2025-05-31 00:00", tz="UTC")
    with pytest.raises(ExternalError, match="missing variable"):
        summarize_init({"sw_src": arrays["sw_src"]}, _LAT, _LON, init, _cfg())
    with pytest.raises(ExternalError, match="inner box"):
        summarize_init(arrays, _LAT, _LON, init, _cfg(inner_lat=(34.0, 34.5), inner_lon=(126.0, 126.5)))


def test_publication_cutoff_boundary() -> None:
    init = pd.Timestamp("2025-06-01 00:00", tz="UTC")  # 대상일 2025-06-02, 마감 2025-06-01 17:00 KST
    at_cutoff = pd.Timestamp("2025-06-01 08:00", tz="UTC")
    assert is_published_before_cutoff(init, at_cutoff, _SPEC)
    assert not is_published_before_cutoff(init, at_cutoff + pd.Timedelta(seconds=1), _SPEC)


def test_add_derived_neighbors_stay_within_operating_day() -> None:
    cfg = _cfg()
    ts = pd.date_range(pd.Timestamp("2025-06-01 01:00", tz=_KST), periods=48, freq="h")
    table = pd.DataFrame({"ts": ts, "gfs00_sw_in": np.arange(48, dtype="float64")})
    out = add_derived(table, cfg, day_hours=(9, 16))
    first_day_end = out.index[out["ts"] == pd.Timestamp("2025-06-02 00:00", tz=_KST)][0]
    assert np.isnan(out.loc[first_day_end, "gfs00_sw_in_n+1"])
    assert np.isnan(out.loc[first_day_end + 1, "gfs00_sw_in_n-1"])
    assert out.loc[1, "gfs00_sw_in_n-1"] == 0.0
    day1 = out["ts"] <= pd.Timestamp("2025-06-02 00:00", tz=_KST)
    expected = table.loc[(table["ts"].dt.hour >= 9) & (table["ts"].dt.hour <= 16) & day1, "gfs00_sw_in"].mean()
    assert np.allclose(out.loc[day1, "gfs00_sw_in_daymean"], expected)


def _write_fixture(root: Path, cfg: ExternalGfsConfig, published: dict[str, str]) -> DataPaths:
    paths = DataPaths(
        raw_dir=root / "raw", canonical_dir=root / "canonical", derived_dir=root / "derived",
        features_dir=root / "features", external_dir=root / "external",
    )
    gfs_dir(paths, 0).mkdir(parents=True)
    for k, day in enumerate(pd.date_range(cfg.init_start_date, cfg.init_end_date, freq="D")):
        arrays, _ = _arrays(10 + k)
        np.savez_compressed(
            gfs_dir(paths, 0) / f"{day:%Y%m%d}00.npz", lat=_LAT, lon=_LON, lead_h=np.arange(49, dtype="int16"), **arrays
        )
    rows = [{"init": k, "last_lead_h": 39, "published_utc": v} for k, v in published.items()]
    pd.DataFrame(rows).to_csv(audit_path(paths), index=False)
    return paths


def _on_time() -> dict[str, str]:
    return {f"202505{d}00": f"2025-05-{d}T03:45:00Z" for d in ("29", "30", "31")}


def test_build_gfs_hourly_masks_late_runs_and_writes_float32(tmp_path: Path) -> None:
    cfg = _cfg()
    published = _on_time() | {"2025053000": "2025-05-30T09:30:00Z"}  # 마감(08:00 UTC) 이후
    paths = _write_fixture(tmp_path, cfg, published)
    out = pd.read_parquet(build_gfs_hourly(paths, _SPEC, FeatureConfig(gfs=cfg)))
    assert out.columns.tolist() == ["ts", *cfg.feature_columns()]
    op_day = (out["ts"] - pd.Timedelta(hours=1)).dt.normalize()
    late = op_day == pd.Timestamp("2025-05-31", tz=_KST)
    assert out.loc[late, "gfs00_sw_in"].isna().all()
    assert out.loc[op_day == pd.Timestamp("2025-06-01", tz=_KST), "gfs00_sw_in"].notna().all()
    assert all(out[c].dtype == "float32" for c in cfg.feature_columns())


def test_build_gfs_hourly_fails_closed_on_missing_audit_or_file(tmp_path: Path) -> None:
    cfg = _cfg()
    paths = _write_fixture(tmp_path, cfg, {k: v for k, v in _on_time().items() if k != "2025053100"})
    with pytest.raises(ExternalError, match="audit"):
        build_gfs_hourly(paths, _SPEC, FeatureConfig(gfs=cfg))
    paths = _write_fixture(tmp_path / "b", cfg, _on_time())
    (gfs_dir(paths, 0) / "2025053000.npz").unlink()
    with pytest.raises(ExternalError, match="missing init file"):
        build_gfs_hourly(paths, _SPEC, FeatureConfig(gfs=cfg))


def test_build_gfs_hourly_requires_eval_coverage(tmp_path: Path) -> None:
    cfg = _cfg(init_end_date="2025-05-30")
    paths = _write_fixture(tmp_path, cfg, _on_time())
    with pytest.raises(ExternalError, match="evaluation hour"):
        build_gfs_hourly(paths, _SPEC, FeatureConfig(gfs=cfg))


def test_build_gfs_hourly_is_invariant_to_later_inits(tmp_path: Path) -> None:
    """대상일 D 값은 D-1 00z 파일에만 의존: 이후 init을 교란해도 D 이전 날짜 행은 그대로다."""
    cfg = _cfg()
    paths = _write_fixture(tmp_path, cfg, _on_time())
    before = pd.read_parquet(build_gfs_hourly(paths, _SPEC, FeatureConfig(gfs=cfg)))
    path = gfs_dir(paths, 0) / "2025053100.npz"
    with np.load(path) as z:
        corrupted = {k: z[k] for k in z.files}
    corrupted["sw_src"] = corrupted["sw_src"] * 0.0 + 999.0
    corrupted["t_src"] = corrupted["t_src"] * 0.0 - 50.0
    np.savez_compressed(path, **corrupted)
    build_gfs_hourly(paths, _SPEC, FeatureConfig(gfs=cfg))
    after = pd.read_parquet(hourly_path(paths))
    op_day = (before["ts"] - pd.Timedelta(hours=1)).dt.normalize()
    keep = op_day < pd.Timestamp("2025-06-01", tz=_KST)
    pd.testing.assert_frame_equal(before[keep], after[keep])
    assert not before[~keep].equals(after[~keep])


def test_summarize_init_rejects_bad_shape() -> None:
    arrays, _ = _arrays(3)
    arrays["t_src"] = arrays["t_src"][:, :2, :]
    with pytest.raises(ExternalError, match="bad shape"):
        summarize_init(arrays, _LAT, _LON, pd.Timestamp("2025-05-31 00:00", tz="UTC"), _cfg())


def test_audit_contract_violations(tmp_path: Path) -> None:
    cfg = _cfg()
    paths = _write_fixture(tmp_path, cfg, _on_time())
    audit_path(paths).unlink()
    with pytest.raises(ExternalError, match="audit missing"):
        build_gfs_hourly(paths, _SPEC, FeatureConfig(gfs=cfg))
    pd.DataFrame({"init": ["2025052900"], "when": ["x"]}).to_csv(audit_path(paths), index=False)
    with pytest.raises(ExternalError, match="unexpected columns"):
        build_gfs_hourly(paths, _SPEC, FeatureConfig(gfs=cfg))
    rows = [{"init": "2025052900", "last_lead_h": 39, "published_utc": "2025-05-29T03:45:00Z"}] * 2
    pd.DataFrame(rows).to_csv(audit_path(paths), index=False)
    with pytest.raises(ExternalError, match="duplicate init"):
        build_gfs_hourly(paths, _SPEC, FeatureConfig(gfs=cfg))


def test_build_gfs_hourly_rejects_irregular_lead_axis(tmp_path: Path) -> None:
    cfg = _cfg()
    paths = _write_fixture(tmp_path, cfg, _on_time())
    path = gfs_dir(paths, 0) / "2025052900.npz"
    with np.load(path) as z:
        data = {k: z[k] for k in z.files}
    data["lead_h"] = data["lead_h"] * 3
    np.savez_compressed(path, **data)
    with pytest.raises(ExternalError, match="lead axis"):
        build_gfs_hourly(paths, _SPEC, FeatureConfig(gfs=cfg))


# --- NOAA 원본 예비 경로 ---

_INIT = pd.Timestamp("2025-05-30 00:00", tz="UTC")


def _noaa_ext() -> ExternalGfsConfig:
    return _cfg(variables={"sw": ("sw_src", "avg6"), "t2m": ("t_src", "inst"), "pr": ("pr_src", "inst")})


def _noaa_cfg(**kw: object) -> NoaaGfsConfig:
    base: dict[str, object] = {
        "fields": {
            "sw_src": ("DSWRF", "surface", "ave", 0.0),
            "t_src": ("TMP", "2 m above ground", "inst", -273.15),
            "pr_src": ("PRATE", "surface", "ave", 0.0),
        },
        "hourly_from_ave": ("pr_src",),
        "equiv_tol": {"sw": 2.0, "t2m": 0.05, "pr": 1e-5},
    }
    base.update(kw)
    return NoaaGfsConfig.model_validate(base)


def _raw(values: dict[str, np.ndarray], leads: np.ndarray) -> dict[str, dict[int, np.ndarray]]:
    return {src: {int(lead): arr[int(lead)] for lead in leads} for src, arr in values.items()}


def test_noaa_fetch_leads_00z_covers_deaverage_prev() -> None:
    assert np.array_equal(noaa_fetch_leads(0), np.arange(15, 40))
    for h in (6, 12, 18):
        assert noaa_fetch_leads(h).min() >= 1 and np.all(np.diff(noaa_fetch_leads(h)) == 1)


_IDX = "1:0:d=2025053000:DSWRF:surface:24-27 hour ave fcst:\n2:100:d=2025053000:DSWRF:surface:26-27 hour ave fcst:\n3:250:d=2025053000:TMP:2 m above ground:27 hour fcst:\n"


def test_parse_idx_ranges_and_last_none() -> None:
    recs = parse_idx(_IDX)
    assert parse_idx("\n" + _IDX + "\n") == recs
    assert recs == [
        (0, 100, "DSWRF:surface:24-27 hour ave fcst"),
        (100, 250, "DSWRF:surface:26-27 hour ave fcst"),
        (250, None, "TMP:2 m above ground:27 hour fcst"),
    ]


@pytest.mark.parametrize(
    "text",
    ["", "garbage line\n", "1:x:d=2025053000:A:b:c:\n", "1:50:d=2025053000:A:b:c:\n2:50:d=2025053000:B:b:c:\n"],
)
def test_parse_idx_rejects_malformed_or_nonincreasing(text: str) -> None:
    with pytest.raises(ExternalError, match="idx"):
        parse_idx(text)


def test_select_message_ave_requires_reset_window() -> None:
    assert select_message(parse_idx(_IDX), "DSWRF", "surface", "ave", 27) == (0, 99)


def test_select_message_inst_and_failures() -> None:
    recs = parse_idx(_IDX + "4:400:d=2025053000:RH:2 m above ground:27 hour fcst:\n")
    assert select_message(recs, "TMP", "2 m above ground", "inst", 27) == (250, 399)
    with pytest.raises(ExternalError, match="0 match"):
        select_message(recs, "TCDC", "entire atmosphere", "ave", 27)
    dup = parse_idx(_IDX.replace("26-27", "24-27"))
    with pytest.raises(ExternalError, match="2 match"):
        select_message(dup, "DSWRF", "surface", "ave", 27)
    with pytest.raises(ExternalError, match="last message"):
        select_message(recs, "RH", "2 m above ground", "inst", 27)


def test_crop_box_orders_north_to_south_and_validates_grid() -> None:
    cfg = ExternalGfsConfig()
    lat = np.arange(30.0, 42.0 + 1e-9, 0.25)  # 남→북 입력
    lon = np.arange(120.0, 135.0 + 1e-9, 0.25)
    values = lat[:, None] * 1000.0 + lon[None, :]
    arr, la, lo = crop_box(values, lat, lon, cfg)
    assert arr.shape == (31, 31) and la[0] == 39.5 and la[-1] == 32.0 and lo[0] == 124.0 and lo[-1] == 131.5
    assert np.allclose(arr, la[:, None] * 1000.0 + lo[None, :])
    lon360 = np.arange(0.0, 360.0, 0.25)
    arr2, _, lo2 = crop_box(np.tile(lon360, (lat.size, 1)), lat, lon360, cfg)
    assert np.allclose(lo2, np.arange(124.0, 131.5 + 1e-9, 0.25)) and np.allclose(arr2[0], lo2)
    with pytest.raises(ExternalError, match=r"0\.25 deg"):
        crop_box(values[:, ::2], lat, lon[::2], cfg)
    bent = lat.copy()
    bent[np.argmin(np.abs(lat - 35.0))] = 35.1
    with pytest.raises(ExternalError, match=r"0\.25 deg"):
        crop_box(values, bent, lon, cfg)
    with pytest.raises(ExternalError, match="values shape"):
        crop_box(values.T, lat, lon, cfg)


def test_assemble_applies_offset_and_nan_outside_fetch_leads() -> None:
    ext, noaa = _noaa_ext(), _noaa_cfg()
    rng = np.random.default_rng(0)
    vals = {k: rng.uniform(0, 300, (49, 4, 4)) for k in ("sw_src", "t_src", "pr_src")}
    out = assemble_noaa_arrays(_raw(vals, noaa_fetch_leads(0)), _INIT, ext, noaa)
    t = out["t_src"]
    assert t.shape == (49, 4, 4) and t.dtype == np.float32
    assert np.allclose(t[15:40], vals["t_src"][15:40] - 273.15, atol=1e-4)
    assert np.isnan(t[:15]).all() and np.isnan(t[40:]).all()


def test_assemble_precip_deaverage_recovers_hourly() -> None:
    ext, noaa = _noaa_ext(), _noaa_cfg()
    rng = np.random.default_rng(1)
    hourly = rng.uniform(0, 1e-3, (49, 4, 4))
    vals = {"sw_src": rng.uniform(0, 800, (49, 4, 4)), "t_src": rng.uniform(270, 300, (49, 4, 4)), "pr_src": _avg6_from_hourly(hourly)}
    pr = assemble_noaa_arrays(_raw(vals, noaa_fetch_leads(0)), _INIT, ext, noaa)["pr_src"]
    assert np.allclose(pr[16:40], hourly[16:40], rtol=1e-6, atol=1e-10)
    assert np.isnan(pr[15]).all()


def test_assemble_keeps_avg6_raw() -> None:
    ext, noaa = _noaa_ext(), _noaa_cfg()
    arrays, _ = _arrays(3)
    rng = np.random.default_rng(3)
    pr_hourly = rng.uniform(0, 1e-3, (49, _LAT.size, _LON.size))
    raw_vals = {"sw_src": arrays["sw_src"], "t_src": arrays["t_src"] + 273.15, "pr_src": _avg6_from_hourly(pr_hourly)}
    out = assemble_noaa_arrays(_raw(raw_vals, noaa_fetch_leads(0)), _INIT, ext, noaa)
    assert np.allclose(out["sw_src"][15:40], arrays["sw_src"][15:40].astype("float32"))
    dyn = {"sw_src": arrays["sw_src"], "t_src": arrays["t_src"], "pr_src": pr_hourly}
    a = summarize_init(out, _LAT, _LON, _INIT, ext)
    b = summarize_init(dyn, _LAT, _LON, _INIT, ext)
    cols = [c for c in b.columns if c not in ("ts", "init")]
    assert np.allclose(a[cols].to_numpy(dtype="float64"), b[cols].to_numpy(dtype="float64"), rtol=1e-4, atol=1e-6)


def test_assemble_fails_closed_on_missing_var_or_lead() -> None:
    ext, noaa = _noaa_ext(), _noaa_cfg()
    vals = {k: np.ones((49, 2, 2)) for k in ("sw_src", "t_src", "pr_src")}
    raw = _raw(vals, noaa_fetch_leads(0))
    with pytest.raises(ExternalError, match="missing NOAA variable"):
        assemble_noaa_arrays({k: v for k, v in raw.items() if k != "t_src"}, _INIT, ext, noaa)
    with pytest.raises(ExternalError, match="no NOAA field mapping"):
        assemble_noaa_arrays(raw, _INIT, ext, _noaa_cfg(fields={"sw_src": ("DSWRF", "surface", "ave", 0.0)}, hourly_from_ave=()))
    with pytest.raises(ExternalError, match="max_lead_h"):
        assemble_noaa_arrays(raw, _INIT, _noaa_ext().model_copy(update={"max_lead_h": 30}), noaa)
    odd = {k: dict(v) for k, v in raw.items()}
    odd["t_src"][20] = np.ones((3, 3))
    with pytest.raises(ExternalError, match="inconsistent grid shape"):
        assemble_noaa_arrays(odd, _INIT, ext, noaa)
    del raw["sw_src"][27]
    with pytest.raises(ExternalError, match="missing leads"):
        assemble_noaa_arrays(raw, _INIT, ext, noaa)


def _summary_pair() -> tuple[pd.DataFrame, pd.DataFrame]:
    ts = pd.date_range("2025-05-31 01:00", periods=3, freq="h", tz=_KST)
    base = pd.DataFrame({"ts": ts, "gfs00_sw_box": [100.0, 200.0, 300.0], "gfs00_sw_insd": [10.0, 20.0, 30.0]})
    return base, base.copy()


def test_compare_summaries_tolerance_and_insd_factor() -> None:
    ext, noaa = _cfg(), _noaa_cfg()
    a, b = _summary_pair()
    b.loc[1, "gfs00_sw_box"] += 1.9
    b.loc[1, "gfs00_sw_insd"] += 5.9  # 2.0 x 3.0 = 6.0 이내
    res = compare_summaries(a, b, ext, noaa).set_index("column")
    assert res["ok"].all() and res.loc["gfs00_sw_insd", "tol"] == pytest.approx(6.0)
    b.loc[2, "gfs00_sw_box"] += 2.1
    assert not compare_summaries(a, b, ext, noaa).set_index("column").loc["gfs00_sw_box", "ok"]
    c, d = _summary_pair()
    d.loc[0, "gfs00_sw_box"] = np.nan
    assert not compare_summaries(c, d, ext, noaa).set_index("column").loc["gfs00_sw_box", "ok"]
    e, f = _summary_pair()
    f["ts"] = f["ts"] + pd.Timedelta(hours=1)
    with pytest.raises(ExternalError, match="ts mismatch"):
        compare_summaries(e, f, ext, noaa)
    g, h = _summary_pair()
    with pytest.raises(ExternalError, match="column mismatch"):
        compare_summaries(g, h.drop(columns="gfs00_sw_insd"), ext, noaa)
    with pytest.raises(ExternalError, match="no tolerance"):
        compare_summaries(g.rename(columns={"gfs00_sw_box": "gfs00_lw_box"}), h.rename(columns={"gfs00_sw_box": "gfs00_lw_box"}), ext, noaa)


def test_parse_init_id_contract() -> None:
    assert parse_init_id("2025071000") == pd.Timestamp("2025-07-10 00:00", tz="UTC")
    for bad in ("2025-07-10", "202507100", "2025133100"):
        with pytest.raises(ExternalError, match="init id"):
            parse_init_id(bad)


def test_noaa_pending_inits_skips_complete_and_rejects_partial(tmp_path: Path) -> None:
    cfg = _cfg()
    paths = DataPaths(
        raw_dir=tmp_path / "raw", canonical_dir=tmp_path / "canonical", derived_dir=tmp_path / "derived",
        features_dir=tmp_path / "features", external_dir=tmp_path / "external",
    )
    d = gfs_dir(paths, 0)
    d.mkdir(parents=True)
    arrays, _ = _arrays(5)
    np.savez_compressed(d / "2025052900.npz", lat=_LAT, lon=_LON, lead_h=np.arange(49, dtype="int16"), **arrays)
    pending = noaa_pending_inits(paths, cfg, 0)
    assert [f"{i:%Y%m%d%H}" for i in pending] == ["2025053000", "2025053100"]
    np.savez_compressed(d / "2025053000.npz", lat=_LAT, lon=_LON, sw_src=arrays["sw_src"])
    with pytest.raises(ExternalError, match="partial file"):
        noaa_pending_inits(paths, cfg, 0)
