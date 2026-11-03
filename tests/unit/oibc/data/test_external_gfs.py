from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.oibc.config import CompetitionSpec, DataPaths, ExternalGfsConfig, FeatureConfig
from src.oibc.data.external_gfs import (
    ExternalError,
    add_derived,
    audit_path,
    build_gfs_hourly,
    deaverage,
    gfs_dir,
    hourly_path,
    is_published_before_cutoff,
    last_used_lead,
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
