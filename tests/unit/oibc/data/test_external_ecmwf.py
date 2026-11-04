from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.oibc.config import CompetitionSpec, DataPaths, EcmwfConfig
from src.oibc.data.external_ecmwf import (
    audit_path,
    block_index,
    block_leads,
    block_means,
    correction_multiplier,
    ec_blocks,
    fit_beta,
    fit_frame,
    gfs_blocks,
    is_before_cutoff,
    needed_leads,
    read_audit,
    run_dir,
    run_init,
    x_table,
)
from src.oibc.data.external_gfs import ExternalError

_KST = "Asia/Seoul"
CFG = EcmwfConfig()
RUNS = {r.name: r for r in CFG.runs}
LAT = np.array([39.0, 36.0, 33.0])
LON = np.array([125.0, 128.0, 131.0])  # 내부 상자에는 (36, 128) 한 점만 들어간다


def _kst(*stamps: str) -> pd.Series:
    return pd.Series(pd.DatetimeIndex(list(stamps)).tz_localize(_KST))


def _acc(run_name: str, inner_rate: float, outer_rate: float = 0.0) -> dict[str, np.ndarray]:
    """리드 h에서 누적 ssrd = rate·h·3600 (내부 한 점만 inner_rate)."""
    out = {}
    for lead in needed_leads(RUNS[run_name], CFG):
        field = np.full((3, 3), outer_rate * lead * 3600.0)
        field[1, 1] = inner_rate * lead * 3600.0
        out[f"ssrd_{lead}"] = field
    return out


def test_block_leads_map_kst_blocks_to_each_run() -> None:
    assert block_leads(RUNS["00z"], CFG) == [(21, 24), (24, 27), (27, 30), (30, 33)]
    assert block_leads(RUNS["18z"], CFG) == [(27, 30), (30, 33), (33, 36), (36, 39)]
    assert block_leads(RUNS["12z"], CFG) == [(33, 36), (36, 39), (39, 42), (42, 45)]
    day = pd.Timestamp("2025-03-02")
    init = run_init(day, RUNS["18z"])
    assert init == pd.Timestamp("2025-02-28 18:00", tz="UTC")
    # 첫 블록 시작(D 06:00 KST)이 init + 시작 리드와 같은 순간이어야 한다
    assert init + pd.Timedelta(hours=27) == pd.Timestamp("2025-03-02 06:00", tz=_KST)


def test_run_init_rejects_tz_aware_or_non_midnight() -> None:
    with pytest.raises(ExternalError, match="tz-naive date"):
        run_init(pd.Timestamp("2025-03-02", tz=_KST), RUNS["00z"])
    with pytest.raises(ExternalError, match="tz-naive date"):
        run_init(pd.Timestamp("2025-03-02 01:00"), RUNS["00z"])


def test_block_index_right_closed_edges() -> None:
    ts = _kst("2025-03-02 06:00", "2025-03-02 07:00", "2025-03-02 09:00", "2025-03-02 10:00", "2025-03-02 18:00", "2025-03-02 19:00", "2025-03-03 00:00")
    assert block_index(ts, CFG).tolist() == [pd.NA, 0, 0, 1, 3, pd.NA, pd.NA]


def test_is_before_cutoff_boundary() -> None:
    day = pd.Timestamp("2025-03-02")
    cut = pd.Timestamp("2025-03-01 08:00", tz="UTC")  # D-1 17:00 KST
    assert is_before_cutoff(day, cut, CompetitionSpec())
    assert not is_before_cutoff(day, cut + pd.Timedelta(seconds=1), CompetitionSpec())


def test_block_means_uses_inner_box_and_requires_all_leads() -> None:
    arrays = _acc("00z", inner_rate=500.0, outer_rate=50.0)
    np.testing.assert_allclose(block_means(arrays, LAT, LON, RUNS["00z"], CFG), 500.0, rtol=0, atol=1e-9)
    del arrays["ssrd_27"]
    assert block_means(arrays, LAT, LON, RUNS["00z"], CFG) is None


def _write_run(paths: DataPaths, run_name: str, day: str, rate: float | None, published: str) -> None:
    run = RUNS[run_name]
    if rate is not None:
        folder = run_dir(paths, run)
        folder.mkdir(parents=True, exist_ok=True)
        np.savez(folder / f"{pd.Timestamp(day):%Y%m%d}.npz", lat=LAT, lon=LON, **_acc(run_name, rate))
    path = audit_path(paths, run)
    row = pd.DataFrame({"target_date": [day], "init": [f"{run_init(pd.Timestamp(day), run):%Y%m%d%H}"], "published_utc": [published]})
    if path.exists():
        row = pd.concat([pd.read_csv(path, dtype=str, keep_default_na=False), row])
    row.to_csv(path, index=False)


def _ec_env(tmp_path: Path) -> tuple[DataPaths, EcmwfConfig]:
    paths = DataPaths(external_dir=tmp_path)
    cfg = EcmwfConfig(target_start="2025-03-02", target_end="2025-03-05")
    # D=03-02: 00z 정시 → 00z. D=03-03: 00z 마감 뒤 → 18z. D=03-04: 00z 늦고 18z 파일 없음 → 12z. D=03-05: 모두 늦음 → 없음
    _write_run(paths, "00z", "2025-03-02", 100.0, "2025-03-01T07:55:00+00:00")
    _write_run(paths, "00z", "2025-03-03", 999.0, "2025-03-02T08:34:00+00:00")
    _write_run(paths, "00z", "2025-03-04", 999.0, "2025-03-03T08:34:00+00:00")
    _write_run(paths, "00z", "2025-03-05", 999.0, "2025-03-04T08:34:00+00:00")
    for day in ("2025-03-02", "2025-03-03", "2025-03-05"):
        _write_run(paths, "18z", day, 200.0, f"{pd.Timestamp(day) - pd.Timedelta(days=1):%Y-%m-%d}T{'01:27' if day != '2025-03-05' else '09:00'}:00+00:00")
    _write_run(paths, "18z", "2025-03-04", None, "2025-03-03T01:27:00+00:00")
    for day in ("2025-03-02", "2025-03-03", "2025-03-04", "2025-03-05"):
        late = day == "2025-03-05"
        _write_run(paths, "12z", day, 300.0, f"{pd.Timestamp(day) - pd.Timedelta(days=2 if not late else 1):%Y-%m-%d}T{'20:34' if not late else '09:00'}:00+00:00")
    return paths, cfg


def test_ec_blocks_picks_first_run_published_before_cutoff(tmp_path: Path) -> None:
    paths, cfg = _ec_env(tmp_path)
    ec = ec_blocks(paths, cfg, CompetitionSpec())
    by_day = ec.groupby("op").agg(src=("src", "first"), v=("ec_in", "mean"), n=("blk", "size"))
    assert by_day["src"].to_dict() == {
        pd.Timestamp("2025-03-02"): "00z",
        pd.Timestamp("2025-03-03"): "18z",
        pd.Timestamp("2025-03-04"): "12z",
    }
    np.testing.assert_allclose(by_day["v"].to_numpy(), [100.0, 200.0, 300.0], rtol=0, atol=1e-9)
    assert (by_day["n"] == 4).all()


def test_ec_blocks_ignores_corruption_of_late_runs(tmp_path: Path) -> None:
    paths, cfg = _ec_env(tmp_path)
    before = ec_blocks(paths, cfg, CompetitionSpec())
    # 마감 뒤 공개된 00z(03-03) 내용을 바꿔도 결과는 같아야 한다
    np.savez(run_dir(paths, RUNS["00z"]) / "20250303.npz", lat=LAT, lon=LON, **_acc("00z", 5.0))
    pd.testing.assert_frame_equal(before, ec_blocks(paths, cfg, CompetitionSpec()))


def test_read_audit_rejects_bad_schema_and_duplicates(tmp_path: Path) -> None:
    p = tmp_path / "a.csv"
    pd.DataFrame({"date": ["2025-03-02"], "published_utc": [""]}).to_csv(p, index=False)
    with pytest.raises(ExternalError, match="unexpected columns"):
        read_audit(p)
    pd.DataFrame({"target_date": ["2025-03-02"] * 2, "init": ["x", "y"], "published_utc": ["", ""]}).to_csv(p, index=False)
    with pytest.raises(ExternalError, match="duplicate"):
        read_audit(p)
    with pytest.raises(ExternalError, match="missing"):
        read_audit(tmp_path / "none.csv")


def test_gfs_blocks_and_x_table() -> None:
    ts = _kst("2025-03-02 07:00", "2025-03-02 08:00", "2025-03-02 09:00", "2025-03-02 10:00")
    g = gfs_blocks(pd.DataFrame({"ts": ts, "gfs00_sw_in": [100.0, 200.0, 300.0, 400.0]}), CFG)
    assert g["g_in"].tolist() == [200.0, 400.0]
    ec = pd.DataFrame({"op": [pd.Timestamp("2025-03-02")] * 2, "blk": [0, 1], "ec_in": [300.0, 250.0], "src": ["00z", "00z"]})
    xt = x_table(ec, g, CFG)
    np.testing.assert_allclose(xt["x"].to_numpy(), [100.0 / 300.0, -150.0 / 500.0], rtol=0, atol=1e-12)
    with pytest.raises(ExternalError, match="missing columns"):
        gfs_blocks(pd.DataFrame({"ts": ts}), CFG)


def test_fit_beta_recovers_multiplier_and_validates_inputs() -> None:
    rng = np.random.default_rng(0)
    f = rng.uniform(10, 100, 200)
    x = rng.uniform(-0.5, 0.5, 200)
    a = f * (1 + 0.5 * x)
    assert fit_beta(f, a, np.ones(200), x, CFG) == pytest.approx(0.5)
    with pytest.raises(ExternalError, match="same length"):
        fit_beta(f, a[:-1], np.ones(200), x, CFG)
    with pytest.raises(ExternalError, match="non-finite"):
        fit_beta(np.r_[f[:-1], np.nan], a, np.ones(200), x, CFG)


def test_correction_multiplier_only_touches_blocks_with_x() -> None:
    ts = _kst("2025-03-02 08:00", "2025-03-02 11:00", "2025-03-02 20:00", "2025-03-03 08:00")
    xt = pd.DataFrame({"op": [pd.Timestamp("2025-03-02")] * 2, "blk": [0, 1], "x": [0.2, -0.4], "src": ["00z"] * 2})
    np.testing.assert_allclose(correction_multiplier(ts, xt, 0.5, CFG), [1.1, 0.8, 1.0, 1.0], rtol=0, atol=1e-12)
    with pytest.raises(ExternalError, match="non-positive"):
        correction_multiplier(ts, xt.assign(x=[-3.0, 0.0]), 0.5, CFG)


def test_fit_frame_filters_start_and_requires_weights() -> None:
    ts = _kst("2024-03-06 08:00", "2024-03-07 08:00", "2024-03-07 20:00")
    oof = pd.DataFrame({"group": ["G1"] * 3, "ts": ts, "actual_kwh": [1.0, 2.0, 0.0], "pred_bag": [1.0, 2.0, 0.0]})
    xt = pd.DataFrame({"op": [pd.Timestamp("2024-03-06"), pd.Timestamp("2024-03-07")], "blk": [0, 0], "x": [0.1, 0.2], "src": ["00z"] * 2})
    w = pd.Series(1.0, index=ts)
    h = fit_frame(oof, w, xt, CFG)
    assert h["op"].tolist() == [pd.Timestamp("2024-03-07")]
    with pytest.raises(ExternalError, match="missing weight"):
        fit_frame(oof, w.iloc[:2], xt, CFG)


def test_block_means_rejects_empty_inner_box_and_bad_shape() -> None:
    arrays = _acc("00z", inner_rate=500.0)
    with pytest.raises(ExternalError, match="empty inner box"):
        block_means(arrays, LAT + 10.0, LON, RUNS["00z"], CFG)
    with pytest.raises(ExternalError, match="shape"):
        block_means({k: v[:2] for k, v in arrays.items()}, LAT, LON, RUNS["00z"], CFG)


def test_ec_blocks_skips_unlisted_publication(tmp_path: Path) -> None:
    paths = DataPaths(external_dir=tmp_path)
    cfg = EcmwfConfig(target_start="2025-03-02", target_end="2025-03-02")
    _write_run(paths, "00z", "2025-03-02", 100.0, "")  # 목록에 없음 = 공개 시각 확인 불가
    _write_run(paths, "18z", "2025-03-02", 200.0, "2025-03-01T01:27:00+00:00")
    _write_run(paths, "12z", "2025-03-02", 300.0, "2025-02-28T20:34:00+00:00")
    assert set(ec_blocks(paths, cfg, CompetitionSpec())["src"]) == {"18z"}


def test_fit_beta_rejects_invalid_grid() -> None:
    with pytest.raises(ExternalError, match="invalid range"):
        fit_beta(np.ones(2), np.ones(2), np.ones(2), np.zeros(2), EcmwfConfig(beta_grid=(1.0, 0.0, 0.1)))


def test_correction_multiplier_rejects_duplicate_keys_and_fit_frame_columns() -> None:
    ts = _kst("2025-03-02 08:00")
    xt = pd.DataFrame({"op": [pd.Timestamp("2025-03-02")] * 2, "blk": [0, 0], "x": [0.1, 0.2], "src": ["00z"] * 2})
    with pytest.raises(ExternalError, match="duplicate x keys"):
        correction_multiplier(ts, xt, 0.5, CFG)
    with pytest.raises(ExternalError, match="missing columns"):
        fit_frame(pd.DataFrame({"ts": ts}), pd.Series(1.0, index=ts), xt, CFG)


def test_ec_blocks_falls_back_when_run_file_is_incomplete(tmp_path: Path) -> None:
    paths = DataPaths(external_dir=tmp_path)
    cfg = EcmwfConfig(target_start="2025-03-02", target_end="2025-03-02")
    _write_run(paths, "00z", "2025-03-02", 100.0, "2025-03-01T07:55:00+00:00")
    partial = {k: v for k, v in _acc("00z", 100.0).items() if k != "ssrd_33"}
    np.savez(run_dir(paths, RUNS["00z"]) / "20250302.npz", lat=LAT, lon=LON, **partial)
    _write_run(paths, "18z", "2025-03-02", 200.0, "2025-03-01T01:27:00+00:00")
    _write_run(paths, "12z", "2025-03-02", 300.0, "2025-02-28T20:34:00+00:00")
    assert set(ec_blocks(paths, cfg, CompetitionSpec())["src"]) == {"18z"}
