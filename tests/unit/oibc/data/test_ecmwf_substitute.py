import numpy as np
import pandas as pd
import pytest

from src.oibc.config import EcmwfConfig, FeatureConfig
from src.oibc.data.ecmwf_substitute import gfs_references, substitute_gfs_hourly, substituted_panel, substitution_targets
from src.oibc.data.external_ecmwf import block_means
from src.oibc.data.external_gfs import ExternalError

_KST = "Asia/Seoul"
CFG = EcmwfConfig()
FEAT = FeatureConfig()


def _hourly(days: list[str], sw: float = 100.0) -> pd.DataFrame:
    ts = pd.concat([pd.Series(pd.date_range(pd.Timestamp(d) + pd.Timedelta(hours=1), periods=24, freq="h", tz=_KST)) for d in days], ignore_index=True)
    h = ts.dt.hour
    val = np.where((h >= 6) & (h <= 19), sw, 0.0).astype("float32")
    t = pd.DataFrame({"ts": ts, "gfs00_sw_in": val, "gfs00_sw_box": val * 2, "gfs00_tcc_in": np.float32(50.0)})
    return t


def test_gfs_references_window_and_block_means() -> None:
    t = _hourly(["2025-07-01"])
    t["gfs00_sw_in"] = t["ts"].dt.hour.astype("float32")
    win, blk = gfs_references(t, FEAT, CFG)
    np.testing.assert_allclose(win.sort_values("win")["gw_in"].to_numpy(), [6.5, 12.5, 18.5])
    np.testing.assert_allclose(blk.sort_values("blk")["gb_in"].to_numpy(), [8.0, 11.0, 14.0, 17.0])
    with pytest.raises(ExternalError, match="missing"):
        gfs_references(t.drop(columns="gfs00_sw_box"), FEAT, CFG)


def test_substitution_targets_combines_hres_aifs_with_edge_ratio_and_split() -> None:
    d0, d1 = pd.Timestamp("2025-07-01"), pd.Timestamp("2025-11-01")
    ec = pd.DataFrame({"op": [d0] * 4 + [d1] * 4, "blk": [0, 1, 2, 3] * 2, "ec_in": [100.0, 200.0, 400.0, 100.0] * 2,
                       "ec_box": [100.0, 200.0, 400.0, 100.0] * 2, "src": ["18z"] * 4 + ["00z"] * 4})
    aifs = pd.DataFrame({"op": [d0] * 4 + [d1] * 4, "blk": [0, 1, 2, 3] * 2, "a_in": [50.0, 600.0, 600.0, 30.0] * 2, "a_box": [50.0, 600.0, 600.0, 30.0] * 2})
    win = pd.DataFrame({"op": [d0] * 3 + [d1] * 3, "win": [0, 1, 2] * 2, "gw_in": [50.0, 300.0, 0.5] * 2, "gw_box": [50.0, 300.0, 0.5] * 2})
    blk = pd.DataFrame({"op": [d0] * 4 + [d1] * 4, "blk": [0, 1, 2, 3] * 2, "gb_in": [120.0, 1.0, 1.0, 80.0] * 2, "gb_box": [120.0, 1.0, 1.0, 80.0] * 2})
    out = substitution_targets(ec, aifs, win, blk, CFG).set_index(["op", "blk"])["e_in"]
    # 18z 날: 블록0 = 120·50/50 = 120, 블록1·2 = 600·(200|400)/300, 블록3은 창 GFS ≤ 1이라 AIFS 없음 → HRES 100
    np.testing.assert_allclose(out.loc[d0].to_numpy(), [120.0, 400.0, 800.0, 100.0])
    # 00z 날 기본(AIFS 비중 1): AIFS 단독, 블록3은 HRES
    np.testing.assert_allclose(out.loc[d1].to_numpy(), [120.0, 400.0, 800.0, 100.0])
    # AIFS 비중 0.5: HRES와 AIFS 평균
    half = CFG.model_copy(update={"substitute_aifs_weight_00z": 0.5})
    out_half = substitution_targets(ec, aifs, win, blk, half).set_index(["op", "blk"])["e_in"]
    np.testing.assert_allclose(out_half.loc[d1].to_numpy(), [110.0, 300.0, 600.0, 100.0])
    np.testing.assert_allclose(out_half.loc[d0].to_numpy(), out.loc[d0].to_numpy())


def test_substitute_moves_block_mean_after_start_only_and_rebuilds_derived() -> None:
    t = _hourly(["2025-05-31", "2025-06-01"])
    tg = pd.DataFrame({"op": [pd.Timestamp("2025-05-31"), pd.Timestamp("2025-06-01")], "blk": [1, 1], "e_in": [300.0, 300.0], "e_box": [np.nan, 100.0]})
    for c in ("gfs00_sw_in_n-1", "gfs00_sw_in_n+1", "gfs00_sw_in_daymean", "gfs00_tcc_in_daymean"):
        t[c] = np.float32(0)
    new = substitute_gfs_hourly(t, tg, FEAT, CFG, pd.Timestamp("2025-06-01"))
    h = new["ts"].dt.hour
    pre = new["ts"] < pd.Timestamp("2025-06-01 01:00", tz=_KST)
    np.testing.assert_array_equal(new.loc[pre, "gfs00_sw_in"].to_numpy(), t.loc[pre.to_numpy(), "gfs00_sw_in"].to_numpy())
    post_blk1 = ~pre & h.between(10, 12)
    np.testing.assert_allclose(new.loc[post_blk1, "gfs00_sw_in"].to_numpy(), 200.0, rtol=1e-6)  # 100 + 0.5·(300 - 100)
    np.testing.assert_allclose(new.loc[post_blk1, "gfs00_sw_box"].to_numpy(), 150.0, rtol=1e-6)  # 200 + 0.5·(100 - 200)
    np.testing.assert_allclose(new.loc[~pre & h.between(13, 15), "gfs00_sw_in"].to_numpy(), 100.0)
    assert new["gfs00_sw_in"].dtype == np.float32 and {"gfs00_sw_in_n-1", "gfs00_sw_in_daymean", "gfs00_tcc_in_daymean"} <= set(new.columns)
    nxt = new.set_index("ts")["gfs00_sw_in_n+1"]
    assert nxt.loc[pd.Timestamp("2025-06-01 09:00", tz=_KST)] == pytest.approx(200.0, rel=1e-6)


def test_substituted_panel_keeps_training_rows_and_fails_closed() -> None:
    ts = pd.Series(pd.date_range("2025-05-31 23:00", periods=3, freq="h", tz=_KST))
    cols = ["gfs00_sw_in", "gfs00_sw_box", "gfs00_sw_in_n-1", "gfs00_sw_in_n+1", "gfs00_sw_in_daymean"]
    panel = pd.DataFrame({"ts": list(ts) * 2, "site_id": ["a"] * 3 + ["b"] * 3, **{c: np.float32(1.0) for c in cols}})
    table = pd.DataFrame({"ts": ts, **{c: np.float32(1.0) for c in cols}})
    table.loc[2, "gfs00_sw_in"] = 5.0
    out = substituted_panel(panel, table, FEAT, pd.Timestamp("2025-06-01 00:00", tz=_KST))
    assert len(out) == len(panel) and out.loc[out["ts"] == ts[2], "gfs00_sw_in"].tolist() == [5.0, 5.0]
    table.loc[0, "gfs00_sw_in"] = 2.0
    with pytest.raises(ExternalError, match="training rows changed"):
        substituted_panel(panel, table, FEAT, pd.Timestamp("2025-06-01 00:00", tz=_KST))
    with pytest.raises(ExternalError, match="row count"):
        substituted_panel(panel, pd.concat([table, table]), FEAT, pd.Timestamp("2025-06-01 00:00", tz=_KST))


def test_block_means_whole_box_uses_every_cell() -> None:
    lat, lon = np.array([39.0, 36.0, 33.0]), np.array([125.0, 128.0, 131.0])
    run = CFG.runs[0]
    arr = {f"ssrd_{h}": np.full((3, 3), 100.0 * h * 3600.0) for h in (21, 24, 27, 30, 33)}
    arr = {k: v.copy() for k, v in arr.items()}
    arr["ssrd_24"][0, 0] += 3 * 3600.0 * 900.0  # 블록 0의 모서리 한 칸만 +900 W/m²
    inner = block_means(arr, lat, lon, run, CFG)
    box = block_means(arr, lat, lon, run, CFG, whole_box=True)
    assert inner[0] == pytest.approx(100.0) and box[0] == pytest.approx(200.0)
