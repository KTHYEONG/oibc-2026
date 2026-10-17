import hashlib
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from pydantic import ValidationError

from src.oibc.config import CompetitionSpec, DataPaths
from src.oibc.submit import validate as vmod
from src.oibc.submit.build import SubmissionManifest, record_leaderboard_score, write_submission
from src.oibc.submit.validate import SubmissionError, load_reference, validate_submission

_TS = (
    "2025-06-01 01:00:00+09:00",
    "2025-06-01 02:00:00+09:00",
    "2025-06-01 03:00:00+09:00",
)
_CAPS = {"PV001": "100", "PV002": "100", "PV003": "99.99", "PV004": "100.01"}


def _spec() -> CompetitionSpec:
    return CompetitionSpec(
        n_sites=4,
        min_group_capacity_kw=200.0,
        train_end=pd.Timestamp("2025-06-01 00:00", tz="Asia/Seoul"),
        eval_start=pd.Timestamp("2025-06-01 01:00", tz="Asia/Seoul"),
        eval_end=pd.Timestamp("2025-06-01 03:00", tz="Asia/Seoul"),
        n_eval_hours=3,
    )


@pytest.fixture
def ctx(tmp_path: Path) -> tuple[vmod.SubmissionReference, CompetitionSpec, Path]:
    spec = _spec()
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "sites.csv").write_text(
        "site_id,capacity_kw\n" + "".join(f"{sid},{cap}\n" for sid, cap in _CAPS.items()),
        encoding="utf-8",
    )
    (raw / "smp.csv").write_text(
        "ts,smp,k\n" + "".join(f"{t},100.0,1\n" for t in _TS), encoding="utf-8"
    )
    ref = load_reference(DataPaths(raw_dir=raw), spec)
    return ref, spec, tmp_path / "subs"


def _assignment(shuffled: bool = False) -> pd.DataFrame:
    rows = [("PV001", "G1"), ("PV002", "G1"), ("PV003", "G2"), ("PV004", "G2")]
    if shuffled:
        rows = [rows[2], rows[0], rows[3], rows[1]]
    return pd.DataFrame(rows, columns=["site_id", "group"])


def _forecast(values: list[float] | None = None, shuffled: bool = False) -> pd.DataFrame:
    ts = pd.to_datetime(list(_TS))
    rows: list[tuple[str, pd.Timestamp, float]] = []
    rows.extend((g, t, 10.0) for g in ("G1", "G2") for t in ts)
    if values is not None:
        rows = [(g, t, values[i]) for i, (g, t, _) in enumerate(rows)]
    if shuffled:
        rows = list(reversed(rows))
    return pd.DataFrame(rows, columns=["group", "ts", "forecast_kwh"])


def test_roundtrip_bit_identical(ctx: tuple[vmod.SubmissionReference, CompetitionSpec, Path]) -> None:
    ref, spec, out = ctx
    fc = _forecast()
    target = write_submission(_assignment(), fc, "s01", out_root=out, ref=ref, spec=spec)
    report = validate_submission(target / "composition.csv", target / "forecast.csv", ref, spec)
    assert report.ok
    back = pd.read_csv(target / "forecast.csv", dtype=str, keep_default_na=False)
    assert back["ts"].tolist() == [t for g in ("G1", "G2") for t in _TS]
    for got, want in zip(back["forecast_kwh"].tolist(), fc.sort_values(["group", "ts"])["forecast_kwh"].tolist(), strict=True):
        assert float(got) == float(want)
        assert repr(float(got)) == repr(float(want))


def test_sorting_convention(ctx: tuple[vmod.SubmissionReference, CompetitionSpec, Path]) -> None:
    ref, spec, out = ctx
    target = write_submission(_assignment(shuffled=True), _forecast(shuffled=True), "s02", out_root=out, ref=ref, spec=spec)
    comp = pd.read_csv(target / "composition.csv", dtype=str)["site_id"].tolist()
    assert comp == sorted(comp)
    fc = pd.read_csv(target / "forecast.csv", dtype=str)
    assert list(zip(fc["group"], fc["ts"], strict=True)) == sorted(zip(fc["group"], fc["ts"], strict=True))


def test_overwrite_forbidden(ctx: tuple[vmod.SubmissionReference, CompetitionSpec, Path]) -> None:
    ref, spec, out = ctx
    target = write_submission(_assignment(), _forecast(), "s03", out_root=out, ref=ref, spec=spec)
    before = {p.name: p.read_bytes() for p in target.glob("*.csv")}
    with pytest.raises(SubmissionError):
        write_submission(_assignment(), _forecast(), "s03", out_root=out, ref=ref, spec=spec)
    assert {p.name: p.read_bytes() for p in target.glob("*.csv")} == before


def test_invalid_assignment_leaves_no_trace(ctx: tuple[vmod.SubmissionReference, CompetitionSpec, Path]) -> None:
    ref, spec, out = ctx
    bad = pd.DataFrame(
        [("PV001", "G1"), ("PV003", "G1"), ("PV002", "G2"), ("PV004", "G2")],
        columns=["site_id", "group"],
    )
    with pytest.raises(SubmissionError):
        write_submission(bad, _forecast(), "s04", out_root=out, ref=ref, spec=spec)
    assert not (out / "s04").exists()
    assert not (out / ".s04.tmp").exists()


def test_negative_and_nan_rejected(ctx: tuple[vmod.SubmissionReference, CompetitionSpec, Path]) -> None:
    ref, spec, out = ctx
    vals = [10.0] * 6
    vals[0] = -0.5
    with pytest.raises(SubmissionError):
        write_submission(_assignment(), _forecast(vals), "s05a", out_root=out, ref=ref, spec=spec)
    vals[0] = float("nan")
    with pytest.raises(SubmissionError):
        write_submission(_assignment(), _forecast(vals), "s05b", out_root=out, ref=ref, spec=spec)


def test_missing_ts_rejected(ctx: tuple[vmod.SubmissionReference, CompetitionSpec, Path]) -> None:
    ref, spec, out = ctx
    fc = _forecast().iloc[1:].reset_index(drop=True)
    with pytest.raises(SubmissionError):
        write_submission(_assignment(), fc, "s06", out_root=out, ref=ref, spec=spec)


def test_negative_zero_normalized(ctx: tuple[vmod.SubmissionReference, CompetitionSpec, Path]) -> None:
    ref, spec, out = ctx
    target = write_submission(_assignment(), _forecast([-0.0] * 6), "s07", out_root=out, ref=ref, spec=spec)
    text = (target / "forecast.csv").read_text(encoding="utf-8")
    assert "-0.0" not in text
    assert ",0.0" in text


def test_manifest_hashes(ctx: tuple[vmod.SubmissionReference, CompetitionSpec, Path]) -> None:
    ref, spec, out = ctx
    target = write_submission(_assignment(), _forecast(), "s08", out_root=out, ref=ref, spec=spec)
    manifest = SubmissionManifest.model_validate_json((target / "manifest.json").read_text(encoding="utf-8"))
    for name, key in (("composition.csv", "composition_sha256"), ("forecast.csv", "forecast_sha256")):
        digest = hashlib.sha256((target / name).read_bytes()).hexdigest()
        assert digest == getattr(manifest, key)


def test_leaderboard_score_once(ctx: tuple[vmod.SubmissionReference, CompetitionSpec, Path]) -> None:
    ref, spec, out = ctx
    target = write_submission(_assignment(), _forecast(), "s09", out_root=out, ref=ref, spec=spec)
    updated = record_leaderboard_score(target, 1.5)
    assert updated.leaderboard_score == 1.5
    rest_before = updated.model_dump(exclude={"leaderboard_score"})
    again = SubmissionManifest.model_validate_json((target / "manifest.json").read_text(encoding="utf-8"))
    assert again.model_dump(exclude={"leaderboard_score"}) == rest_before
    with pytest.raises(SubmissionError):
        record_leaderboard_score(target, 2.5)


def test_tamper_detected(ctx: tuple[vmod.SubmissionReference, CompetitionSpec, Path]) -> None:
    ref, spec, out = ctx
    target = write_submission(_assignment(), _forecast(), "s10", out_root=out, ref=ref, spec=spec)
    fp = target / "forecast.csv"
    data = fp.read_bytes()
    fp.write_bytes(data[:-1] + (b"1" if data[-1:] != b"1" else b"2"))
    with pytest.raises(SubmissionError):
        record_leaderboard_score(target, 1.0)


def test_manifest_rejects_nonfinite_metric() -> None:
    with pytest.raises(ValidationError, match="non-finite metric"):
        SubmissionManifest(
            sub_id="s",
            created_at=pd.Timestamp("2025-06-01", tz="Asia/Seoul").to_pydatetime(),
            run_id=None,
            composition_id=None,
            git_commit=None,
            composition_sha256="a",
            forecast_sha256="b",
            group_capacity_kw={},
            group_forecast_total_kwh={},
            local_metrics={"w": float("nan")},
            warnings=(),
        )


def test_manifest_json_indent(ctx: tuple[vmod.SubmissionReference, CompetitionSpec, Path]) -> None:
    ref, spec, out = ctx
    target = write_submission(_assignment(), _forecast(), "s11", out_root=out, ref=ref, spec=spec)
    raw = (target / "manifest.json").read_text(encoding="utf-8")
    assert json.loads(raw)["sub_id"] == "s11"
    assert "\n  " in raw


def test_invalid_sub_id_rejected(ctx: tuple[vmod.SubmissionReference, CompetitionSpec, Path]) -> None:
    ref, spec, out = ctx
    with pytest.raises(SubmissionError):
        write_submission(_assignment(), _forecast(), "bad id!", out_root=out, ref=ref, spec=spec)


def test_manifest_contracts() -> None:
    good = {
        "sub_id": "s",
        "created_at": pd.Timestamp("2025-06-01", tz="Asia/Seoul").to_pydatetime(),
        "run_id": None,
        "composition_id": None,
        "git_commit": None,
        "composition_sha256": "a",
        "forecast_sha256": "b",
        "group_capacity_kw": {},
        "group_forecast_total_kwh": {},
        "local_metrics": {},
        "warnings": (),
    }
    with pytest.raises(ValidationError, match="sub_id"):
        SubmissionManifest(**{**good, "sub_id": ""})
    with pytest.raises(ValidationError, match="tz-aware"):
        SubmissionManifest(**{**good, "created_at": datetime(2025, 6, 1)})


def test_git_failure_yields_none_commit(
    ctx: tuple[vmod.SubmissionReference, CompetitionSpec, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.oibc.submit.build as bmod

    def fail(*args: object, **kwargs: object) -> object:
        raise OSError("no git")

    monkeypatch.setattr(bmod.subprocess, "run", fail)
    ref, spec, out = ctx
    target = write_submission(_assignment(), _forecast(), "s12", out_root=out, ref=ref, spec=spec)
    manifest = SubmissionManifest.model_validate_json((target / "manifest.json").read_text(encoding="utf-8"))
    assert manifest.git_commit is None

    class _Failed:
        returncode = 1
        stdout = ""

    monkeypatch.setattr(bmod.subprocess, "run", lambda *a, **k: _Failed())
    target2 = write_submission(_assignment(), _forecast(), "s12b", out_root=out, ref=ref, spec=spec)
    manifest2 = SubmissionManifest.model_validate_json((target2 / "manifest.json").read_text(encoding="utf-8"))
    assert manifest2.git_commit is None


@pytest.mark.parametrize(
    "frame",
    [
        pd.DataFrame(columns=["site_id", "group"]),
        pd.DataFrame([("PV001", "G1")], columns=["site_id", "wrong"]),
        pd.DataFrame([("", "G1"), ("PV002", "G1"), ("PV003", "G2"), ("PV004", "G2")], columns=["site_id", "group"]),
        pd.DataFrame(
            [("PV001", "G1"), ("PV001", "G1"), ("PV003", "G2"), ("PV004", "G2")],
            columns=["site_id", "group"],
        ),
    ],
)
def test_assignment_key_contracts(
    ctx: tuple[vmod.SubmissionReference, CompetitionSpec, Path], frame: pd.DataFrame
) -> None:
    ref, spec, out = ctx
    with pytest.raises(SubmissionError):
        write_submission(frame, _forecast(), "s13", out_root=out, ref=ref, spec=spec)


def test_forecast_frame_contracts(ctx: tuple[vmod.SubmissionReference, CompetitionSpec, Path]) -> None:
    ref, spec, out = ctx
    empty = pd.DataFrame(columns=["group", "ts", "forecast_kwh"])
    with pytest.raises(SubmissionError):
        write_submission(_assignment(), empty, "s14a", out_root=out, ref=ref, spec=spec)
    wrong = _forecast().rename(columns={"forecast_kwh": "value"})
    with pytest.raises(SubmissionError):
        write_submission(_assignment(), wrong, "s14b", out_root=out, ref=ref, spec=spec)
    dupe = pd.concat([_forecast(), _forecast().iloc[:1]], ignore_index=True)
    with pytest.raises(SubmissionError):
        write_submission(_assignment(), dupe, "s14c", out_root=out, ref=ref, spec=spec)


def test_forecast_value_contracts(ctx: tuple[vmod.SubmissionReference, CompetitionSpec, Path]) -> None:
    ref, spec, out = ctx
    bad_str = _forecast().astype({"forecast_kwh": object})
    bad_str.loc[0, "forecast_kwh"] = "abc"
    with pytest.raises(SubmissionError):
        write_submission(_assignment(), bad_str, "s15a", out_root=out, ref=ref, spec=spec)
    inf_vals = [10.0] * 6
    inf_vals[0] = float("inf")
    with pytest.raises(SubmissionError):
        write_submission(_assignment(), _forecast(inf_vals), "s15b", out_root=out, ref=ref, spec=spec)


def test_forecast_ts_contracts(ctx: tuple[vmod.SubmissionReference, CompetitionSpec, Path]) -> None:
    ref, spec, out = ctx
    garbage = _forecast().astype({"ts": object})
    garbage.loc[0, "ts"] = "tomorrow"
    with pytest.raises(SubmissionError):
        write_submission(_assignment(), garbage, "s16a", out_root=out, ref=ref, spec=spec)
    naive = pd.DataFrame(
        {
            "group": ["G1"] * 3 + ["G2"] * 3,
            "ts": [pd.Timestamp(f"2025-06-01 0{h}:00:00") for h in (1, 2, 3)] * 2,
            "forecast_kwh": [10.0] * 6,
        }
    )
    with pytest.raises(SubmissionError):
        write_submission(_assignment(), naive, "s16b", out_root=out, ref=ref, spec=spec)
    unknown = _forecast()
    unknown.loc[0, "ts"] = pd.Timestamp("2026-01-01 01:00", tz="Asia/Seoul")
    with pytest.raises(SubmissionError):
        write_submission(_assignment(), unknown, "s16c", out_root=out, ref=ref, spec=spec)
    bad_group = _forecast()
    bad_group.loc[0, "group"] = "A"
    with pytest.raises(SubmissionError):
        write_submission(_assignment(), bad_group, "s16d", out_root=out, ref=ref, spec=spec)


def test_tmp_collision_rejected(ctx: tuple[vmod.SubmissionReference, CompetitionSpec, Path]) -> None:
    ref, spec, out = ctx
    out.mkdir(parents=True, exist_ok=True)
    (out / ".s17.tmp").mkdir()
    with pytest.raises(SubmissionError):
        write_submission(_assignment(), _forecast(), "s17", out_root=out, ref=ref, spec=spec)


def test_record_score_contracts(ctx: tuple[vmod.SubmissionReference, CompetitionSpec, Path]) -> None:
    ref, spec, out = ctx
    with pytest.raises(SubmissionError):
        record_leaderboard_score(out / "ghost", 1.0)
    target = write_submission(_assignment(), _forecast(), "s18", out_root=out, ref=ref, spec=spec)
    with pytest.raises(SubmissionError):
        record_leaderboard_score(target, float("nan"))
    (target / "composition.csv").write_text("site_id,group\nPV001,G1\n", encoding="utf-8")
    with pytest.raises(SubmissionError, match="composition"):
        record_leaderboard_score(target, 1.0)


def test_same_instant_other_offset_is_duplicate(
    ctx: tuple[vmod.SubmissionReference, CompetitionSpec, Path],
) -> None:
    ref, spec, out = ctx
    utc_copy = _forecast().iloc[:1].copy()
    utc_copy["ts"] = utc_copy["ts"].dt.tz_convert("UTC")
    frame = pd.concat([_forecast().astype({"ts": object}), utc_copy.astype({"ts": object})], ignore_index=True)
    with pytest.raises(SubmissionError, match="duplicate pair"):
        write_submission(_assignment(), frame, "s19", out_root=out, ref=ref, spec=spec)
    assert not (out / "s19").exists()


def test_record_score_corrupt_manifest_or_missing_file(
    ctx: tuple[vmod.SubmissionReference, CompetitionSpec, Path],
) -> None:
    ref, spec, out = ctx
    target = write_submission(_assignment(), _forecast(), "s20", out_root=out, ref=ref, spec=spec)
    (target / "forecast.csv").unlink()
    with pytest.raises(SubmissionError, match=r"cannot read forecast\.csv"):
        record_leaderboard_score(target, 1.0)
    (target / "manifest.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(SubmissionError, match="cannot read manifest"):
        record_leaderboard_score(target, 1.0)


def test_target_created_during_write_not_clobbered(
    ctx: tuple[vmod.SubmissionReference, CompetitionSpec, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    ref, spec, out = ctx
    from src.oibc.submit import build as bmod

    def racing_validate(*args: object, **kwargs: object) -> vmod.ValidationReport:
        (out / "s21").mkdir()
        return validate_submission(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(bmod, "validate_submission", racing_validate)
    with pytest.raises(SubmissionError, match="exists"):
        write_submission(_assignment(), _forecast(), "s21", out_root=out, ref=ref, spec=spec)
    assert list((out / "s21").iterdir()) == []
    assert not (out / ".s21.tmp").exists()


def test_manifest_records_calibration(ctx: tuple[vmod.SubmissionReference, CompetitionSpec, Path]) -> None:
    import json

    ref, spec, out = ctx
    target = write_submission(
        _assignment(), _forecast(), "scal", out_root=out, ref=ref, spec=spec,
        calibration={"a": 0.975, "b": 0.04},
    )
    raw = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    assert raw["calibration"] == {"a": 0.975, "b": 0.04}


def test_manifest_calibration_rejects_nonfinite(ctx: tuple[vmod.SubmissionReference, CompetitionSpec, Path]) -> None:
    ref, spec, out = ctx
    with pytest.raises((SubmissionError, ValidationError)):
        write_submission(
            _assignment(), _forecast(), "sbad", out_root=out, ref=ref, spec=spec,
            calibration={"a": float("nan")},
        )
    assert not (out / "sbad").exists()
    assert not (out / ".sbad.tmp").exists()


def test_manifest_legacy_without_calibration(ctx: tuple[vmod.SubmissionReference, CompetitionSpec, Path]) -> None:
    import json

    ref, spec, out = ctx
    target = write_submission(_assignment(), _forecast(), "sleg", out_root=out, ref=ref, spec=spec)
    manifest_path = target / "manifest.json"
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw.pop("calibration", None)
    manifest_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    updated = record_leaderboard_score(target, 1.0)
    assert updated.calibration is None


def test_manifest_calibration_validator() -> None:
    with pytest.raises(ValidationError, match="non-finite calibration"):
        SubmissionManifest(
            sub_id="s",
            created_at=pd.Timestamp("2025-06-01", tz="Asia/Seoul").to_pydatetime(),
            run_id=None,
            composition_id=None,
            git_commit=None,
            composition_sha256="a",
            forecast_sha256="b",
            group_capacity_kw={},
            group_forecast_total_kwh={},
            local_metrics={},
            warnings=(),
            calibration={"a": float("inf")},
        )


def test_write_submission_calibration_bad_value(ctx: tuple[vmod.SubmissionReference, CompetitionSpec, Path]) -> None:
    ref, spec, out = ctx
    with pytest.raises(SubmissionError, match="calibration"):
        write_submission(
            _assignment(), _forecast(), "scv", out_root=out, ref=ref, spec=spec,
            calibration={"a": "abc"},
        )
    assert not (out / "scv").exists()
