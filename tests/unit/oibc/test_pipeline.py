import logging
import runpy
from pathlib import Path

import pandas as pd
import pytest

from src.oibc.config import CompetitionSpec, DataPaths
from src.oibc.data import derived as derived_mod
from src.oibc.data import ingest as ingest_mod
from src.oibc.pipeline import main
from src.oibc.submit import validate as validate_mod
from src.oibc.submit.validate import load_reference


def test_unknown_subcommand_exits_with_usage_error() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["bogus"])
    assert exc.value.code == 2


def test_ingest_dispatches_to_build_canonical(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(ingest_mod, "build_canonical", lambda *args: calls.append(args))
    assert main(["ingest"]) == 0
    assert len(calls) == 1


def test_derive_dispatches_to_build_derived(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(derived_mod, "build_derived", lambda *args: calls.append(args))
    assert main(["derive"]) == 0
    assert len(calls) == 1


def test_cli_entrypoint_dispatch_and_logs(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(derived_mod, "build_derived", lambda *args: calls.append(args))
    monkeypatch.setattr("sys.argv", ["pipeline", "derive"])
    with caplog.at_level(logging.INFO), pytest.raises(SystemExit) as exc:
        runpy.run_path("src/oibc/pipeline.py", run_name="__main__")
    assert exc.value.code == 0
    assert len(calls) == 1
    assert caplog.records[0].message == "[SYS] pipeline derive start"
    assert "[SYS] pipeline derive done in" in caplog.records[-1].message


def test_pipeline_propagates_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: object) -> None:
        raise ValueError("invalid source")

    monkeypatch.setattr(derived_mod, "build_derived", fail)
    with pytest.raises(ValueError, match="invalid source"):
        main(["derive"])


_TS = (
    "2025-06-01 01:00:00+09:00",
    "2025-06-01 02:00:00+09:00",
    "2025-06-01 03:00:00+09:00",
)


def _mini_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    spec = CompetitionSpec(
        n_sites=4,
        min_group_capacity_kw=200.0,
        train_end=pd.Timestamp("2025-06-01 00:00", tz="Asia/Seoul"),
        eval_start=pd.Timestamp("2025-06-01 01:00", tz="Asia/Seoul"),
        eval_end=pd.Timestamp("2025-06-01 03:00", tz="Asia/Seoul"),
        n_eval_hours=3,
    )
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "sites.csv").write_text(
        "site_id,capacity_kw\nPV001,100\nPV002,100\nPV003,100\nPV004,100\n", encoding="utf-8"
    )
    (raw / "smp.csv").write_text(
        "ts,smp,k\n" + "".join(f"{t},100.0,1\n" for t in _TS), encoding="utf-8"
    )
    monkeypatch.setattr(validate_mod, "DataPaths", lambda *a: DataPaths(raw_dir=raw))  # type: ignore[arg-type]
    monkeypatch.setattr(validate_mod, "CompetitionSpec", lambda *a, **k: spec)  # type: ignore[assignment]
    comp = tmp_path / "composition.csv"
    comp.write_text(
        "site_id,group\nPV001,G1\nPV002,G1\nPV003,G2\nPV004,G2\n", encoding="utf-8"
    )
    fc = tmp_path / "forecast.csv"
    lines = ["group,ts,forecast_kwh"]
    lines.extend(f"{g},{t},10.0" for g in ("G1", "G2") for t in _TS)
    fc.write_text("\n".join(lines) + "\n", encoding="utf-8")
    ref = load_reference(DataPaths(raw_dir=raw), spec)
    assert ref.score_ts == _TS
    return comp, fc


def test_validate_submission_exit_codes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    comp, fc = _mini_env(tmp_path, monkeypatch)
    assert main(["validate-submission", "--composition", str(comp), "--forecast", str(fc)]) == 0
    bad = tmp_path / "bad.csv"
    bad.write_text("site_id,group\nPV001,G1\n", encoding="utf-8")
    assert main(["validate-submission", "--composition", str(bad), "--forecast", str(fc)]) == 1


def test_validate_submission_requires_forecast(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    comp, _ = _mini_env(tmp_path, monkeypatch)
    with pytest.raises(SystemExit):
        main(["validate-submission", "--composition", str(comp)])


def test_validate_submission_warnings_keep_exit_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    comp, fc = _mini_env(tmp_path, monkeypatch)
    fc.write_text(fc.read_text(encoding="utf-8").replace(",10.0", ",1000.0"), encoding="utf-8")
    with caplog.at_level(logging.INFO):
        assert main(["validate-submission", "--composition", str(comp), "--forecast", str(fc)]) == 0
    assert "over capacity" in caplog.text
