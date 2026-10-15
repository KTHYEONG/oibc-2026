import logging
import runpy

import pytest

from src.oibc.data import derived as derived_mod
from src.oibc.data import ingest as ingest_mod
from src.oibc.pipeline import main


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
