import logging
import runpy
from pathlib import Path

import pandas as pd
import pytest

from src.oibc.config import CompetitionSpec, DataPaths
from src.oibc.data import derived as derived_mod
from src.oibc.data import ingest as ingest_mod
from src.oibc.features import panel as panel_mod
from src.oibc.pipeline import main
from src.oibc.submit import validate as validate_mod
from src.oibc.submit.validate import load_reference
from src.oibc.train import runner as runner_mod


def test_validate_run_dispatches(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []
    monkeypatch.setattr(runner_mod, "run_validation", lambda cfg: calls.append(cfg))
    assert main(["validate-run", "--run-id", "r1", "--hypothesis", "h"]) == 0
    assert len(calls) == 1
    assert main(["validate-run", "--run-id", "r2", "--hypothesis", "h", "--drop-group", "cs", "--weight", "none"]) == 0
    assert len(calls) == 2


def test_submit_unknown_run_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.oibc.evaluate import registry as registry_mod

    monkeypatch.setattr(registry_mod, "read_records", lambda *a, **k: [])
    with pytest.raises(runner_mod.RunnerError, match="unknown run_id"):
        main(["submit", "--run-id", "nope", "--sub-id", "s1"])


def test_submit_missing_bag_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    from datetime import UTC, datetime

    from src.oibc.evaluate import registry as registry_mod
    from src.oibc.evaluate.registry import ExperimentRecord

    rec = ExperimentRecord(
        run_id="r1",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        hypothesis="h",
        config={},
        git_commit=None,
        seeds=(0,),
        folds=("A",),
        metrics={"A/0": {"W": 1.0}},
        comparison={},
    )
    monkeypatch.setattr(registry_mod, "read_records", lambda *a, **k: [rec])
    with pytest.raises(runner_mod.RunnerError, match="bag"):
        main(["submit", "--run-id", "r1", "--sub-id", "s1"])


def test_submit_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import hashlib
    from datetime import UTC, datetime

    import pandas as pd

    from src.oibc.evaluate import registry as registry_mod
    from src.oibc.evaluate.registry import ExperimentRecord
    from src.oibc.submit import build as build_mod
    from src.oibc.config import FeatureConfig, ModelConfig, RunConfig

    panel_path = tmp_path / "panel.parquet"
    panel_path.write_bytes(b"validated panel")
    cfg = RunConfig(run_id="r1", hypothesis="h", features=FeatureConfig(groups=frozenset()), model=ModelConfig(seeds=(0,), weight="none"))
    config = cfg.model_dump(mode="json") | {
        "panel_path": str(panel_path), "panel_sha256": hashlib.sha256(panel_path.read_bytes()).hexdigest(),
        "competition": CompetitionSpec().model_dump(mode="json"),
    }

    rec = ExperimentRecord(
        run_id="r1",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        hypothesis="h",
        config=config,
        git_commit=None,
        seeds=(0,),
        folds=("A",),
        metrics={"A/0": {"W": 1.0}},
        comparison={"bag": {"A": {"W_all": 5.0, "W_warm": 2.0}}},
    )
    monkeypatch.setattr(registry_mod, "read_records", lambda *a, **k: [rec])
    monkeypatch.setattr(pd, "read_parquet", lambda *a, **k: pd.DataFrame({"site_id": ["s"], "current_group": ["A"]}))
    def predict(actual_cfg, **kwargs):
        assert actual_cfg == cfg
        assert kwargs["panel_path"] == panel_path
        return pd.DataFrame({"group": ["G1"], "ts": [pd.Timestamp("2025-06-01 01:00", tz="Asia/Seoul")], "forecast_kwh": [1.0]})

    monkeypatch.setattr(runner_mod, "forecast_eval", predict)
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(build_mod, "write_submission", lambda *a, **k: calls.append((a, k)))
    assert main(["submit", "--run-id", "r1", "--sub-id", "s1"]) == 0
    assert len(calls) == 1
    assert calls[0][1]["local_metrics"] == {"W_all": 5.0, "W_warm": 2.0}


@pytest.mark.parametrize("defect", ["config", "hash", "seeds", "bag_folds", "metrics", "negative"])
def test_submit_rejects_changed_or_incomplete_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, defect: str) -> None:
    import hashlib
    from datetime import UTC, datetime

    from src.oibc.config import ModelConfig, RunConfig
    from src.oibc.evaluate import registry as registry_mod
    from src.oibc.evaluate.registry import ExperimentRecord

    panel = tmp_path / "validated.parquet"
    panel.write_bytes(b"panel")
    cfg = RunConfig(run_id="r1", hypothesis="h", model=ModelConfig(seeds=(0,)))
    config = cfg.model_dump(mode="json") | {
        "panel_path": str(panel), "panel_sha256": hashlib.sha256(panel.read_bytes()).hexdigest(),
        "competition": CompetitionSpec().model_dump(mode="json"),
    }
    bag = {"A": {"W_all": 1.0, "W_warm": 0.5}}
    if defect == "config":
        config = {}
    elif defect == "hash":
        panel.write_bytes(b"changed")
    elif defect == "seeds":
        config["model"]["seeds"] = [1]
    elif defect == "bag_folds":
        bag = {"B": {"W_all": 1.0, "W_warm": 0.5}}
    elif defect == "metrics":
        bag["A"] = {"W_all": 1.0}
    else:
        bag["A"]["W_all"] = -1.0
    rec = ExperimentRecord(
        run_id="r1", created_at=datetime.now(UTC), hypothesis="h", config=config,
        git_commit=None, seeds=(0,), folds=("A",), metrics={"A/0": {"W": 1.0}}, comparison={"bag": bag},
    )
    monkeypatch.setattr(registry_mod, "read_records", lambda: [rec])

    def no_forecast(*args, **kwargs):
        pytest.fail("재현 정보 검증 전에 예측 실행")

    monkeypatch.setattr(runner_mod, "forecast_eval", no_forecast)
    with pytest.raises(runner_mod.RunnerError):
        main(["submit", "--run-id", "r1", "--sub-id", "s1"])


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


def test_features_dispatches_to_build_features(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(panel_mod, "build_features", lambda *args: calls.append(args))
    assert main(["features"]) == 0
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
