import logging
from types import SimpleNamespace
import runpy
from pathlib import Path

import numpy as np
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
    monkeypatch.setattr(runner_mod, "run_validation", lambda cfg, **k: calls.append((cfg, k)))
    assert main(["validate-run", "--run-id", "r1", "--hypothesis", "h"]) == 0
    assert len(calls) == 1
    assert calls[0][1].get("save_oof", False) is False
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
        "composition_sha256": "abc",
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
    monkeypatch.setattr(
        runner_mod, "resolve_composition",
        lambda *a, **k: (pd.DataFrame({"site_id": ["s"], "group": ["G1"]}), "abc"),
    )
    def predict(actual_cfg, **kwargs):
        assert actual_cfg == cfg
        assert kwargs["panel_path"] == panel_path
        return pd.DataFrame({"group": ["G1"], "ts": [pd.Timestamp("2025-06-01 01:00", tz="Asia/Seoul")], "forecast_kwh": [1.0]})

    monkeypatch.setattr("src.oibc.pipeline._check_input_breaks", lambda *a, **k: [])
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


def test_pipeline_compose_writes_composition(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.oibc.pipeline as pipeline_mod
    from src.oibc.compose import store as store_mod
    from src.oibc.config import ArtifactPaths, DataPaths

    canon = tmp_path / "canonical"
    canon.mkdir(exist_ok=True)
    comps = tmp_path / "comps"
    import numpy as np

    n = 253
    ids = [f"PV{i:03d}" for i in range(1, n + 1)]
    sites = pd.DataFrame(
        {"site_id": ids, "capacity_kw": [200.0] * n,
         "lat": 33.0 + (np.arange(n) % 23) * 0.1, "lon": 124.0 + (np.arange(n) // 23) * 0.1,
         "current_group": ["A" if i % 2 == 0 else "B" for i in range(n)]}
    )
    sites.to_parquet(canon / "sites.parquet", index=False)
    monkeypatch.setattr(pipeline_mod, "DataPaths", lambda *a, **k: DataPaths(canonical_dir=canon))
    paths = ArtifactPaths(root=tmp_path, registry=tmp_path / "registry.jsonl",
                          submissions=tmp_path / "subs", runs=tmp_path / "runs", compositions=comps)
    monkeypatch.setattr(store_mod, "ArtifactPaths", lambda *a, **k: paths)
    assert main(["compose", "--comp-id", "t1"]) == 0
    assert (comps / "t1.csv").exists() and (comps / "t1.json").exists()
    from src.oibc.compose.store import load_composition

    load_composition("t1", pd.read_parquet(canon / "sites.parquet"))


def test_pipeline_compose_refuses_existing_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.oibc.pipeline as pipeline_mod
    from src.oibc.compose import store as store_mod
    from src.oibc.compose.constraints import CompositionError
    from src.oibc.config import ArtifactPaths, DataPaths

    canon = tmp_path / "canonical"
    canon.mkdir(exist_ok=True)
    comps = tmp_path / "comps"
    import numpy as np

    n = 253
    ids = [f"PV{i:03d}" for i in range(1, n + 1)]
    sites = pd.DataFrame(
        {"site_id": ids, "capacity_kw": [200.0] * n,
         "lat": 33.0 + (np.arange(n) % 23) * 0.1, "lon": 124.0 + (np.arange(n) // 23) * 0.1,
         "current_group": ["A" if i % 2 == 0 else "B" for i in range(n)]}
    )
    sites.to_parquet(canon / "sites.parquet", index=False)
    monkeypatch.setattr(pipeline_mod, "DataPaths", lambda *a, **k: DataPaths(canonical_dir=canon))
    paths = ArtifactPaths(root=tmp_path, registry=tmp_path / "registry.jsonl",
                          submissions=tmp_path / "subs", runs=tmp_path / "runs", compositions=comps)
    monkeypatch.setattr(store_mod, "ArtifactPaths", lambda *a, **k: paths)
    assert main(["compose", "--comp-id", "t1"]) == 0
    before_csv = (comps / "t1.csv").read_bytes()
    before_json = (comps / "t1.json").read_bytes()
    with pytest.raises(CompositionError):
        main(["compose", "--comp-id", "t1"])
    assert (comps / "t1.csv").read_bytes() == before_csv
    assert (comps / "t1.json").read_bytes() == before_json


def test_pipeline_validate_run_passes_composition(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []
    monkeypatch.setattr(runner_mod, "run_validation", lambda cfg, **k: calls.append(cfg))
    assert main(["validate-run", "--run-id", "r1", "--hypothesis", "h", "--composition", "t1"]) == 0
    assert calls[0].composition_id == "t1"


def test_pipeline_validate_run_passes_save_oof(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[object, object]] = []
    monkeypatch.setattr(runner_mod, "run_validation", lambda cfg, **k: calls.append((cfg, k.get("save_oof"))))
    assert main(["validate-run", "--run-id", "r1", "--hypothesis", "h", "--save-oof"]) == 0
    assert calls[0][1] is True
    calls.clear()
    assert main(["validate-run", "--run-id", "r2", "--hypothesis", "h"]) == 0
    assert calls[0][1] is False


@pytest.mark.parametrize("defect", ["before", "during", "null"])
def test_pipeline_submit_rejects_composition_hash_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, defect: str,
) -> None:
    import hashlib
    from datetime import UTC, datetime

    import src.oibc.pipeline as pipeline_mod
    from src.oibc.compose import store as store_mod
    from src.oibc.config import ArtifactPaths, ComposeConfig, DataPaths, RunConfig
    from src.oibc.compose.store import assignment_sha256, write_composition
    from src.oibc.evaluate import registry as registry_mod
    from src.oibc.evaluate.registry import ExperimentRecord
    from src.oibc.submit import build as build_mod

    canon = tmp_path / "canonical"
    canon.mkdir(exist_ok=True)
    comps = tmp_path / "comps"
    import numpy as np

    n = 253
    ids = [f"PV{i:03d}" for i in range(1, n + 1)]
    sites = pd.DataFrame(
        {"site_id": ids, "capacity_kw": [200.0] * n,
         "lat": 33.0 + (np.arange(n) % 23) * 0.1, "lon": 124.0 + (np.arange(n) // 23) * 0.1,
         "current_group": ["A" if i % 2 == 0 else "B" for i in range(n)]}
    )
    sites.to_parquet(canon / "sites.parquet", index=False)
    paths = ArtifactPaths(root=tmp_path, registry=tmp_path / "registry.jsonl",
                          submissions=tmp_path / "subs", runs=tmp_path / "runs", compositions=comps)
    monkeypatch.setattr(pipeline_mod, "DataPaths", lambda *a, **k: DataPaths(canonical_dir=canon))
    monkeypatch.setattr(store_mod, "ArtifactPaths", lambda *a, **k: paths)
    half = n // 2
    assignment = pd.DataFrame({"site_id": ids, "group": ["G1"] * half + ["G2"] * (n - half)})
    write_composition(assignment, "t1", ComposeConfig(), sites, paths=paths)
    panel_path = tmp_path / "panel.parquet"
    panel_path.write_bytes(b"validated panel")
    cfg = RunConfig(run_id="r1", hypothesis="h", composition_id="t1")
    config = cfg.model_dump(mode="json") | {
        "panel_path": str(panel_path), "panel_sha256": hashlib.sha256(panel_path.read_bytes()).hexdigest(),
        "composition_sha256": None if defect == "null" else assignment_sha256(assignment),
        "competition": CompetitionSpec().model_dump(mode="json"),
    }
    rec = ExperimentRecord(
        run_id="r1", created_at=datetime(2026, 1, 1, tzinfo=UTC), hypothesis="h", config=config,
        git_commit=None, seeds=tuple(cfg.model.seeds), folds=("A",),
        metrics={}, comparison={"bag": {"A": {"W_all": 1.0, "W_warm": 0.5}}},
    )
    monkeypatch.setattr(registry_mod, "read_records", lambda *a, **k: [rec])

    def replace_composition():
        (comps / "t1.csv").unlink()
        (comps / "t1.json").unlink()
        changed = assignment.copy()
        changed.loc[[0, half], "group"] = ["G2", "G1"]
        write_composition(changed, "t1", ComposeConfig(), sites, paths=paths)

    if defect == "before":
        replace_composition()

    def predict(*args, **kwargs):
        assert defect == "during", "배정 검증 전에 예측 실행"
        replace_composition()
        return pd.DataFrame()

    monkeypatch.setattr("src.oibc.pipeline._check_input_breaks", lambda *a, **k: [])
    monkeypatch.setattr(runner_mod, "forecast_eval", predict)
    monkeypatch.setattr(build_mod, "write_submission", lambda *a, **k: pytest.fail("drift 이전에 실패해야 함"))
    with pytest.raises(runner_mod.RunnerError, match="composition"):
        main(["submit", "--run-id", "r1", "--sub-id", "s1"])


def test_pipeline_submit_legacy_reference_record_allowed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import hashlib
    from datetime import UTC, datetime

    import src.oibc.pipeline as pipeline_mod
    from src.oibc.config import CompetitionSpec, DataPaths, RunConfig
    from src.oibc.evaluate import registry as registry_mod
    from src.oibc.evaluate.registry import ExperimentRecord
    from src.oibc.submit import build as build_mod

    panel_path = tmp_path / "panel.parquet"
    panel_path.write_bytes(b"validated panel")
    cfg = RunConfig(run_id="r1", hypothesis="h")
    config = cfg.model_dump(mode="json")
    config.pop("composition_id")
    config |= {
        "panel_path": str(panel_path), "panel_sha256": hashlib.sha256(panel_path.read_bytes()).hexdigest(),
        "competition": CompetitionSpec().model_dump(mode="json"),
    }
    rec = ExperimentRecord(
        run_id="r1", created_at=datetime(2026, 1, 1, tzinfo=UTC), hypothesis="h", config=config,
        git_commit=None, seeds=tuple(cfg.model.seeds), folds=("A",),
        metrics={}, comparison={"bag": {"A": {"W_all": 1.0, "W_warm": 0.5}}},
    )
    monkeypatch.setattr(registry_mod, "read_records", lambda *a, **k: [rec])
    monkeypatch.setattr("src.oibc.pipeline._check_input_breaks", lambda *a, **k: [])
    monkeypatch.setattr(runner_mod, "forecast_eval", lambda *a, **k: pd.DataFrame({"ok": [1]}))
    monkeypatch.setattr(
        pd, "read_parquet",
        lambda *a, **k: pd.DataFrame({"site_id": ["s"], "current_group": ["A"]}),
    )
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(build_mod, "write_submission", lambda *a, **k: calls.append((a, k)))
    monkeypatch.setattr(
        runner_mod, "resolve_composition",
        lambda *a, **k: (pd.DataFrame({"site_id": ["s"], "group": ["G1"]}), "abc"),
    )
    assert main(["submit", "--run-id", "r1", "--sub-id", "s1"]) == 0
    assert len(calls) == 1


def test_pipeline_submit_missing_sha_non_reference_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hashlib
    from datetime import UTC, datetime

    from src.oibc.config import CompetitionSpec, RunConfig
    from src.oibc.evaluate import registry as registry_mod
    from src.oibc.evaluate.registry import ExperimentRecord
    from src.oibc.submit import build as build_mod

    panel_path = tmp_path / "panel.parquet"
    panel_path.write_bytes(b"validated panel")
    cfg = RunConfig(run_id="r1", hypothesis="h", composition_id="t1")
    config = cfg.model_dump(mode="json") | {
        "panel_path": str(panel_path), "panel_sha256": hashlib.sha256(panel_path.read_bytes()).hexdigest(),
        "competition": CompetitionSpec().model_dump(mode="json"),
    }
    assert "composition_sha256" not in config
    rec = ExperimentRecord(
        run_id="r1", created_at=datetime(2026, 1, 1, tzinfo=UTC), hypothesis="h", config=config,
        git_commit=None, seeds=tuple(cfg.model.seeds), folds=("A",),
        metrics={}, comparison={"bag": {"A": {"W_all": 1.0, "W_warm": 0.5}}},
    )
    monkeypatch.setattr(registry_mod, "read_records", lambda *a, **k: [rec])
    monkeypatch.setattr(runner_mod, "forecast_eval", lambda *a, **k: pd.DataFrame({"ok": [1]}))
    monkeypatch.setattr(
        pd, "read_parquet",
        lambda *a, **k: pd.DataFrame({"site_id": ["s"], "current_group": ["A"]}),
    )
    monkeypatch.setattr(
        runner_mod, "resolve_composition",
        lambda *a, **k: (pd.DataFrame({"site_id": ["s"], "group": ["G1"]}), "abc"),
    )
    monkeypatch.setattr(build_mod, "write_submission", lambda *a, **k: pytest.fail("sha 없이 제출 진행"))
    with pytest.raises(runner_mod.RunnerError, match="composition_sha256"):
        main(["submit", "--run-id", "r1", "--sub-id", "s1"])


def test_pipeline_report_netting_writes_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json

    import src.oibc.pipeline as pipeline_mod
    from src.oibc.train import runner as runner_mod

    report = {
        "run_id": "r1",
        "pred_col": "pred_bag",
        "n_random": 4,
        "seed": 0,
        "compositions": {"current_group": "abc"},
        "folds": {},
    }
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        runner_mod, "netting_report", lambda *a, **k: calls.append((a, k)) or report
    )
    import src.oibc.config as config_mod

    paths = config_mod.ArtifactPaths(
        root=tmp_path,
        registry=tmp_path / "registry.jsonl",
        submissions=tmp_path / "subs",
        runs=tmp_path / "runs",
        compositions=tmp_path / "comps",
    )
    monkeypatch.setattr(config_mod, "ArtifactPaths", lambda *a, **k: paths)
    monkeypatch.setattr(pipeline_mod, "ArtifactPaths", lambda *a, **k: paths, raising=False)
    run_dir = paths.runs / "r1"
    run_dir.mkdir(parents=True)
    assert main(["report-netting", "--run-id", "r1", "--composition", "current_group", "--n-random", "4", "--seed", "0"]) == 0
    out = run_dir / "netting_n4_seed0.json"
    assert out.exists()
    assert json.loads(out.read_text(encoding="utf-8")) == report
    before = out.read_bytes()
    with pytest.raises(FileExistsError):
        main(["report-netting", "--run-id", "r1", "--composition", "current_group", "--n-random", "4", "--seed", "0"])
    assert out.read_bytes() == before


def test_validate_run_regional_weight_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.oibc.config import RegionalConfig

    calls: list[object] = []
    monkeypatch.setattr(runner_mod, "run_validation", lambda cfg, **k: calls.append(cfg))
    assert main(["validate-run", "--run-id", "r1", "--hypothesis", "h", "--regional-weight", "0.6"]) == 0
    assert len(calls) == 1
    assert calls[0].model.regional.weight == 0.6
    assert calls[0].model.regional == RegionalConfig(weight=0.6)


def test_validate_run_regional_default_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.oibc.config import RegionalConfig

    calls: list[object] = []
    monkeypatch.setattr(runner_mod, "run_validation", lambda cfg, **k: calls.append(cfg))
    assert main(["validate-run", "--run-id", "r1", "--hypothesis", "h"]) == 0
    assert calls[0].model.regional == RegionalConfig()


def test_validate_run_regional_weight_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []
    monkeypatch.setattr(runner_mod, "run_validation", lambda cfg, **k: calls.append(cfg))
    with pytest.raises(ValueError, match="weight"):
        main(["validate-run", "--run-id", "r1", "--hypothesis", "h", "--regional-weight", "1.5"])
    assert calls == []


def _cal_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import src.oibc.config as config_mod

    paths = config_mod.ArtifactPaths(
        root=tmp_path,
        registry=tmp_path / "registry.jsonl",
        submissions=tmp_path / "subs",
        runs=tmp_path / "runs",
        compositions=tmp_path / "comps",
    )
    monkeypatch.setattr(config_mod, "ArtifactPaths", lambda *a, **k: paths)
    return paths


def test_report_calibration_writes_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.oibc.pipeline as pipeline_mod
    from src.oibc.train import runner as runner_mod

    report = {"run_id": "r1", "decision": "adopt", "config": {}, "production": {}}
    monkeypatch.setattr(runner_mod, "calibration_report", lambda *a, **k: report)
    paths = _cal_paths(tmp_path, monkeypatch)
    monkeypatch.setattr(pipeline_mod, "ArtifactPaths", lambda *a, **k: paths, raising=False)
    assert main(["report-calibration", "--run-id", "r1"]) == 0
    out = paths.runs / "r1" / "calibration.json"
    assert out.exists()
    import json

    assert json.loads(out.read_text(encoding="utf-8"))["decision"] == "adopt"
    before = out.read_bytes()
    with pytest.raises(FileExistsError):
        main(["report-calibration", "--run-id", "r1"])
    assert out.read_bytes() == before


def _submit_cal_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, decision: str):
    import hashlib
    import json
    from datetime import UTC, datetime

    import src.oibc.pipeline as pipeline_mod
    from src.oibc.evaluate import registry as registry_mod
    from src.oibc.evaluate.registry import ExperimentRecord
    from src.oibc.config import FeatureConfig, ModelConfig, RunConfig

    paths = _cal_paths(tmp_path, monkeypatch)
    monkeypatch.setattr(pipeline_mod, "ArtifactPaths", lambda *a, **k: paths, raising=False)
    panel_path = tmp_path / "panel.parquet"
    panel_path.write_bytes(b"validated panel")
    panel_hash = hashlib.sha256(panel_path.read_bytes()).hexdigest()
    cfg = RunConfig(run_id="r1", hypothesis="h", features=FeatureConfig(groups=frozenset()), model=ModelConfig(seeds=(0,), weight="none"))
    config = cfg.model_dump(mode="json") | {
        "panel_path": str(panel_path), "panel_sha256": panel_hash,
        "composition_sha256": "abc",
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
    monkeypatch.setattr(
        runner_mod, "resolve_composition",
        lambda *a, **k: (pd.DataFrame({"site_id": ["s"], "group": ["G1"]}), "abc"),
    )
    run_dir = paths.runs / "r1"
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": "r1",
        "panel_sha256": panel_hash,
        "decision": decision,
        "reasons": [],
        "config": {"cloud_vars": ["wa_cloud_fleet", "wb_cloud_fleet"], "a_min": 0.90, "a_max": 1.05,
                   "b_min": -0.05, "b_max": 0.15, "step": 0.0025},
        "production": {"a": 0.975, "b": 0.04, "at_boundary": False, "n_fit_hours": 10,
                       "rel_delta_in_sample": -0.02},
    }
    (run_dir / "calibration.json").write_text(json.dumps(payload), encoding="utf-8")
    return panel_path, run_dir


def test_submit_calibration_requires_adopt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _submit_cal_env(tmp_path, monkeypatch, "reject")

    def no_forecast(*args, **kwargs):
        pytest.fail("reject 보고서로 예측 실행")

    monkeypatch.setattr(runner_mod, "forecast_eval", no_forecast)
    with pytest.raises(runner_mod.RunnerError):
        main(["submit", "--run-id", "r1", "--sub-id", "s1", "--calibration"])


def test_submit_calibration_passes_production_params(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.oibc.submit import build as build_mod

    _submit_cal_env(tmp_path, monkeypatch, "adopt")
    seen: dict[str, object] = {}

    def predict(cfg, **kwargs):
        seen.update(kwargs)
        return pd.DataFrame({"group": ["G1"], "ts": [pd.Timestamp("2025-06-01 01:00", tz="Asia/Seoul")], "forecast_kwh": [1.0]})

    monkeypatch.setattr("src.oibc.pipeline._check_input_breaks", lambda *a, **k: [])
    monkeypatch.setattr(runner_mod, "forecast_eval", predict)
    writes: list[tuple[object, ...]] = []
    monkeypatch.setattr(build_mod, "write_submission", lambda *a, **k: writes.append((a, k)))
    assert main(["submit", "--run-id", "r1", "--sub-id", "s1", "--calibration"]) == 0
    cal = seen["calibration"]
    assert (cal.a, cal.b) == (0.975, 0.04)
    assert writes[0][1]["calibration"] == {"a": 0.975, "b": 0.04}
    assert writes[0][1]["note"] == "calibration: production a,b from calibration.json"


def test_submit_without_flag_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.oibc.submit import build as build_mod

    _submit_cal_env(tmp_path, monkeypatch, "adopt")
    seen: dict[str, object] = {}

    def predict(cfg, **kwargs):
        seen.update(kwargs)
        return pd.DataFrame({"group": ["G1"], "ts": [pd.Timestamp("2025-06-01 01:00", tz="Asia/Seoul")], "forecast_kwh": [1.0]})

    monkeypatch.setattr("src.oibc.pipeline._check_input_breaks", lambda *a, **k: [])
    monkeypatch.setattr(runner_mod, "forecast_eval", predict)
    monkeypatch.setattr(build_mod, "write_submission", lambda *a, **k: None)
    assert main(["submit", "--run-id", "r1", "--sub-id", "s1"]) == 0
    assert seen.get("calibration") is None


@pytest.mark.parametrize(
    "defect", ["missing", "nondict", "run", "panel", "prodtype", "prodkeys", "nonfinite", "boundary", "config"]
)
def test_submit_calibration_guard_rejects(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, defect: str) -> None:
    import json

    _, run_dir = _submit_cal_env(tmp_path, monkeypatch, "adopt")
    cal_path = run_dir / "calibration.json"
    payload = json.loads(cal_path.read_text(encoding="utf-8"))
    if defect == "missing":
        cal_path.unlink()
    else:
        if defect == "nondict":
            payload = [1, 2]
        elif defect == "run":
            payload["run_id"] = "other"
        elif defect == "panel":
            payload["panel_sha256"] = "dead"
        elif defect == "prodtype":
            payload["production"] = [1]
        elif defect == "prodkeys":
            payload["production"] = {"a": 1.0}
        elif defect == "nonfinite":
            payload["production"]["a"] = float("inf")
        elif defect == "boundary":
            payload["production"]["at_boundary"] = True
        else:
            payload["config"] = {"cloud_vars": []}
        cal_path.write_text(json.dumps(payload, allow_nan=True), encoding="utf-8")

    def no_forecast(*args, **kwargs):
        pytest.fail("가드 위반 보고서로 예측 실행")

    monkeypatch.setattr(runner_mod, "forecast_eval", no_forecast)
    with pytest.raises(runner_mod.RunnerError):
        main(["submit", "--run-id", "r1", "--sub-id", "s1", "--calibration"])


def test_validate_run_passes_family_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []
    monkeypatch.setattr(runner_mod, "run_validation", lambda cfg, **k: calls.append(cfg))
    assert main(["validate-run", "--run-id", "r1", "--hypothesis", "h", "--cat-weight", "0.2", "--mlp-weight", "0.2"]) == 0
    assert len(calls) == 1
    assert calls[0].model.families.cat_weight == 0.2
    assert calls[0].model.families.mlp_weight == 0.2


def test_validate_run_family_weights_default_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []
    monkeypatch.setattr(runner_mod, "run_validation", lambda cfg, **k: calls.append(cfg))
    assert main(["validate-run", "--run-id", "r1", "--hypothesis", "h"]) == 0
    assert calls[0].model.families.cat_weight == 0.0
    assert calls[0].model.families.mlp_weight == 0.0


def test_validate_run_rejects_family_weight_sum(monkeypatch: pytest.MonkeyPatch) -> None:
    from pydantic import ValidationError

    calls: list[object] = []
    monkeypatch.setattr(runner_mod, "run_validation", lambda cfg, **k: calls.append(cfg))
    with pytest.raises(ValidationError):
        main(["validate-run", "--run-id", "r1", "--hypothesis", "h", "--cat-weight", "0.6", "--mlp-weight", "0.5"])
    assert calls == []


def test_validate_run_drop_var_and_clip_reach_run_config(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.oibc.config import FeatureConfig

    calls: list[object] = []
    monkeypatch.setattr(runner_mod, "run_validation", lambda cfg, **k: calls.append(cfg))
    argv = ["validate-run", "--run-id", "r1", "--hypothesis", "h", "--drop-var", "wa_ground_press", "--mlp-clip", "0.001", "0.999"]
    assert main(argv) == 0
    cfg = calls[0]
    default = FeatureConfig()
    assert "wa_ground_press" not in cfg.features.wa_vars
    assert cfg.features.wa_vars == tuple(v for v in default.wa_vars if v != "wa_ground_press")
    assert cfg.features.wb_vars == default.wb_vars
    assert cfg.model.families.mlp_clip_quantiles == (0.001, 0.999)


def test_validate_run_rejects_unknown_drop_var(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []
    monkeypatch.setattr(runner_mod, "run_validation", lambda cfg, **k: calls.append(cfg))
    with pytest.raises(SystemExit):
        main(["validate-run", "--run-id", "r1", "--hypothesis", "h", "--drop-var", "not_a_var"])
    assert calls == []


def test_validate_run_defaults_keep_features_and_clip_off(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.oibc.config import FeatureConfig

    calls: list[object] = []
    monkeypatch.setattr(runner_mod, "run_validation", lambda cfg, **k: calls.append(cfg))
    assert main(["validate-run", "--run-id", "r1", "--hypothesis", "h"]) == 0
    assert calls[0].features == FeatureConfig(groups=frozenset(FeatureConfig().groups))
    assert calls[0].model.families.mlp_clip_quantiles is None


def test_submit_aborts_on_unallowed_break(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.oibc.data.drift as drift_mod
    from src.oibc.config import CompetitionSpec, FeatureConfig, RunConfig
    from src.oibc.pipeline import _check_input_breaks
    from src.oibc.train.runner import RunnerError

    cfg = RunConfig(run_id="r1", hypothesis="h", features=FeatureConfig())
    spec = CompetitionSpec()
    breaks = {"wa_ground_press": pd.Timestamp("2026-01-23", tz="Asia/Seoul")}
    monkeypatch.setattr(pd, "read_parquet", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(drift_mod, "detect_breaks", lambda fc, p, variables, end, c: {v: breaks.get(v) for v in variables})
    with pytest.raises(RunnerError, match="input break wa_ground_press from 2026-01-23"):
        _check_input_breaks(cfg, spec, set())
    notes = _check_input_breaks(cfg, spec, {"wa_ground_press"})
    assert notes == ["break:wa_ground_press:2026-01-23"]


def _write_sub(root: Path, name: str, run_id: str, comp_sha: str, groups_value: float, ts_list: list[str]) -> None:
    import json as _json

    sub = root / name
    sub.mkdir(parents=True)
    (sub / "composition.csv").write_text("site_id,group\ns1,G1\ns2,G2\n", encoding="utf-8")
    rows = ["group,ts,forecast_kwh"]
    rows.extend(f"{g},{t},{groups_value}" for g in ("G1", "G2") for t in ts_list)
    (sub / "forecast.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    manifest = {
        "sub_id": name,
        "created_at": "2026-11-02T10:00:00+09:00",
        "run_id": run_id,
        "composition_id": "geo64-s0",
        "git_commit": None,
        "composition_sha256": comp_sha,
        "forecast_sha256": "0" * 64,
        "group_capacity_kw": {"G1": 20000.0, "G2": 20000.0},
        "group_forecast_total_kwh": {"G1": 1.0, "G2": 1.0},
        "local_metrics": {"W_all": 1.0, "W_warm": 1.0},
        "warnings": [],
    }
    (sub / "manifest.json").write_text(_json.dumps(manifest), encoding="utf-8")


def _splice_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, base_sha: str, fb_sha: str):
    import src.oibc.config as config_mod
    from src.oibc.config import CompetitionSpec, FeatureConfig, RunConfig

    root = tmp_path / "submissions"
    root.mkdir()
    paths = config_mod.ArtifactPaths(submissions=root)
    monkeypatch.setattr(config_mod, "ArtifactPaths", lambda *a, **k: paths)
    default = FeatureConfig()
    base_features = default
    fb_features = FeatureConfig(wa_vars=tuple(v for v in default.wa_vars if v != "wa_ground_press"))
    spec_cfg = CompetitionSpec().model_dump(mode="json")
    records = []
    for run_id, feats in (("base-run", base_features), ("fb-run", fb_features)):
        cfg = RunConfig(run_id=run_id, hypothesis="h", features=feats).model_dump(mode="json")
        cfg["competition"] = spec_cfg
        records.append(SimpleNamespace(run_id=run_id, config=cfg))
    import src.oibc.evaluate.registry as registry_mod

    monkeypatch.setattr(registry_mod, "read_records", lambda *a, **k: records)
    ts = ["2026-01-22T12:00:00+09:00", "2026-01-23T01:00:00+09:00"]
    _write_sub(root, "base", "base-run", base_sha, 1.0, ts)
    _write_sub(root, "fb", "fb-run", fb_sha, 2.0, ts)
    return root


def test_splice_submission_rejects_composition_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.oibc.pipeline import splice_submission
    from src.oibc.train.runner import RunnerError

    _splice_env(tmp_path, monkeypatch, base_sha="a" * 64, fb_sha="b" * 64)
    with pytest.raises(RunnerError, match="composition hash"):
        splice_submission("base", "fb", "spliced")


def test_splice_submission_rejects_fallback_with_break(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.oibc.data.drift as drift_mod
    from src.oibc.pipeline import splice_submission
    from src.oibc.train.runner import RunnerError

    _splice_env(tmp_path, monkeypatch, base_sha="a" * 64, fb_sha="a" * 64)
    monkeypatch.setattr(pd, "read_parquet", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(
        drift_mod,
        "detect_breaks",
        lambda fc, p, variables, end, cfg: {v: pd.Timestamp("2026-01-23", tz="Asia/Seoul") for v in variables},
    )
    with pytest.raises(RunnerError, match="fallback input break"):
        splice_submission("base", "fb", "spliced")


def test_splice_submission_switches_at_first_break(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.oibc.data.drift as drift_mod
    import src.oibc.submit.build as build_mod
    from src.oibc.pipeline import splice_submission

    _splice_env(tmp_path, monkeypatch, base_sha="a" * 64, fb_sha="a" * 64)
    monkeypatch.setattr(pd, "read_parquet", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(
        drift_mod,
        "detect_breaks",
        lambda fc, p, variables, end, cfg: {
            v: (pd.Timestamp("2026-01-23", tz="Asia/Seoul") if v == "wa_ground_press" else None) for v in variables
        },
    )
    calls: list[tuple[tuple, dict]] = []
    monkeypatch.setattr(build_mod, "write_submission", lambda *a, **k: calls.append((a, k)))
    splice_submission("base", "fb", "spliced")
    (assignment, forecast, sub_id), kwargs = calls[0]
    assert sub_id == "spliced"
    assert kwargs["note"].startswith("splice base=base fallback=fb switch=2026-01-23 vars=wa_ground_press")
    assert kwargs["run_id"] == "fb-run"
    keys = zip(forecast["group"], forecast["ts"].dt.strftime("%Y-%m-%d %H"), strict=True)
    values = dict(zip(keys, forecast["forecast_kwh"], strict=True))
    assert values[("G1", "2026-01-22 12")] == 1.0
    assert values[("G1", "2026-01-23 01")] == 2.0


def test_submit_records_allowed_break_in_note(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import hashlib
    from datetime import UTC, datetime

    from src.oibc.config import FeatureConfig, ModelConfig, RunConfig
    from src.oibc.evaluate import registry as registry_mod
    from src.oibc.evaluate.registry import ExperimentRecord
    from src.oibc.submit import build as build_mod

    panel_path = tmp_path / "panel.parquet"
    panel_path.write_bytes(b"validated panel")
    cfg = RunConfig(run_id="r1", hypothesis="h", features=FeatureConfig(groups=frozenset()), model=ModelConfig(seeds=(0,), weight="none"))
    config = cfg.model_dump(mode="json") | {
        "panel_path": str(panel_path), "panel_sha256": hashlib.sha256(panel_path.read_bytes()).hexdigest(),
        "composition_sha256": "abc", "competition": CompetitionSpec().model_dump(mode="json"),
    }
    rec = ExperimentRecord(
        run_id="r1", created_at=datetime(2026, 1, 1, tzinfo=UTC), hypothesis="h", config=config, git_commit=None,
        seeds=(0,), folds=("A",), metrics={"A/0": {"W": 1.0}}, comparison={"bag": {"A": {"W_all": 5.0, "W_warm": 2.0}}},
    )
    monkeypatch.setattr(registry_mod, "read_records", lambda *a, **k: [rec])
    monkeypatch.setattr(
        runner_mod, "resolve_composition",
        lambda *a, **k: (pd.DataFrame({"site_id": ["s"], "group": ["G1"]}), "abc"),
    )
    monkeypatch.setattr(
        "src.oibc.pipeline._check_input_breaks",
        lambda *a, **k: ["break:wa_ground_press:2026-01-23"],
    )
    monkeypatch.setattr(
        runner_mod, "forecast_eval",
        lambda *a, **k: pd.DataFrame({"group": ["G1"], "ts": [pd.Timestamp("2025-06-01 01:00", tz="Asia/Seoul")], "forecast_kwh": [1.0]}),
    )
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(build_mod, "write_submission", lambda *a, **k: calls.append((a, k)))
    assert main(["submit", "--run-id", "r1", "--sub-id", "s1"]) == 0
    assert calls[0][1]["note"] == "break:wa_ground_press:2026-01-23"


def test_splice_submission_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str, str]] = []
    monkeypatch.setattr(
        "src.oibc.pipeline.splice_submission", lambda base, fb, sid: calls.append((base, fb, sid))
    )
    assert main(["splice-submission", "--base-sub", "b", "--fallback-sub", "f", "--sub-id", "s"]) == 0
    assert calls == [("b", "f", "s")]


def test_splice_env_rejects_unreadable_and_unknown_and_calibrated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json as _json

    import src.oibc.evaluate.registry as registry_mod
    from src.oibc.pipeline import splice_submission
    from src.oibc.train.runner import RunnerError

    root = _splice_env(tmp_path, monkeypatch, base_sha="a" * 64, fb_sha="a" * 64)
    with pytest.raises(RunnerError, match="unreadable"):
        splice_submission("missing", "fb", "spliced")
    monkeypatch.setattr(registry_mod, "read_records", lambda *a, **k: [])
    with pytest.raises(RunnerError, match="unknown run_id"):
        splice_submission("base", "fb", "spliced")
    monkeypatch.undo()
    (tmp_path / "again").mkdir()
    root = _splice_env(tmp_path / "again", monkeypatch, base_sha="a" * 64, fb_sha="a" * 64)
    manifest_path = root / "fb" / "manifest.json"
    data = _json.loads(manifest_path.read_text(encoding="utf-8"))
    data["calibration"] = {"a": 1.0, "b": 0.0}
    manifest_path.write_text(_json.dumps(data), encoding="utf-8")
    with pytest.raises(RunnerError, match="calibrated"):
        splice_submission("base", "fb", "spliced")


def test_splice_submission_rejects_invalid_config_and_train_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.oibc.evaluate.registry as registry_mod
    from src.oibc.pipeline import splice_submission
    from src.oibc.train.runner import RunnerError

    _splice_env(tmp_path, monkeypatch, base_sha="a" * 64, fb_sha="a" * 64)
    records = registry_mod.read_records()
    bad = SimpleNamespace(run_id="fb-run", config={"no": "run"})
    monkeypatch.setattr(registry_mod, "read_records", lambda *a, **k: [records[0], bad])
    with pytest.raises(RunnerError, match="invalid recorded config"):
        splice_submission("base", "fb", "spliced")

    shifted = dict(records[1].config)
    comp = dict(shifted["competition"])
    for key in ("train_end", "eval_start", "eval_end"):
        comp[key] = (pd.Timestamp(comp[key]) + pd.Timedelta(days=1)).isoformat()
    shifted["competition"] = comp
    monkeypatch.setattr(
        registry_mod, "read_records",
        lambda *a, **k: [records[0], SimpleNamespace(run_id="fb-run", config=shifted)],
    )
    with pytest.raises(RunnerError, match="train_end mismatch"):
        splice_submission("base", "fb", "spliced")


def test_splice_submission_rejects_identical_inputs_and_no_break(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.oibc.data.drift as drift_mod
    import src.oibc.evaluate.registry as registry_mod
    from src.oibc.pipeline import splice_submission
    from src.oibc.train.runner import RunnerError

    _splice_env(tmp_path, monkeypatch, base_sha="a" * 64, fb_sha="a" * 64)
    records = registry_mod.read_records()
    fb_config = records[1].config
    same_features = [SimpleNamespace(run_id="base-run", config=fb_config), SimpleNamespace(run_id="fb-run", config=fb_config)]
    monkeypatch.setattr(registry_mod, "read_records", lambda *a, **k: same_features)
    with pytest.raises(RunnerError, match="same raw variables"):
        splice_submission("base", "fb", "spliced")

    monkeypatch.setattr(registry_mod, "read_records", lambda *a, **k: records)
    monkeypatch.setattr(pd, "read_parquet", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(drift_mod, "detect_breaks", lambda fc, p, variables, end, cfg: dict.fromkeys(variables))
    with pytest.raises(RunnerError, match="no break"):
        splice_submission("base", "fb", "spliced")


def _scale_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    import src.oibc.config as config_mod

    root = tmp_path / "submissions"
    root.mkdir()
    paths = config_mod.ArtifactPaths(submissions=root)
    monkeypatch.setattr(config_mod, "ArtifactPaths", lambda *a, **k: paths)
    _write_sub(root, "base", "base-run", "a" * 64, 100.0, ["2026-01-22T12:00:00+09:00", "2026-01-23T01:00:00+09:00"])
    return root


def test_scale_submission_multiplies_every_value_and_records_note(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.oibc.submit.build as build_mod
    from src.oibc.pipeline import scale_submission

    _scale_env(tmp_path, monkeypatch)
    calls: list[tuple[tuple, dict]] = []
    monkeypatch.setattr(build_mod, "write_submission", lambda *a, **k: calls.append((a, k)))
    scale_submission("base", 0.98, "scaled")
    (assignment, forecast, sub_id), kwargs = calls[0]
    assert sub_id == "scaled"
    assert len(forecast) == 4
    np.testing.assert_allclose(forecast["forecast_kwh"].to_numpy(), 98.0, rtol=0, atol=1e-12)
    assert str(forecast["ts"].dt.tz) == "Asia/Seoul"
    assert list(assignment.columns) == ["site_id", "group"]
    assert kwargs["note"] == "level-scale base=base scale=0.98"
    assert kwargs["run_id"] == "base-run"
    assert kwargs["local_metrics"] == {"W_all": 1.0, "W_warm": 1.0}


@pytest.mark.parametrize("scale", [0.0, -0.5, 2.5, float("nan"), float("inf")])
def test_scale_submission_rejects_out_of_range_scale(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scale: float) -> None:
    from src.oibc.pipeline import scale_submission
    from src.oibc.train.runner import RunnerError

    _scale_env(tmp_path, monkeypatch)
    with pytest.raises(RunnerError, match="scale must be finite"):
        scale_submission("base", scale, "scaled")


def test_scale_submission_keeps_base_note_and_rejects_missing_base(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json as _json

    import src.oibc.submit.build as build_mod
    from src.oibc.pipeline import scale_submission
    from src.oibc.train.runner import RunnerError

    root = _scale_env(tmp_path, monkeypatch)
    manifest_path = root / "base" / "manifest.json"
    manifest = _json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["note"] = "splice base=x fallback=y"
    manifest_path.write_text(_json.dumps(manifest), encoding="utf-8")
    calls: list[tuple[tuple, dict]] = []
    monkeypatch.setattr(build_mod, "write_submission", lambda *a, **k: calls.append((a, k)))
    scale_submission("base", 0.98, "scaled")
    assert calls[0][1]["note"] == "level-scale base=base scale=0.98; base_note=splice base=x fallback=y"
    with pytest.raises(RunnerError, match="scale: unreadable submission missing"):
        scale_submission("missing", 0.98, "scaled")


def test_scale_submission_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, float, str, tuple[int, ...] | None]] = []
    monkeypatch.setattr(
        "src.oibc.pipeline.scale_submission",
        lambda base, s, sid, *, months=None: calls.append((base, s, sid, months)),
    )
    assert main(["scale-submission", "--base-sub", "b", "--scale", "0.98", "--sub-id", "s"]) == 0
    assert main(["scale-submission", "--base-sub", "b", "--scale", "0.97", "--sub-id", "w", "--months", "11", "12", "1", "2"]) == 0
    assert calls == [("b", 0.98, "s", None), ("b", 0.97, "w", (11, 12, 1, 2))]


def test_scale_submission_months_follow_right_closed_operating_day(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.oibc.config as config_mod
    import src.oibc.submit.build as build_mod
    from src.oibc.pipeline import scale_submission

    root = tmp_path / "submissions"
    root.mkdir()
    paths = config_mod.ArtifactPaths(submissions=root)
    monkeypatch.setattr(config_mod, "ArtifactPaths", lambda *a, **k: paths)
    # 2026-03-01 00:00은 2월 운영일의 마지막 시각, 01:00부터 3월 운영일이다.
    _write_sub(root, "base", "base-run", "a" * 64, 100.0, ["2026-03-01T00:00:00+09:00", "2026-03-01T01:00:00+09:00"])
    calls: list[tuple[tuple, dict]] = []
    monkeypatch.setattr(build_mod, "write_submission", lambda *a, **k: calls.append((a, k)))
    scale_submission("base", 0.97, "winter", months=(11, 12, 1, 2))
    (_, forecast, _), kwargs = calls[0]
    by_ts = forecast.set_index(forecast["ts"].dt.strftime("%Y-%m-%d %H:%M"))["forecast_kwh"]
    np.testing.assert_allclose(by_ts.loc["2026-03-01 00:00"].to_numpy(), 97.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(by_ts.loc["2026-03-01 01:00"].to_numpy(), 100.0, rtol=0, atol=0)
    assert kwargs["note"] == "level-scale base=base scale=0.97 months=11,12,1,2"


@pytest.mark.parametrize("months", [(), (0,), (13,), (1, 1)])
def test_scale_submission_rejects_invalid_months(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, months: tuple[int, ...]) -> None:
    from src.oibc.pipeline import scale_submission
    from src.oibc.train.runner import RunnerError

    _scale_env(tmp_path, monkeypatch)
    with pytest.raises(RunnerError, match="months must be"):
        scale_submission("base", 0.97, "scaled", months=months)


def test_gfs_hourly_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.oibc.data.external_gfs as gfs_mod

    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(gfs_mod, "build_gfs_hourly", lambda *args: calls.append(args))
    assert main(["gfs-hourly"]) == 0
    assert len(calls) == 1


def test_gfs_fetch_source_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.oibc.data.external_gfs as gfs_mod

    calls: list[dict[str, object]] = []
    monkeypatch.setattr(gfs_mod, "fetch_gfs", lambda *args, **kw: calls.append(kw))
    assert main(["gfs-fetch", "--source", "noaa"]) == 0
    assert main(["gfs-fetch"]) == 0
    assert [c["source"] for c in calls] == ["noaa", "dynamical"]


def test_gfs_compare_dispatch_parses_inits(monkeypatch: pytest.MonkeyPatch) -> None:
    import pandas as pd

    import src.oibc.data.external_gfs as gfs_mod

    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(gfs_mod, "compare_gfs_sources", lambda *args: calls.append(args))
    assert main(["gfs-compare", "--inits", "2025071000"]) == 0
    assert calls[0][3] == [pd.Timestamp("2025-07-10 00:00", tz="UTC")]
    assert main(["gfs-compare", "--sample", "5"]) == 0
    assert calls[1][3] is None and calls[1][2].compare_sample == 5
    with pytest.raises(gfs_mod.ExternalError, match="YYYYMMDDHH"):
        main(["gfs-compare", "--inits", "2025-07-10"])


def _ecmwf_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sites: tuple[str, ...] = ("s1", "s2")):
    import src.oibc.data.external_ecmwf as ec_mod
    import src.oibc.submit.build as build_mod
    import src.oibc.train.runner as runner_mod
    from src.oibc.config import DataPaths

    _scale_env(tmp_path, monkeypatch)
    paths = DataPaths(canonical_dir=tmp_path / "canon", derived_dir=tmp_path / "derived")
    paths.canonical_dir.mkdir()
    paths.derived_dir.mkdir()
    train_ts = pd.Timestamp("2025-03-02 11:00", tz="Asia/Seoul")
    pd.DataFrame({"ts": [train_ts], "smp": [100.0], "k": [3]}).to_parquet(paths.canonical_dir / "smp.parquet")
    pd.DataFrame({"ts": [train_ts], "gfs00_sw_in": [100.0]}).to_parquet(paths.derived_dir / "gfs_hourly.parquet")
    oof = pd.DataFrame({"site_id": list(sites), "ts": [train_ts] * len(sites), "actual_kwh": 5.5, "pred_bag": 5.0})
    monkeypatch.setattr(runner_mod, "load_oof", lambda run_id, fold, **k: oof.copy())
    ops = [pd.Timestamp("2025-03-02"), pd.Timestamp("2026-01-22")]
    # x = (140 - 100) / (100 + 100) = 0.2 → 학습 F=5, A=5.5에서 β=0.5
    monkeypatch.setattr(ec_mod, "ec_blocks", lambda *a, **k: pd.DataFrame({"op": ops, "blk": [1, 1], "ec_in": [140.0, 140.0], "src": ["00z", "18z"]}))
    monkeypatch.setattr(ec_mod, "gfs_blocks", lambda *a, **k: pd.DataFrame({"op": ops, "blk": [1, 1], "g_in": [100.0, 100.0]}))
    calls: list[tuple[tuple, dict]] = []
    monkeypatch.setattr(build_mod, "write_submission", lambda *a, **k: calls.append((a, k)))
    return paths, calls


def test_ecmwf_correct_submission_fits_beta_on_oof_and_scales_block_hours(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.oibc.pipeline import ecmwf_correct_submission

    paths, calls = _ecmwf_env(tmp_path, monkeypatch)
    ecmwf_correct_submission("base", "corr", paths=paths)
    (_, forecast, sub_id), kwargs = calls[0]
    assert sub_id == "corr"
    by_ts = forecast.set_index(forecast["ts"].dt.strftime("%m-%d %H"))["forecast_kwh"]
    # 01-22 12:00은 블록 1(10~12시)이라 ×1.1, 01-23 01:00은 블록 밖이라 그대로
    np.testing.assert_allclose(by_ts.loc["01-22 12"].to_numpy(), 110.0, rtol=0, atol=1e-9)
    np.testing.assert_allclose(by_ts.loc["01-23 01"].to_numpy(), 100.0, rtol=0, atol=1e-12)
    assert kwargs["note"].startswith("ecmwf-correct base=base beta=0.5 fit_rows=4")
    assert kwargs["run_id"] == "base-run"


def test_ecmwf_correct_submission_rejects_oof_site_outside_composition(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.oibc.pipeline import ecmwf_correct_submission
    from src.oibc.train.runner import RunnerError

    paths, _ = _ecmwf_env(tmp_path, monkeypatch, sites=("s1", "s9"))
    with pytest.raises(RunnerError, match="OOF site missing"):
        ecmwf_correct_submission("base", "corr", paths=paths)


def test_ecmwf_correct_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.oibc.pipeline as pipeline_mod

    seen: list[tuple[str, str]] = []
    monkeypatch.setattr(pipeline_mod, "ecmwf_correct_submission", lambda b, s: seen.append((b, s)))
    assert pipeline_mod.main(["ecmwf-correct", "--base-sub", "b", "--sub-id", "s"]) == 0
    assert seen == [("b", "s")]


def test_ecmwf_correct_submission_contract_errors_and_base_note(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json as _json

    import src.oibc.data.external_ecmwf as ec_mod
    import src.oibc.train.runner as runner_mod
    from src.oibc.data.external_gfs import ExternalError
    from src.oibc.pipeline import ecmwf_correct_submission
    from src.oibc.train.runner import RunnerError

    paths, calls = _ecmwf_env(tmp_path, monkeypatch)
    manifest_path = tmp_path / "submissions" / "base" / "manifest.json"
    manifest = _json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["note"] = "level-scale base=x"
    manifest_path.write_text(_json.dumps(manifest), encoding="utf-8")
    ecmwf_correct_submission("base", "corr", paths=paths)
    assert calls[0][1]["note"].endswith("; base_note=level-scale base=x")

    def boom(*a: object, **k: object) -> None:
        raise ExternalError("gfs_blocks: missing columns")

    monkeypatch.setattr(ec_mod, "gfs_blocks", boom)
    with pytest.raises(RunnerError, match="ecmwf: gfs_blocks"):
        ecmwf_correct_submission("base", "corr", paths=paths)
    late = pd.DataFrame({"site_id": ["s1"], "ts": [pd.Timestamp("2025-07-01 11:00", tz="Asia/Seoul")], "actual_kwh": [1.0], "pred_bag": [1.0]})
    monkeypatch.setattr(runner_mod, "load_oof", lambda run_id, fold, **k: late)
    with pytest.raises(RunnerError, match="beyond train_end"):
        ecmwf_correct_submission("base", "corr", paths=paths)
    manifest["run_id"] = None
    manifest_path.write_text(_json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RunnerError, match="no run_id"):
        ecmwf_correct_submission("base", "corr", paths=paths)
