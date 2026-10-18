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
