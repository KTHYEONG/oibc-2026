from datetime import UTC, datetime
from pathlib import Path

import pytest

from src.oibc.evaluate.registry import (
    ExperimentRecord,
    RegistryError,
    append_record,
    count_comparisons,
    read_records,
)


def _record(run_id: str, **over: object) -> ExperimentRecord:
    fields: dict[str, object] = {
        "run_id": run_id,
        "created_at": datetime(2026, 1, 1, tzinfo=UTC),
        "hypothesis": "lighter model keeps skill",
        "config": {"depth": 3},
        "git_commit": None,
        "seeds": (0, 1),
        "folds": ("A", "B"),
        "metrics": {
            "A/0": {"W": 10.0, "revenue": 100.0, "S": 90.0},
            "A/1": {"W": 11.0, "revenue": 100.0, "S": 89.0},
            "B/0": {"W": 12.0, "revenue": 100.0, "S": 88.0},
            "B/1": {"W": 13.0, "revenue": 100.0, "S": 87.0},
        },
    }
    fields.update(over)
    return ExperimentRecord(**fields)  # type: ignore[arg-type]


def test_append_then_read_in_order(tmp_path: Path) -> None:
    path = tmp_path / "registry.jsonl"
    first, second = _record("run-1"), _record("run-2")
    append_record(first, path)
    append_record(second, path)
    records = read_records(path)
    assert [r.run_id for r in records] == ["run-1", "run-2"]
    assert records[0] == first
    assert records[1] == second


def test_missing_file_reads_empty(tmp_path: Path) -> None:
    assert read_records(tmp_path / "none.jsonl") == []


def test_duplicate_run_id_rejected(tmp_path: Path) -> None:
    path = tmp_path / "registry.jsonl"
    append_record(_record("run-1"), path)
    before = path.read_bytes()
    with pytest.raises(RegistryError):
        append_record(_record("run-1"), path)
    assert path.read_bytes() == before


def test_existing_bytes_preserved(tmp_path: Path) -> None:
    path = tmp_path / "sub" / "registry.jsonl"
    append_record(_record("run-1"), path)
    before = path.read_bytes()
    append_record(_record("run-2"), path)
    after = path.read_bytes()
    assert after.startswith(before)
    assert after.decode("utf-8").count("\n") == before.decode("utf-8").count("\n") + 1


def test_corrupt_file_rejected(tmp_path: Path) -> None:
    path = tmp_path / "registry.jsonl"
    path.write_text('{"run_id": "broken"\n', encoding="utf-8")
    before = path.read_bytes()
    with pytest.raises(RegistryError):
        append_record(_record("run-1"), path)
    with pytest.raises(RegistryError):
        read_records(path)
    assert path.read_bytes() == before


def test_blank_line_rejected(tmp_path: Path) -> None:
    path = tmp_path / "registry.jsonl"
    append_record(_record("run-1"), path)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write("\n")
    with pytest.raises(RegistryError):
        read_records(path)


def test_missing_trailing_newline_rejected(tmp_path: Path) -> None:
    path = tmp_path / "registry.jsonl"
    line = _record("run-1").model_dump_json(ensure_ascii=False)
    path.write_text(line, encoding="utf-8")
    with pytest.raises(RegistryError):
        read_records(path)


def test_hypothesis_required() -> None:
    with pytest.raises(ValueError, match="hypothesis"):
        _record("run-1", hypothesis="")
    with pytest.raises(ValueError, match="hypothesis"):
        _record("run-1", hypothesis="   ")


def test_bad_run_id_rejected() -> None:
    with pytest.raises(ValueError, match="run_id"):
        _record("bad id!")


def test_naive_created_at_rejected() -> None:
    with pytest.raises(ValueError, match="tz-aware"):
        _record("run-1", created_at=datetime(2026, 1, 1))


def test_metrics_key_consistency() -> None:
    with pytest.raises(ValueError, match="metrics key"):
        _record("run-1", metrics={"C/0": {"W": 1.0}})
    with pytest.raises(ValueError, match="metrics key"):
        _record("run-1", metrics={"A/9": {"W": 1.0}})
    with pytest.raises(ValueError, match="metrics key"):
        _record("run-1", metrics={"A": {"W": 1.0}})
    with pytest.raises(ValueError, match="metrics key"):
        _record("run-1", metrics={"A/x": {"W": 1.0}})
    with pytest.raises(ValueError, match="finite"):
        _record("run-1", metrics={"A/0": {"W": float("inf")}})


def test_count_comparisons() -> None:
    records = [
        _record("r1", comparison={"decision": "adopt"}, baseline_run_id="X"),
        _record("r2", comparison={"decision": "reject"}, baseline_run_id="X"),
        _record("r3", comparison={"decision": "adopt"}, baseline_run_id="Y"),
        _record("r4"),
    ]
    assert count_comparisons(records) == 3
    assert count_comparisons(records, "X") == 2
    assert count_comparisons(records, "Y") == 1


@pytest.mark.parametrize(
    "over",
    [
        {"config": {"lr": float("nan")}},
        {"comparison": {"folds": [{"rel_delta": float("inf")}]}},
        {"config": {"grid": (1.0, float("-inf"))}},
    ],
)
def test_nonfinite_payload_rejected(over: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="finite"):
        _record("run-1", **over)
