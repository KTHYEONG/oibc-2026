from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_NAME = "instruction_budget_under_test"

VALID_TOML = """\
agents_file = "AGENTS.md"
rules_dir = ".agents/rules"
agents_max_lines = 100
always_loaded_max_bytes = 6000
scoped_rules_max_bytes = 10000

[[retention]]
file = "AGENTS.md"
phrase = "needle"
"""


def _load_budget() -> ModuleType:
    spec = importlib.util.spec_from_file_location(MODULE_NAME, str(REPO_ROOT / "tools" / "instruction_budget.py"))
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def _spec(budget: ModuleType, **overrides: Any) -> Any:
    fields: dict[str, Any] = {
        "agents_file": "AGENTS.md",
        "rules_dir": "rules",
        "agents_max_lines": 100,
        "always_loaded_max_bytes": 10_000,
        "scoped_rules_max_bytes": 10_000,
        "retention": (),
    }
    fields.update(overrides)
    return budget.BudgetSpec(**fields)


def _write(root: Path, rel: str, text: str) -> None:
    target = root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def _single_rule(tmp_path: Path, text: str) -> Any:
    budget = _load_budget()
    _write(tmp_path, "AGENTS.md", "a\n")
    _write(tmp_path, "rules/rule.md", text)
    measure = budget.measure_instructions(tmp_path, _spec(budget))
    (rule,) = measure.rules
    return rule


def test_scoped_rule_detected(tmp_path: Path) -> None:
    budget = _load_budget()
    rule_text = '---\npaths:\n  - "src/**"\n---\nbody\n'
    _write(tmp_path, "AGENTS.md", "a\n")
    _write(tmp_path, "rules/scoped.md", rule_text)

    measure = budget.measure_instructions(tmp_path, _spec(budget))

    (rule,) = measure.rules
    assert rule.scoped is True
    assert measure.scoped_bytes == len(rule_text.encode())
    assert measure.always_loaded_bytes == measure.agents_bytes


def test_rule_without_frontmatter_is_always_loaded(tmp_path: Path) -> None:
    budget = _load_budget()
    rule_text = "# plain rule\n"
    _write(tmp_path, "AGENTS.md", "a\n")
    _write(tmp_path, "rules/plain.md", rule_text)

    measure = budget.measure_instructions(tmp_path, _spec(budget))

    (rule,) = measure.rules
    assert rule.scoped is False
    assert measure.scoped_bytes == 0
    assert measure.always_loaded_bytes == measure.agents_bytes + len(rule_text.encode())


def test_unterminated_frontmatter_is_not_scoped(tmp_path: Path) -> None:
    assert _single_rule(tmp_path, "---\npaths: [a]\n").scoped is False


def test_frontmatter_without_paths_is_not_scoped(tmp_path: Path) -> None:
    assert _single_rule(tmp_path, "---\ndescription: x\n---\nbody\n").scoped is False


def test_indented_paths_key_is_not_scoped(tmp_path: Path) -> None:
    assert _single_rule(tmp_path, "---\nmeta:\n  paths: [a]\n---\nbody\n").scoped is False


def test_budget_boundary_is_inclusive(tmp_path: Path) -> None:
    budget = _load_budget()
    _write(tmp_path, "AGENTS.md", "a\n")
    _write(tmp_path, "rules/plain.md", "x\n")
    measure = budget.measure_instructions(tmp_path, _spec(budget))
    limit = measure.always_loaded_bytes

    assert budget.check_budget(measure, _spec(budget, always_loaded_max_bytes=limit)) == []
    violations = budget.check_budget(measure, _spec(budget, always_loaded_max_bytes=limit - 1))
    assert len(violations) == 1


def test_each_budget_breach_reported_once(tmp_path: Path) -> None:
    budget = _load_budget()
    _write(tmp_path, "AGENTS.md", "a\nb\n")
    _write(tmp_path, "rules/plain.md", "x" * 8)
    _write(tmp_path, "rules/scoped.md", "---\npaths: [x]\n---\n" + "y" * 12)
    spec = _spec(budget, agents_max_lines=1, always_loaded_max_bytes=1, scoped_rules_max_bytes=1)

    measure = budget.measure_instructions(tmp_path, spec)
    violations = budget.check_budget(measure, spec)

    assert [violation.split()[0] for violation in violations] == [
        "agents_lines",
        "always_loaded_bytes",
        "scoped_bytes",
    ]


def test_missing_agents_file_fails_closed(tmp_path: Path) -> None:
    budget = _load_budget()
    with pytest.raises(FileNotFoundError):
        budget.measure_instructions(tmp_path, _spec(budget))


def test_missing_rules_dir_is_empty(tmp_path: Path) -> None:
    budget = _load_budget()
    _write(tmp_path, "AGENTS.md", "a\n")
    assert budget.measure_instructions(tmp_path, _spec(budget)).rules == ()


def test_line_count_handles_missing_trailing_newline(tmp_path: Path) -> None:
    budget = _load_budget()
    for name, text in [("bare", "a\nb"), ("final", "a\nb\n")]:
        _write(tmp_path / name, "AGENTS.md", text)
        assert budget.measure_instructions(tmp_path / name, _spec(budget)).agents_lines == 2


def test_retention_missing_phrase_reported(tmp_path: Path) -> None:
    budget = _load_budget()
    _write(tmp_path, "AGENTS.md", "Needle here\n")
    spec = _spec(
        budget,
        retention=(
            budget.RetentionAnchor(file="AGENTS.md", phrase="Needle"),
            budget.RetentionAnchor(file="AGENTS.md", phrase="needle"),
        ),
    )

    violations = budget.check_retention(tmp_path, spec)

    assert len(violations) == 1
    assert "AGENTS.md" in violations[0]
    assert "needle" in violations[0]


def test_retention_missing_file_reported(tmp_path: Path) -> None:
    budget = _load_budget()
    spec = _spec(budget, retention=(budget.RetentionAnchor(file="missing.md", phrase="x"),))

    violations = budget.check_retention(tmp_path, spec)

    assert len(violations) == 1
    assert "missing.md" in violations[0]


@pytest.mark.parametrize(
    ("old", "new", "key"),
    [
        ("agents_max_lines = 100\n", "", "agents_max_lines"),
        ("agents_max_lines = 100", "agents_max_lines = 0", "agents_max_lines"),
        ("agents_max_lines = 100", 'agents_max_lines = "100"', "agents_max_lines"),
        ('phrase = "needle"', 'phrase = ""', "phrase"),
    ],
)
def test_spec_rejects_bad_toml(tmp_path: Path, old: str, new: str, key: str) -> None:
    budget = _load_budget()
    assert old in VALID_TOML
    good = tmp_path / "good.toml"
    good.write_text(VALID_TOML, encoding="utf-8")
    assert budget.load_budget_spec(good).agents_max_lines == 100

    bad = tmp_path / "bad.toml"
    bad.write_text(VALID_TOML.replace(old, new), encoding="utf-8")
    with pytest.raises(ValueError, match=key):
        budget.load_budget_spec(bad)


def test_main_exit_codes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    budget = _load_budget()
    synthetic = _spec(budget, agents_max_lines=2)
    monkeypatch.setattr(budget, "load_budget_spec", lambda _path: synthetic)
    clean = tmp_path / "clean"
    dirty = tmp_path / "dirty"
    _write(clean, "AGENTS.md", "a\nb\n")
    _write(dirty, "AGENTS.md", "a\nb\nc\n")

    for root, expected in [(clean, 0), (dirty, 1)]:
        assert budget.main(["--root", str(root)]) == expected
        first = capsys.readouterr().out
        assert budget.main(["--root", str(root)]) == expected
        assert capsys.readouterr().out == first


def _repo_spec(budget: ModuleType) -> Any:
    return budget.load_budget_spec(REPO_ROOT / "tools" / "instruction_budget.toml")


def test_repo_instructions_within_budget() -> None:
    budget = _load_budget()
    spec = _repo_spec(budget)

    assert budget.check_budget(budget.measure_instructions(REPO_ROOT, spec), spec) == []


def test_repo_hard_boundaries_retained() -> None:
    budget = _load_budget()

    assert budget.check_retention(REPO_ROOT, _repo_spec(budget)) == []


def test_every_rule_file_is_path_scoped() -> None:
    budget = _load_budget()
    measure = budget.measure_instructions(REPO_ROOT, _repo_spec(budget))

    assert measure.rules
    assert [rule.path for rule in measure.rules if not rule.scoped] == []
