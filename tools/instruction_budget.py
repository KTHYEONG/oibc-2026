#!/usr/bin/env python3
"""Measures always-loaded and scoped instruction files against the budget and retention anchors."""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SPEC_PATH = Path(__file__).resolve().with_name("instruction_budget.toml")
_FENCE = "---"
_PATHS_KEY = re.compile(r"^paths\s*:")


@dataclass(frozen=True)
class RetentionAnchor:
    file: str
    phrase: str


@dataclass(frozen=True)
class BudgetSpec:
    agents_file: str
    rules_dir: str
    agents_max_lines: int
    always_loaded_max_bytes: int
    scoped_rules_max_bytes: int
    retention: tuple[RetentionAnchor, ...]


@dataclass(frozen=True)
class RuleMeasure:
    path: str
    size_bytes: int
    scoped: bool


@dataclass(frozen=True)
class InstructionMeasure:
    agents_lines: int
    agents_bytes: int
    rules: tuple[RuleMeasure, ...]
    always_loaded_bytes: int
    scoped_bytes: int


def load_budget_spec(path: Path) -> BudgetSpec:
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    return BudgetSpec(
        agents_file=_string(raw, "agents_file"),
        rules_dir=_string(raw, "rules_dir"),
        agents_max_lines=_positive_int(raw, "agents_max_lines"),
        always_loaded_max_bytes=_positive_int(raw, "always_loaded_max_bytes"),
        scoped_rules_max_bytes=_positive_int(raw, "scoped_rules_max_bytes"),
        retention=_retention(raw),
    )


def measure_instructions(root: Path, spec: BudgetSpec) -> InstructionMeasure:
    agents_path = root / spec.agents_file
    if not agents_path.is_file():
        raise FileNotFoundError(f"agents file not found: {agents_path}")
    agents_bytes = agents_path.stat().st_size
    rules = tuple(_measure_rule(root, path) for path in _rule_files(root / spec.rules_dir))
    return InstructionMeasure(
        agents_lines=_line_count(agents_path.read_bytes()),
        agents_bytes=agents_bytes,
        rules=rules,
        always_loaded_bytes=agents_bytes + sum(rule.size_bytes for rule in rules if not rule.scoped),
        scoped_bytes=sum(rule.size_bytes for rule in rules if rule.scoped),
    )


def check_budget(measure: InstructionMeasure, spec: BudgetSpec) -> list[str]:
    violations: list[str] = []
    if measure.agents_lines > spec.agents_max_lines:
        violations.append(f"agents_lines {measure.agents_lines} exceeds limit {spec.agents_max_lines}")
    if measure.always_loaded_bytes > spec.always_loaded_max_bytes:
        violations.append(
            f"always_loaded_bytes {measure.always_loaded_bytes} exceeds limit {spec.always_loaded_max_bytes}"
        )
    if measure.scoped_bytes > spec.scoped_rules_max_bytes:
        violations.append(f"scoped_bytes {measure.scoped_bytes} exceeds limit {spec.scoped_rules_max_bytes}")
    return violations


def check_retention(root: Path, spec: BudgetSpec) -> list[str]:
    violations: list[str] = []
    for anchor in spec.retention:
        target = root / anchor.file
        if not target.is_file():
            violations.append(f"retention: {anchor.file} is missing (anchor {anchor.phrase!r})")
        elif anchor.phrase not in target.read_text(encoding="utf-8"):
            violations.append(f"retention: {anchor.file} does not contain {anchor.phrase!r}")
    return violations


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check instruction budget and retention anchors.")
    parser.add_argument("--root", type=Path, default=_REPO_ROOT)
    args = parser.parse_args(argv)
    root = Path(args.root)
    spec = load_budget_spec(_SPEC_PATH)
    measure = measure_instructions(root, spec)
    violations = [*check_budget(measure, spec), *check_retention(root, spec)]
    lines = [_render_table(measure, spec), *(f"VIOLATION {v}" for v in violations)]
    lines.append(f"RESULT {'FAIL' if violations else 'OK'} violations={len(violations)}")
    sys.stdout.write("\n".join(lines) + "\n")
    return 1 if violations else 0


def _field(table: Mapping[str, Any], key: str, where: str = "") -> Any:
    if key not in table:
        raise ValueError(f"{where}{key}: missing required key")
    return table[key]


def _string(table: Mapping[str, Any], key: str, where: str = "") -> str:
    value = _field(table, key, where)
    if not isinstance(value, str):
        raise ValueError(f"{where}{key}: expected string, got {type(value).__name__}")
    return value


def _positive_int(table: Mapping[str, Any], key: str) -> int:
    value = _field(table, key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key}: expected integer, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{key}: must be positive, got {value}")
    return value


def _retention(raw: Mapping[str, Any]) -> tuple[RetentionAnchor, ...]:
    tables = _field(raw, "retention")
    if not isinstance(tables, list):
        raise ValueError("retention: expected array of tables")
    anchors: list[RetentionAnchor] = []
    for index, table in enumerate(tables):
        where = f"retention[{index}]."
        if not isinstance(table, dict):
            raise ValueError(f"retention[{index}]: expected table")
        phrase = _string(table, "phrase", where)
        if not phrase:
            raise ValueError(f"{where}phrase: must be non-empty")
        anchors.append(RetentionAnchor(file=_string(table, "file", where), phrase=phrase))
    return tuple(anchors)


def _rule_files(rules_dir: Path) -> list[Path]:
    if not rules_dir.is_dir():
        return []
    return sorted((path for path in rules_dir.glob("*.md") if path.is_file()), key=lambda path: path.name)


def _measure_rule(root: Path, path: Path) -> RuleMeasure:
    return RuleMeasure(
        path=path.relative_to(root).as_posix(),
        size_bytes=path.stat().st_size,
        scoped=_is_scoped(path.read_text(encoding="utf-8")),
    )


def _is_scoped(text: str) -> bool:
    lines = text.splitlines()
    if not lines or lines[0] != _FENCE:
        return False
    try:
        closing = lines.index(_FENCE, 1)
    except ValueError:
        return False
    return any(_PATHS_KEY.match(line) for line in lines[1:closing])


def _line_count(data: bytes) -> int:
    if not data:
        return 0
    return data.count(b"\n") + (0 if data.endswith(b"\n") else 1)


def _render_table(measure: InstructionMeasure, spec: BudgetSpec) -> str:
    rows = [
        f"{'file':<40} {'bytes':>8}  scope",
        f"{spec.agents_file:<40} {measure.agents_bytes:>8}  always",
    ]
    rows.extend(
        f"{rule.path:<40} {rule.size_bytes:>8}  {'scoped' if rule.scoped else 'always'}" for rule in measure.rules
    )
    rows.extend(
        [
            f"agents_lines {measure.agents_lines} / {spec.agents_max_lines}",
            f"always_loaded_bytes {measure.always_loaded_bytes} / {spec.always_loaded_max_bytes}",
            f"scoped_bytes {measure.scoped_bytes} / {spec.scoped_rules_max_bytes}",
        ]
    )
    return "\n".join(rows)


if __name__ == "__main__":
    raise SystemExit(main())
