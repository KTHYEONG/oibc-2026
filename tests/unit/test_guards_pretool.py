from __future__ import annotations

import dataclasses
import importlib.util
import io
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any
from zoneinfo import ZoneInfo

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GUARDS_PATH = REPO_ROOT / "tools" / "guards" / "guards.py"
SPEC_PATH = REPO_ROOT / "tests" / "unit" / "guards_test_config.toml"
MARKET_HOURS = datetime(2026, 11, 3, 10, 0, tzinfo=ZoneInfo("Asia/Seoul"))


def _load_guards() -> ModuleType:
    spec = importlib.util.spec_from_file_location("guards_pretool_under_test", str(GUARDS_PATH))
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _bash_payload(command: str) -> str:
    return json.dumps({"tool_name": "Bash", "tool_input": {"command": command}})


@pytest.fixture(scope="module")
def guards() -> ModuleType:
    return _load_guards()


@pytest.fixture(scope="module")
def spec(guards: ModuleType) -> Any:
    return guards.load_guard_spec(SPEC_PATH)


def _run(guards: ModuleType, spec: Any, stdin_text: str, project: Path) -> tuple[int, str]:
    stderr = io.StringIO()
    code = guards.main_pretool(io.StringIO(stdin_text), stderr, MARKET_HOURS, project, spec)
    return code, stderr.getvalue()


def test_non_bash_tool_ignored(guards: ModuleType, spec: Any, tmp_path: Path) -> None:
    payload = json.dumps({"tool_name": "Edit", "tool_input": {"file_path": "rm -rf ~/x"}})
    assert _run(guards, spec, payload, tmp_path) == (0, "")


def test_deny_exits_2_with_reason(
    guards: ModuleType, spec: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, stderr = _run(guards, spec, _bash_payload("rm -rf ~/x"), tmp_path)
    assert code == 2
    assert "recursive rm" in stderr
    assert capsys.readouterr().out == ""


def test_allow_exits_0_silently(guards: ModuleType, spec: Any, tmp_path: Path) -> None:
    assert _run(guards, spec, _bash_payload("echo hi"), tmp_path) == (0, "")


def test_invalid_json_fails_closed(guards: ModuleType, spec: Any, tmp_path: Path) -> None:
    code, _ = _run(guards, spec, "not json", tmp_path)
    assert code == 2


def test_non_object_payload_fails_closed(guards: ModuleType, spec: Any, tmp_path: Path) -> None:
    code, _ = _run(guards, spec, "[]", tmp_path)
    assert code == 2


def test_missing_command_fails_closed(guards: ModuleType, spec: Any, tmp_path: Path) -> None:
    code, _ = _run(guards, spec, json.dumps({"tool_name": "Bash", "tool_input": {}}), tmp_path)
    assert code == 2


def test_internal_error_on_risky_command_blocks(
    guards: ModuleType, spec: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(guards, "evaluate_bash", boom)
    assert _run(guards, spec, _bash_payload("rm -rf x"), tmp_path) == (
        2,
        "guard internal error (fail-closed): RuntimeError\n",
    )
    assert _run(guards, spec, _bash_payload("echo hi"), tmp_path) == (0, "")


def test_oversized_risky_command_fails_closed(guards: ModuleType, spec: Any, tmp_path: Path) -> None:
    code, _ = _run(guards, spec, _bash_payload("rm -rf " + "a" * 100_000), tmp_path)
    assert code == 2


def test_oversized_benign_command_allowed(guards: ModuleType, spec: Any, tmp_path: Path) -> None:
    code, _ = _run(guards, spec, _bash_payload("echo " + "a" * 100_000), tmp_path)
    assert code == 0


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (_bash_payload("echo hi"), 0),
        (_bash_payload("rm -rf ~/x"), 2),
        ("not json", 2),
    ],
)
def test_subprocess_isolated_mode_works(payload: str, expected: int, tmp_path: Path) -> None:
    completed = subprocess.run(  # noqa: S603
        [sys.executable, "-I", str(GUARDS_PATH), "pretool"],
        input=payload,
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env={"CLAUDE_PROJECT_DIR": str(tmp_path)},
        timeout=60,
        check=False,
    )
    assert completed.returncode == expected
    assert completed.stdout == ""


def test_main_rejects_unknown_subcommand(guards: ModuleType, capsys: pytest.CaptureFixture[str]) -> None:
    assert guards.main([]) == 2
    assert guards.main(["bogus"]) == 2
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("tool", ["WebFetch", "WebSearch"])
def test_deny_tool_blocks_non_bash_tool(
    guards: ModuleType, spec: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str], tool: str
) -> None:
    payload = json.dumps({"tool_name": tool, "tool_input": {"query": "krx"}})
    code, stderr = _run(guards, spec, payload, tmp_path)
    assert code == 2
    assert stderr == f"{spec.deny_tools[0].reason}\n"
    assert capsys.readouterr().out == ""


def test_unlisted_tool_allowed(guards: ModuleType, spec: Any, tmp_path: Path) -> None:
    payload = json.dumps({"tool_name": "Read", "tool_input": {"file_path": "AGENTS.md"}})
    assert _run(guards, spec, payload, tmp_path) == (0, "")


def test_deny_tools_not_applied_to_bash(guards: ModuleType, spec: Any, tmp_path: Path) -> None:
    catch_all = dataclasses.replace(spec, deny_tools=(guards.DenyTool(tool=".*", reason="all tools denied"),))
    assert _run(guards, catch_all, _bash_payload("echo hi"), tmp_path) == (0, "")
    code, stderr = _run(guards, catch_all, _bash_payload("rm -rf ~/x"), tmp_path)
    assert code == 2
    assert "recursive rm" in stderr


def test_no_deny_tools_means_no_denials(guards: ModuleType, tmp_path: Path) -> None:
    path = tmp_path / "guards.toml"
    path.write_text(_fixture_text_before_deny_tools(), encoding="utf-8")
    config = guards.load_guard_spec(path)
    assert config.deny_tools == ()
    payload = json.dumps({"tool_name": "WebFetch", "tool_input": {}})
    assert _run(guards, config, payload, tmp_path) == (0, "")


@pytest.mark.parametrize(
    ("entry", "match"),
    [
        ('tool = "("\nreason = "x"\n', "invalid regex"),
        ('tool = "WebFetch"\n', "missing key 'reason'"),
        ('reason = "x"\n', "missing key 'tool'"),
        ('tool = "WebFetch"\nreason = "' + "x" * 301 + '"\n', "exceeds 300"),
    ],
)
def test_bad_deny_tool_entry_rejected(guards: ModuleType, tmp_path: Path, entry: str, match: str) -> None:
    path = tmp_path / "guards.toml"
    path.write_text(_fixture_text_before_deny_tools() + "[[deny_tools]]\n" + entry, encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        guards.load_guard_spec(path)


def _fixture_text_before_deny_tools() -> str:
    return SPEC_PATH.read_text(encoding="utf-8").split("[[deny_tools]]", maxsplit=1)[0]
