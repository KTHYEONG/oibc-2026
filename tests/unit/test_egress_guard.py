from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GUARDS_PATH = REPO_ROOT / "tools" / "guards" / "guards.py"
SETTINGS_PATH = REPO_ROOT / ".claude" / "settings.json"


def _pretool(tool_name: str) -> subprocess.CompletedProcess[str]:
    payload = json.dumps({"tool_name": tool_name, "tool_input": {}})
    env = {**os.environ, "CLAUDE_PROJECT_DIR": str(REPO_ROOT)}
    return subprocess.run(  # noqa: S603 - fixed argv, project-owned script
        [sys.executable, "-I", str(GUARDS_PATH), "pretool"],
        input=payload,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


@pytest.mark.parametrize(
    "tool",
    [
        "WebFetch",
        "WebSearch",
        "Artifact",
        "ArtifactData",
        "ArtifactComments",
        "RemoteTrigger",
        "mcp__claude_ai_Gmail__authenticate",
        "mcp__claude_ai_Claude_Docs__create",
        "mcp__claude_ai_Google_Drive__authenticate",
    ],
)
def test_egress_tools_are_denied(tool: str) -> None:
    result = _pretool(tool)

    assert result.returncode == 2
    assert "must not leave this machine" in result.stderr


@pytest.mark.parametrize("tool", ["Read", "Edit", "mcp__serena__find_symbol", "mcp__duckdb__query"])
def test_local_tools_are_not_denied_by_egress_rule(tool: str) -> None:
    assert _pretool(tool).returncode == 0


def test_settings_hooks_cover_every_denied_tool() -> None:
    settings = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    matchers = [group["matcher"] for group in settings["hooks"]["PreToolUse"]]

    assert any("WebFetch" in m and "WebSearch" in m and "Artifact" in m and "mcp__claude_ai_" in m for m in matchers)
