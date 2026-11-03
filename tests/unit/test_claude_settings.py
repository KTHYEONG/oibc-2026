from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SETTINGS_PATH = REPO_ROOT / ".claude" / "settings.json"
ENV_READ_RULES = ("Read(./.env)", "Read(./.env.*)", "Read(./.envrc)")


def _settings() -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    return loaded


def _bash_hook_handler(settings: dict[str, Any]) -> dict[str, Any]:
    groups = settings["hooks"]["PreToolUse"]
    group = next(item for item in groups if item.get("matcher") == "Bash")
    handler: dict[str, Any] = group["hooks"][0]
    return handler


def test_settings_hook_points_to_existing_script() -> None:
    handler = _bash_hook_handler(_settings())
    assert handler["type"] == "command"
    command = handler["command"].replace("$CLAUDE_PROJECT_DIR", str(REPO_ROOT))
    script = next(token for token in shlex.split(command) if token.endswith("guards.py"))
    assert Path(script).is_file()


def test_settings_hook_fails_closed() -> None:
    assert _bash_hook_handler(_settings())["onFailure"] == "block"


def test_settings_denies_env_reads() -> None:
    deny_rules = _settings()["permissions"]["deny"]
    for rule in ENV_READ_RULES:
        assert rule in deny_rules
