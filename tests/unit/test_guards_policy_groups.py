from __future__ import annotations

import importlib.util
import sys
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any
from zoneinfo import ZoneInfo

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GUARDS_PATH = REPO_ROOT / "tools" / "guards" / "guards.py"
MARKET_HOURS = datetime(2026, 11, 3, 10, 0, tzinfo=ZoneInfo("Asia/Seoul"))

MINIMAL = r"""
forbidden_force_push_flags = ["--force", "-f"]
coarse_risk_pattern = '\b(rm|docker|git\s+push)\b'
forbidden_staged_prefixes = ["scratch/"]
forbidden_name_globs = [".env"]
max_staged_bytes = 5000000
"""

ALWAYS_REQUIRED = (
    "forbidden_force_push_flags",
    "coarse_risk_pattern",
    "forbidden_staged_prefixes",
    "forbidden_name_globs",
    "max_staged_bytes",
)


def _load_guards() -> ModuleType:
    spec = importlib.util.spec_from_file_location("guards_policy_under_test", str(GUARDS_PATH))
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def guards() -> ModuleType:
    return _load_guards()


def _load(guards: ModuleType, tmp_path: Path, text: str) -> Any:
    path = tmp_path / "guards.toml"
    path.write_text(text, encoding="utf-8")
    return guards.load_guard_spec(path)


def _without_key(text: str, key: str) -> str:
    return "\n".join(line for line in text.splitlines() if not line.startswith(f"{key} ")) + "\n"


def test_minimal_config_loads_with_optional_groups_disabled(guards: ModuleType, tmp_path: Path) -> None:
    spec = _load(guards, tmp_path, MINIMAL)

    assert spec.timezone is None
    assert spec.freeze_start is None
    assert spec.freeze_end is None
    assert spec.freeze_command_patterns == ()
    assert spec.order_cli_markers == ()
    assert spec.order_subcommands == ()
    assert spec.allowed_exec_modes == ()
    assert spec.exec_mode_env is None
    assert spec.live_arm_env is None
    assert spec.allowed_staged_names == ()
    assert spec.deny_tools == ()


@pytest.mark.parametrize(
    "command",
    [
        "docker compose up -d",
        "uv run python -m src.cli.main order --symbol 1 --side buy --qty 1",
        "KRX_ALPHA_EXEC_MODE=live x",
        "KRX_ALPHA_EXEC_LIVE_ARMED=true x",
    ],
)
def test_disabled_groups_never_fire(guards: ModuleType, tmp_path: Path, command: str) -> None:
    spec = _load(guards, tmp_path, MINIMAL)

    assert guards.evaluate_bash(command, MARKET_HOURS, tmp_path, spec).allow


@pytest.mark.parametrize(("command", "rule"), [("rm -rf ~/x", "rm"), ("git push --force origin main", "force_push")])
def test_always_on_rules_still_deny(guards: ModuleType, tmp_path: Path, command: str, rule: str) -> None:
    spec = _load(guards, tmp_path, MINIMAL)

    decision = guards.evaluate_bash(command, MARKET_HOURS, tmp_path, spec)

    assert not decision.allow
    assert decision.rule == rule


def test_partial_freeze_group_rejected(guards: ModuleType, tmp_path: Path) -> None:
    text = MINIMAL + r"""
freeze_command_patterns = ['\bdocker\b']
freeze_start = "08:10"
freeze_end = "22:00"
"""
    with pytest.raises(ValueError, match="timezone"):
        _load(guards, tmp_path, text)


def test_empty_freeze_patterns_ignore_other_freeze_fields(guards: ModuleType, tmp_path: Path) -> None:
    spec = _load(guards, tmp_path, MINIMAL + 'freeze_command_patterns = []\ntimezone = "Not/AZone"\n')

    assert spec.timezone is None
    assert spec.freeze_command_patterns == ()


def test_partial_order_group_rejected(guards: ModuleType, tmp_path: Path) -> None:
    text = MINIMAL + r"""
order_cli_markers = ["src.cli.main"]
exec_mode_env = "KRX_ALPHA_EXEC_MODE"
allowed_exec_modes = ["paper", "mock"]
"""
    with pytest.raises(ValueError, match="order_subcommands"):
        _load(guards, tmp_path, text)


def test_live_arm_only_group(guards: ModuleType, tmp_path: Path) -> None:
    spec = _load(guards, tmp_path, MINIMAL + 'live_arm_env = "FOO_ARMED"\n')

    armed = guards.evaluate_bash("FOO_ARMED=true cmd", MARKET_HOURS, tmp_path, spec)
    assert not armed.allow
    assert armed.rule == "live_env"
    assert guards.evaluate_bash("X_MODE=live cmd", MARKET_HOURS, tmp_path, spec).allow


def test_exec_mode_only_live_group(guards: ModuleType, tmp_path: Path) -> None:
    spec = _load(guards, tmp_path, MINIMAL + 'exec_mode_env = "X_MODE"\n')

    mode = guards.evaluate_bash("X_MODE=live cmd", MARKET_HOURS, tmp_path, spec)
    assert not mode.allow
    assert mode.rule == "live_env"
    assert guards.evaluate_bash("FOO_ARMED=true cmd", MARKET_HOURS, tmp_path, spec).allow


@pytest.mark.parametrize("key", ALWAYS_REQUIRED)
def test_missing_always_required_key_rejected(guards: ModuleType, tmp_path: Path, key: str) -> None:
    with pytest.raises(ValueError, match=key):
        _load(guards, tmp_path, _without_key(MINIMAL, key))


def test_freeze_hint_is_appended_and_defaults_to_generic_text(guards: ModuleType, tmp_path: Path) -> None:
    base = (
        'freeze_command_patterns = ["deploy-now"]\ntimezone = "Asia/Seoul"\nfreeze_start = "08:10"\nfreeze_end = "22:00"\n'
        'forbidden_force_push_flags = ["--force"]\ncoarse_risk_pattern = "rm"\n'
        'forbidden_staged_prefixes = ["scratch/"]\nforbidden_name_globs = [".env"]\nmax_staged_bytes = 1000\n'
    )
    plain = tmp_path / "plain.toml"
    plain.write_text(base, encoding="utf-8")
    hinted = tmp_path / "hinted.toml"
    hinted.write_text(base + 'freeze_hint = "Use the nightly job."\n', encoding="utf-8")
    now = datetime(2026, 11, 3, 10, 0, tzinfo=ZoneInfo("Asia/Seoul"))

    generic = guards.evaluate_bash("deploy-now", now, tmp_path, guards.load_guard_spec(plain))
    custom = guards.evaluate_bash("deploy-now", now, tmp_path, guards.load_guard_spec(hinted))

    assert generic.rule == "freeze"
    assert generic.reason.endswith("Run it after the freeze ends.")
    assert custom.rule == "freeze"
    assert custom.reason.endswith("Use the nightly job.")
    assert "krx" not in generic.reason
