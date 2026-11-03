from __future__ import annotations

import importlib.util
import sys
from datetime import UTC, datetime, time
from pathlib import Path
from types import ModuleType
from zoneinfo import ZoneInfo

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GUARDS_PATH = REPO_ROOT / "tools" / "guards" / "guards.py"
SPEC_PATH = REPO_ROOT / "tests" / "unit" / "guards_test_config.toml"
KST = ZoneInfo("Asia/Seoul")
MARKET_HOURS = datetime(2026, 11, 3, 10, 0, tzinfo=KST)


def _load_guards() -> ModuleType:
    spec = importlib.util.spec_from_file_location("guards_under_test", str(GUARDS_PATH))
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def guards() -> ModuleType:
    return _load_guards()


@pytest.fixture(scope="module")
def spec(guards: ModuleType):
    return guards.load_guard_spec(SPEC_PATH)


@pytest.fixture
def decide(guards: ModuleType, spec, tmp_path: Path):
    def _decide(command: str, now: datetime = MARKET_HOURS):
        return guards.evaluate_bash(command, now, tmp_path, spec)

    return _decide


def _assert_denied(decision, rule: str) -> None:
    assert not decision.allow
    assert decision.rule == rule
    assert 0 < len(decision.reason) <= 300


@pytest.mark.parametrize("command", ["rm -rf ~/x", "rm -rf $HOME/x", 'rm -rf "$HOME"'])
def test_rm_home_denied(decide, command: str) -> None:
    _assert_denied(decide(command), "rm")


@pytest.mark.parametrize("command", ["rm -rf /", "rm -rf .", "rm -rf ..", "rm -rf .git", "rm -rf *"])
def test_rm_dangerous_targets_denied(decide, command: str) -> None:
    _assert_denied(decide(command), "rm")


@pytest.mark.parametrize(
    "command",
    ["rm -rf ../other", "rm -rf /etc/x", "rm -rf .git/objects", "rm -rf scratch/..", "rm -rf ./"],
)
def test_rm_outside_or_project_root_denied(decide, command: str) -> None:
    _assert_denied(decide(command), "rm")


def test_rm_inside_scratch_allowed(decide) -> None:
    assert decide("rm -rf scratch/_trash/20261001-x").allow


def test_rm_non_recursive_allowed(decide) -> None:
    assert decide("rm scratch/a.py").allow


def test_rm_in_compound_command_denied(decide) -> None:
    _assert_denied(decide("echo ok && rm -rf ~/x"), "rm")


def test_rm_quoted_operator_does_not_hide_later_targets(decide) -> None:
    _assert_denied(decide('rm -rf scratch/a "|" ~/x'), "rm")


def test_rm_redirect_to_devnull_allowed(decide) -> None:
    assert decide("rm -rf scratch/x 2>/dev/null").allow


@pytest.mark.parametrize("command", ["echo $(rm -rf ~/x)", "echo `rm -rf ~/x`"])
def test_rm_inside_substitution_denied(decide, command: str) -> None:
    assert not decide(command).allow


def test_rm_with_command_substitution_target_denied(decide) -> None:
    _assert_denied(decide("rm -rf $(pwd)"), "rm")


@pytest.mark.parametrize(
    "command",
    [
        "git push --force origin main",
        "git push -f",
        "git push origin +main",
        "git -C /tmp/repo push -fu origin main",
        "git push --force-with-lease origin main",
    ],
)
def test_force_push_denied(decide, command: str) -> None:
    _assert_denied(decide(command), "force_push")


def test_plain_push_allowed(decide) -> None:
    assert decide("git push origin main").allow


@pytest.mark.parametrize(
    "command",
    [
        "uv run python -m src.cli.main order --symbol 005930 --side buy --qty 1",
        "uv run python -m src.cli.main order-cancel --order-id 1",
    ],
)
def test_order_requires_explicit_mock_or_paper(decide, command: str) -> None:
    _assert_denied(decide(command), "order")


@pytest.mark.parametrize("mode", ["mock", "paper"])
def test_order_with_explicit_mode_allowed(decide, mode: str) -> None:
    command = f"KRX_ALPHA_EXEC_MODE={mode} uv run python -m src.cli.main order --symbol 005930 --side buy --qty 1"
    assert decide(command).allow


def test_order_with_live_mode_denied_by_live_env(decide) -> None:
    command = "KRX_ALPHA_EXEC_MODE=live uv run python -m src.cli.main order --symbol 005930"
    _assert_denied(decide(command), "live_env")


def test_order_status_is_read_only(decide) -> None:
    assert decide("uv run python -m src.cli.main order-status").allow


@pytest.mark.parametrize(
    "command",
    [
        "KRX_ALPHA_EXEC_LIVE_ARMED=true ls",
        "KRX_ALPHA_EXEC_LIVE_ARMED=TRUE ls",
        "KRX_ALPHA_EXEC_LIVE_ARMED=1 uv run python -m src.cli.main order-status",
        "echo KRX_ALPHA_EXEC_LIVE_ARMED=yes",
        "KRX_ALPHA_EXEC_MODE=LIVE ls",
    ],
)
def test_live_arm_always_denied(decide, command: str) -> None:
    _assert_denied(decide(command), "live_env")


@pytest.mark.parametrize(
    ("hour", "minute", "frozen"),
    [(10, 0, True), (8, 9, False), (8, 10, True), (21, 59, True), (22, 0, False)],
)
def test_freeze_denies_inside_window(decide, hour: int, minute: int, frozen: bool) -> None:
    now = datetime(2026, 11, 3, hour, minute, tzinfo=KST)
    decision = decide("docker compose up -d", now)
    assert decision.allow is not frozen
    if frozen:
        assert decision.rule == "freeze"


def test_freeze_converts_timezone(decide) -> None:
    now = datetime(2026, 11, 3, 1, 0, tzinfo=UTC)
    _assert_denied(decide("docker compose up -d", now), "freeze")


def test_freeze_denies_substituted_compose(decide) -> None:
    _assert_denied(decide("echo $(docker compose up -d)"), "freeze")


@pytest.mark.parametrize(
    "command",
    [
        "docker compose ps",
        "docker compose logs krx-collector",
        "docker compose logs --since start",
        "docker compose config",
    ],
)
def test_freeze_ignores_read_only_compose(decide, command: str) -> None:
    assert decide(command).allow


@pytest.mark.parametrize("command", ["ls", "cat AGENTS.md", "git status", "git log -3"])
def test_read_only_commands_never_denied(decide, command: str) -> None:
    assert decide(command).allow


@pytest.mark.parametrize("command", ["rm -rf 'unterminated ~/x", "KRX_ALPHA_EXEC_LIVE_ARMED=true echo 'x"])
def test_unparseable_command_uses_regex_fallback(decide, command: str) -> None:
    assert not decide(command).allow


def test_unparseable_force_push_uses_regex_fallback(decide) -> None:
    _assert_denied(decide("git push --force origin 'main"), "force_push")


def test_empty_command_allowed(decide) -> None:
    assert decide("   ").allow


def test_naive_datetime_rejected(guards: ModuleType, spec, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        guards.evaluate_bash("ls", datetime(2026, 11, 3, 10, 0), tmp_path, spec)


def test_oversized_command_raises_for_internal_path(guards: ModuleType, spec, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="size limit"):
        guards.evaluate_bash("rm -rf " + "a" * 100_000, MARKET_HOURS, tmp_path, spec)


def test_evaluate_is_deterministic(decide) -> None:
    command = "docker compose up -d && rm -rf ~/x"
    assert decide(command) == decide(command)


def test_load_guard_spec_reads_policy(guards: ModuleType, spec) -> None:
    assert spec.timezone == "Asia/Seoul"
    assert spec.freeze_start == time(8, 10)
    assert spec.freeze_end == time(22, 0)
    assert spec.allowed_exec_modes == ("paper", "mock")


def test_load_guard_spec_rejects_missing_key(guards: ModuleType, tmp_path: Path) -> None:
    text = SPEC_PATH.read_text(encoding="utf-8")
    broken = tmp_path / "guards.toml"
    broken.write_text(
        "\n".join(line for line in text.splitlines() if not line.startswith("timezone")), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="missing key 'timezone'"):
        guards.load_guard_spec(broken)
