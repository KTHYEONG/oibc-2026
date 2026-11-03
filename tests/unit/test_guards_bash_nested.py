from __future__ import annotations

import importlib.util
import sys
from datetime import datetime
from pathlib import Path
from types import ModuleType
from zoneinfo import ZoneInfo

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GUARDS_PATH = REPO_ROOT / "tools" / "guards" / "guards.py"
SPEC_PATH = REPO_ROOT / "tests" / "unit" / "guards_test_config.toml"
KST = ZoneInfo("Asia/Seoul")
MARKET_HOURS = datetime(2026, 11, 3, 10, 0, tzinfo=KST)
AFTER_FREEZE = datetime(2026, 11, 3, 22, 30, tzinfo=KST)


@pytest.fixture(scope="module")
def guards() -> ModuleType:
    spec = importlib.util.spec_from_file_location("guards_nested_under_test", str(GUARDS_PATH))
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def decide(guards: ModuleType, tmp_path: Path):
    spec = guards.load_guard_spec(SPEC_PATH)

    def _decide(command: str, now: datetime = MARKET_HOURS):
        return guards.evaluate_bash(command, now, tmp_path, spec)

    return _decide


@pytest.mark.parametrize(
    "command",
    [
        "bash -c 'rm -rf ~/x'",
        'sh -c "rm -rf /home/someone"',
        "bash -lc 'echo ok; rm -rf ~/x'",
        "eval 'rm -rf ~/x'",
        "ssh host 'rm -rf ~/x'",
        "ssh -i key -p 22 host rm -rf ~/x",
    ],
)
def test_nested_command_inherits_rm_rules(decide, command: str) -> None:
    decision = decide(command)

    assert not decision.allow
    assert decision.rule == "rm"


def test_nested_payload_allowed_when_inner_is_safe(decide) -> None:
    assert decide("bash -c 'echo hi'").allow
    assert decide("ssh host 'ls -la'").allow


def test_nesting_beyond_limit_is_denied(decide) -> None:
    command = "bash -c \"bash -c 'bash -c \\\"bash -c true\\\"'\""

    decision = decide(command)

    assert not decision.allow
    assert decision.rule == "internal_error"


def test_ssh_remote_freeze_is_enforced_inside_window(decide) -> None:
    assert decide("ssh host 'docker restart krx-collector'").rule == "freeze"
    assert decide("ssh host 'docker restart krx-collector'", AFTER_FREEZE).allow


@pytest.mark.parametrize("command", ["rm -rf $TARGET", "rm -rf scratch/$NAME", "rm -rf ${DIR}/x"])
def test_variable_rm_target_denied(decide, command: str) -> None:
    decision = decide(command)

    assert not decision.allow
    assert decision.rule == "rm"


def test_xargs_recursive_rm_denied(decide) -> None:
    decision = decide("xargs rm -rf < list.txt")

    assert not decision.allow
    assert decision.rule == "rm"


def test_xargs_non_recursive_rm_allowed(decide) -> None:
    assert decide("xargs rm < list.txt").allow


@pytest.mark.parametrize("command", ["find . -delete", "find / -name x -delete", "find ~ -exec rm {} ;", "find .. -delete"])
def test_find_delete_outside_safe_scope_denied(decide, command: str) -> None:
    decision = decide(command)

    assert not decision.allow
    assert decision.rule == "rm"


def test_find_delete_inside_trash_allowed(decide) -> None:
    assert decide("find scratch/_trash -mtime +14 -delete").allow


def test_find_without_delete_is_never_denied(decide) -> None:
    assert decide("find . -name '*.py'").allow


def test_quoted_heredoc_body_is_not_parsed_as_commands(decide) -> None:
    command = "git commit -F - <<'EOF'\nfix: rm -rf ~/x and `docker restart` notes\nEOF"

    assert decide(command).allow


def test_unquoted_heredoc_with_substitution_is_denied(decide) -> None:
    command = "cat <<EOF\n$(rm -rf ~/x)\nEOF"

    decision = decide(command)

    assert not decision.allow


def test_heredoc_does_not_hide_following_commands(decide) -> None:
    command = "cat <<'EOF'\nhello\nEOF\nrm -rf ~/x"

    decision = decide(command)

    assert not decision.allow
    assert decision.rule == "rm"


@pytest.mark.parametrize(
    "command",
    ["echo 'docker compose up is frozen'", "printf '%s' 'docker restart x'", "git commit -m 'docker restart notes'"],
)
def test_text_commands_are_not_treated_as_deploy_actions(decide, command: str) -> None:
    assert decide(command).allow


def test_real_compose_up_still_frozen(decide) -> None:
    assert decide("docker compose up -d").rule == "freeze"


@pytest.mark.parametrize(
    "command",
    [
        'echo "docker status: $(git status)"',
        'git commit -m "$(cat msg.txt)"',
        "rm scratch/$NAME.txt",
        "ls $(pwd)",
        "echo '`rm -rf ~/x`'",
        "echo '$(rm -rf ~/x)'",
    ],
)
def test_benign_substitution_and_inert_quotes_allowed(decide, command: str) -> None:
    assert decide(command).allow


@pytest.mark.parametrize(
    ("command", "rule"),
    [
        ("echo $(rm -rf ~/x)", "rm"),
        ("echo `rm -rf ~/x`", "rm"),
        ('echo "$(rm -rf ~/x)"', "rm"),
        ('echo "`rm -rf ~/x`"', "rm"),
        ("cat <<EOF\n$(rm -rf ~/x)\nEOF", "rm"),
        ("ls <(rm -rf ~/x)", "rm"),
    ],
)
def test_substitution_bodies_are_evaluated_everywhere_active(decide, command: str, rule: str) -> None:
    decision = decide(command)

    assert not decision.allow
    assert decision.rule == rule


def test_quoted_heredoc_body_substitution_is_inert(decide) -> None:
    assert decide("cat <<'EOF'\n$(rm -rf ~/x)\nEOF").allow


@pytest.mark.parametrize(
    "command",
    [
        "$(echo rm) -rf ~/x",
        "$(echo git) push --force origin main",
        "$(echo docker) compose up -d",
        "X=rm; $X -rf ~/x",
        'bash -c "$(echo rm -rf ~/x)"',
    ],
)
def test_dynamic_command_word_with_risky_tool_denied(decide, command: str) -> None:
    decision = decide(command)

    assert not decision.allow
    assert decision.rule == "internal_error"


@pytest.mark.parametrize("command", ['"$(git rev-parse --show-toplevel)/scripts/run.sh"', '"$VENV/bin/python" x.py'])
def test_dynamic_command_word_without_risky_tool_allowed(decide, command: str) -> None:
    assert decide(command).allow


def test_substitution_nesting_beyond_limit_denied(decide) -> None:
    decision = decide("echo $(echo $(echo $(echo $(true))))")

    assert not decision.allow
    assert decision.rule == "internal_error"


@pytest.mark.parametrize("command", ["rm -rf `pwd", "echo $(pwd", 'echo "$(pwd"'])
def test_unterminated_substitution_never_raises(decide, command: str) -> None:
    decide(command)


def test_unterminated_substitution_with_recursive_rm_denied(decide) -> None:
    assert not decide("rm -rf `pwd").allow


@pytest.mark.parametrize("command", ["git push origin $BRANCH", "git push origin $(echo main)", "git push origin `echo main`"])
def test_push_with_expansion_denied(decide, command: str) -> None:
    decision = decide(command)

    assert not decision.allow
    assert decision.rule == "internal_error"


def test_push_without_expansion_allowed(decide) -> None:
    assert decide("git push origin main").allow


def test_order_with_expansion_denied(decide) -> None:
    command = "KRX_ALPHA_EXEC_MODE=mock uv run python -m src.cli.main order --symbol $S --side buy --qty 1"

    decision = decide(command)

    assert not decision.allow
    assert decision.rule == "internal_error"
