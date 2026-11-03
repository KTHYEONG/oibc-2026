from __future__ import annotations

import dataclasses
import importlib.util
import io
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GUARDS_PATH = REPO_ROOT / "tools" / "guards" / "guards.py"
SPEC_PATH = REPO_ROOT / "tests" / "unit" / "guards_test_config.toml"
HOOK_PATH = REPO_ROOT / ".githooks" / "pre-commit"
WRAPPER_TEXT = b'#!/bin/sh\nexec python3 -I "$(git rev-parse --show-toplevel)/tools/guards/guards.py" precommit\n'
GIT_TIMEOUT_SECONDS = 60


def _load_guards() -> ModuleType:
    spec = importlib.util.spec_from_file_location("guards_commit_under_test", str(GUARDS_PATH))
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
def spec(guards: ModuleType) -> Any:
    return guards.load_guard_spec(SPEC_PATH)


def _entries(guards: ModuleType, *paths: str, size: int = 1) -> list[Any]:
    return [guards.StagedEntry(path=path, size_bytes=size) for path in paths]


def _fake_git(listing: bytes, sizes: dict[str, bytes]) -> Callable[[Sequence[str]], bytes]:
    def run(args: Sequence[str]) -> bytes:
        if "diff" in args:
            return listing
        return sizes[args[-1].removeprefix(":")]

    return run


def test_empty_stage_allowed(guards: ModuleType, spec: Any) -> None:
    assert guards.evaluate_staged([], spec).allow


@pytest.mark.parametrize("path", ["scratch/20261103-x/a.py", "tmp/pytest/x", "docs/specs/a_spec.md"])
def test_forbidden_prefix_denied(guards: ModuleType, spec: Any, path: str) -> None:
    decision = guards.evaluate_staged(_entries(guards, path), spec)
    assert not decision.allow
    assert decision.rule == "staged_path"
    assert f"  staged_path: {path}" in decision.reason


@pytest.mark.parametrize("path", ["scratch_utils/a.py", "docs/specs_archive/a.md"])
def test_similar_prefix_allowed(guards: ModuleType, spec: Any, path: str) -> None:
    assert guards.evaluate_staged(_entries(guards, path), spec).allow


@pytest.mark.parametrize("path", [".env", ".env.local", "sub/.envrc", "keys/id.pem", "keys/id.key"])
def test_secret_name_globs_denied(guards: ModuleType, spec: Any, path: str) -> None:
    decision = guards.evaluate_staged(_entries(guards, path), spec)
    assert not decision.allow
    assert decision.rule == "staged_name"


def test_name_globs_are_case_sensitive(guards: ModuleType, spec: Any) -> None:
    assert guards.evaluate_staged(_entries(guards, "keys/ID.PEM"), spec).allow


def test_size_cap_is_strict(guards: ModuleType, spec: Any) -> None:
    at_cap = _entries(guards, "src/big.bin", size=spec.max_staged_bytes)
    assert guards.evaluate_staged(at_cap, spec).allow
    over = _entries(guards, "src/big.bin", size=spec.max_staged_bytes + 1)
    decision = guards.evaluate_staged(over, spec)
    assert not decision.allow
    assert decision.rule == "staged_size"


@pytest.mark.parametrize("path", ["scratch/.gitkeep", "docs/specs/.gitkeep"])
def test_allowed_name_exempt_from_prefix_rule(guards: ModuleType, spec: Any, path: str) -> None:
    assert guards.evaluate_staged(_entries(guards, path), spec).allow


def test_non_allowed_name_under_prefix_still_denied(guards: ModuleType, spec: Any) -> None:
    decision = guards.evaluate_staged(_entries(guards, "scratch/a.py"), spec)
    assert not decision.allow
    assert decision.rule == "staged_path"


def test_allowed_name_still_size_checked(guards: ModuleType, spec: Any) -> None:
    entries = _entries(guards, "scratch/.gitkeep", size=spec.max_staged_bytes + 1)
    decision = guards.evaluate_staged(entries, spec)
    assert not decision.allow
    assert decision.rule == "staged_size"


def test_allowed_name_not_exempt_from_name_glob(guards: ModuleType, spec: Any) -> None:
    config = dataclasses.replace(spec, allowed_staged_names=(".env",))
    decision = guards.evaluate_staged(_entries(guards, "scratch/.env"), config)
    assert not decision.allow
    assert decision.rule == "staged_name"


def test_all_offenders_listed_and_capped(guards: ModuleType, spec: Any) -> None:
    paths = [f"scratch/f{index:02d}.py" for index in range(25)]
    decision = guards.evaluate_staged(_entries(guards, *paths), spec)
    lines = decision.reason.splitlines()
    listed = [line for line in lines if line.startswith("  staged_path: ")]
    assert len(listed) == 20
    assert listed[-1] == "  staged_path: scratch/f19.py"
    assert "  ... and 5 more" in lines
    assert "git restore --staged" in decision.reason


def test_one_rule_label_per_entry(guards: ModuleType, spec: Any) -> None:
    entries = _entries(guards, "scratch/.env", size=spec.max_staged_bytes + 1)
    decision = guards.evaluate_staged(entries, spec)
    assert decision.rule == "staged_path"
    assert decision.reason.splitlines().count("  staged_path: scratch/.env") == 1
    assert "staged_name" not in decision.reason
    assert "staged_size" not in decision.reason


def test_rule_lists_distinct_labels_in_first_seen_order(guards: ModuleType, spec: Any) -> None:
    entries = _entries(guards, "scratch/a.py") + _entries(guards, ".env") + _entries(guards, "src/b.bin", size=10**9)
    assert guards.evaluate_staged(entries, spec).rule == "staged_path,staged_name,staged_size"


def test_unicode_and_spaced_paths(guards: ModuleType, spec: Any) -> None:
    assert guards.evaluate_staged(_entries(guards, "src/한 글 파일.py"), spec).allow
    decision = guards.evaluate_staged(_entries(guards, "scratch/한 글.py"), spec)
    assert not decision.allow
    assert "scratch/한 글.py" in decision.reason


@pytest.mark.parametrize("value", ["-1", "true"])
def test_load_rejects_invalid_size_cap(guards: ModuleType, tmp_path: Path, value: str) -> None:
    text = SPEC_PATH.read_text(encoding="utf-8")
    broken = tmp_path / "guards.toml"
    broken.write_text(text.replace("max_staged_bytes = 5000000", f"max_staged_bytes = {value}"), encoding="utf-8")
    with pytest.raises(ValueError, match="max_staged_bytes"):
        guards.load_guard_spec(broken)


def test_precommit_allows_clean_stage_silently(guards: ModuleType, spec: Any, tmp_path: Path) -> None:
    stderr = io.StringIO()
    git = _fake_git(b"src/a.py\0", {"src/a.py": b"12\n"})
    assert guards.main_precommit(tmp_path, git, stderr, spec) == 0
    assert stderr.getvalue() == ""


def test_precommit_denies_with_reason_on_stderr(guards: ModuleType, spec: Any, tmp_path: Path) -> None:
    stderr = io.StringIO()
    git = _fake_git(b"src/a.py\0scratch/b.py\0", {"src/a.py": b"1\n", "scratch/b.py": b"1\n"})
    assert guards.main_precommit(tmp_path, git, stderr, spec) == 1
    assert "  staged_path: scratch/b.py" in stderr.getvalue()
    assert "src/a.py" not in stderr.getvalue()


def test_git_failure_fails_closed(guards: ModuleType, spec: Any, tmp_path: Path) -> None:
    def failing(args: Sequence[str]) -> bytes:
        raise subprocess.CalledProcessError(128, ["git", *args])

    stderr = io.StringIO()
    assert guards.main_precommit(tmp_path, failing, stderr, spec) == 1
    assert stderr.getvalue().startswith("guard internal error (fail-closed)")


def test_unparsable_blob_size_fails_closed(guards: ModuleType, spec: Any, tmp_path: Path) -> None:
    stderr = io.StringIO()
    git = _fake_git(b"src/a.py\0", {"src/a.py": b"not-a-size\n"})
    assert guards.main_precommit(tmp_path, git, stderr, spec) == 1
    assert stderr.getvalue().startswith("guard internal error (fail-closed)")


def test_non_utf8_path_fails_closed(guards: ModuleType, spec: Any, tmp_path: Path) -> None:
    stderr = io.StringIO()
    assert guards.main_precommit(tmp_path, _fake_git(b"\xff.py\0", {}), stderr, spec) == 1
    assert stderr.getvalue().startswith("guard internal error (fail-closed)")


def test_precommit_wrapper_is_exact_contract() -> None:
    assert HOOK_PATH.read_bytes() == WRAPPER_TEXT
    assert os.access(HOOK_PATH, os.X_OK)


def _hermetic_env(home: Path) -> dict[str, str]:
    home.mkdir(parents=True, exist_ok=True)
    return {
        "PATH": os.environ.get("PATH", os.defpath),
        "HOME": str(home),
        "LC_ALL": "C.UTF-8",
        "PYTHONIOENCODING": "utf-8",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_SYSTEM": os.devnull,
        "GIT_AUTHOR_NAME": "Guard Test",
        "GIT_AUTHOR_EMAIL": "guard-test@example.invalid",
        "GIT_COMMITTER_NAME": "Guard Test",
        "GIT_COMMITTER_EMAIL": "guard-test@example.invalid",
    }


def _git(repo: Path, env: dict[str, str], *args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(["git", *args], cwd=repo, env=env, capture_output=True, check=check, timeout=GIT_TIMEOUT_SECONDS)  # noqa: S603, S607


def _head(repo: Path, env: dict[str, str]) -> str:
    return _git(repo, env, "rev-parse", "HEAD").stdout.decode("ascii").strip()


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture
def env(tmp_path: Path) -> dict[str, str]:
    return _hermetic_env(tmp_path / "home")


@pytest.fixture
def hooked_repo(tmp_path: Path, env: dict[str, str]) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, env, "init", "-q", "-b", "main")
    _write(repo / "src" / "base.py", "BASE = 1\n")
    _write(repo / "tmp" / "old.txt", "stale\n")
    _git(repo, env, "add", "-f", "--", "src/base.py", "tmp/old.txt")
    _git(repo, env, "commit", "-q", "-m", "base")
    (repo / ".githooks").mkdir()
    shutil.copy2(HOOK_PATH, repo / ".githooks" / "pre-commit")
    (repo / "tools" / "guards").mkdir(parents=True)
    shutil.copy2(GUARDS_PATH, repo / "tools" / "guards" / "guards.py")
    shutil.copy2(SPEC_PATH, repo / "tools" / "guards" / "guards.toml")
    _git(repo, env, "config", "core.hooksPath", ".githooks")
    return repo


def test_hook_blocks_commit_with_scratch_file(hooked_repo: Path, env: dict[str, str]) -> None:
    _write(hooked_repo / "scratch" / "a.py", "x = 1\n")
    _git(hooked_repo, env, "add", "scratch/a.py")
    before = _head(hooked_repo, env)
    result = _git(hooked_repo, env, "commit", "-q", "-m", "scratch", check=False)
    assert result.returncode != 0
    assert b"scratch/a.py" in result.stderr
    assert b"git restore --staged" in result.stderr
    assert _head(hooked_repo, env) == before


def test_hook_allows_clean_commit(hooked_repo: Path, env: dict[str, str]) -> None:
    _write(hooked_repo / "src" / "a.py", "A = 1\n")
    _git(hooked_repo, env, "add", "src/a.py")
    before = _head(hooked_repo, env)
    _git(hooked_repo, env, "commit", "-q", "-m", "clean")
    assert _head(hooked_repo, env) != before


def test_hook_allows_deletion_of_forbidden_path(hooked_repo: Path, env: dict[str, str]) -> None:
    _git(hooked_repo, env, "rm", "-q", "tmp/old.txt")
    before = _head(hooked_repo, env)
    _git(hooked_repo, env, "commit", "-q", "-m", "drop stale")
    assert _head(hooked_repo, env) != before


def test_hook_handles_spaces_and_unicode_source_paths(hooked_repo: Path, env: dict[str, str]) -> None:
    _write(hooked_repo / "src" / "한 글 파일.py", "K = 1\n")
    _write(hooked_repo / "src" / "a b.py", "B = 2\n")
    _git(hooked_repo, env, "add", "--", "src/한 글 파일.py", "src/a b.py")
    before = _head(hooked_repo, env)
    _git(hooked_repo, env, "commit", "-q", "-m", "unicode")
    assert _head(hooked_repo, env) != before


def test_hook_blocks_unicode_scratch_path(hooked_repo: Path, env: dict[str, str]) -> None:
    _write(hooked_repo / "scratch" / "한 글.py", "x = 1\n")
    _git(hooked_repo, env, "add", "--", "scratch/한 글.py")
    before = _head(hooked_repo, env)
    result = _git(hooked_repo, env, "commit", "-q", "-m", "scratch", check=False)
    assert result.returncode != 0
    assert "scratch/한 글.py".encode() in result.stderr
    assert _head(hooked_repo, env) == before


def test_hook_fails_closed_when_guard_is_missing(hooked_repo: Path, env: dict[str, str]) -> None:
    (hooked_repo / "tools" / "guards" / "guards.py").unlink()
    _write(hooked_repo / "src" / "a.py", "A = 1\n")
    _git(hooked_repo, env, "add", "src/a.py")
    before = _head(hooked_repo, env)
    result = _git(hooked_repo, env, "commit", "-q", "-m", "clean", check=False)
    assert result.returncode != 0
    assert _head(hooked_repo, env) == before
