from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[2]
GUARDS_PATH = REPO_ROOT / "tools" / "guards" / "guards.py"
SPEC_PATH = REPO_ROOT / "tools" / "guards" / "guards.toml"
HOOK_PATH = REPO_ROOT / ".githooks" / "pre-commit"


def _load_guards() -> ModuleType:
    spec = importlib.util.spec_from_file_location("guards_project_under_test", str(GUARDS_PATH))
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_project_guard_config_loads() -> None:
    _load_guards().load_guard_spec(SPEC_PATH)


def test_pre_commit_hook_is_executable() -> None:
    assert HOOK_PATH.is_file()
    assert os.access(HOOK_PATH, os.X_OK)
