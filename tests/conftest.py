"""Pin pytest and stdlib temporary roots inside the repository tmp/ directory."""

from __future__ import annotations

import os

# 다중 프로젝트 및 로컬 동시성 환경 리소스 안전 가드:
# Polars, NumPy, OpenBLAS, MKL, Numba 등이 8코어 머신에서 스레드를 과도하게 점유하지 못하도록 상한선 강제
for _key, _val in (
    ("POLARS_MAX_THREADS", "2"),
    ("OMP_NUM_THREADS", "1"),
    ("OPENBLAS_NUM_THREADS", "1"),
    ("MKL_NUM_THREADS", "1"),
    ("NUMBA_NUM_THREADS", "1"),
    ("RAY_ACCEL_NUM_WORKERS", "1"),
    ("PYTHONDONTWRITEBYTECODE", "1"),
):
    os.environ.setdefault(_key, _val)

import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PROJECT_TMP = _REPO_ROOT / "tmp"
_PROJECT_TMP.mkdir(parents=True, exist_ok=True)

tempfile.tempdir = str(_PROJECT_TMP)

import pytest


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config: pytest.Config) -> None:
    basetemp = config.option.basetemp
    if basetemp is None:
        config.option.basetemp = str(_PROJECT_TMP / "pytest-of-pytest")


def pytest_sessionstart(session: pytest.Session) -> None:
    import os

    os.environ["TMPDIR"] = str(_PROJECT_TMP)
