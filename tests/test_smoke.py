"""Smoke test to ensure test environment and module structure work properly."""

from __future__ import annotations


def test_import_smoke() -> None:
    import src
    import src.oibc

    assert src is not None
    assert src.oibc is not None
