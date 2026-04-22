"""
Shared pytest fixtures.

Tests run against a throw-away SQLite DB and LanceDB directory so they never
touch the developer's local stores.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest


@pytest.fixture(scope="session", autouse=True)
def _isolated_storage() -> None:
    """Point the app at a temporary data directory for the whole test session."""
    tmp = Path(tempfile.mkdtemp(prefix="platform-tests-"))
    os.environ.setdefault("DB_PATH", str(tmp / "test.db"))
    os.environ.setdefault("LANCEDB_PATH", str(tmp / "lancedb"))
    os.environ.setdefault("SOURCES_DIR", str(tmp / "sources"))
    os.environ.setdefault("GROQ_API_KEY", "test-key-not-used")
    yield
