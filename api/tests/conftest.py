from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from footnote.config import Settings
from footnote.db import connect, migrate
from footnote.library import create_collection
from tests.fakes import HashingEmbedder

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Defaults only: a developer's FOOTNOTE_* variables or .env must not leak in."""
    for name in [n for n in os.environ if n.startswith("FOOTNOTE_")]:
        monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)  # .env is resolved relative to the working directory
    return Settings(data_dir=tmp_path)


@pytest.fixture
def conn(settings: Settings) -> Iterator[sqlite3.Connection]:
    connection = connect(settings.db_path)
    migrate(connection)
    yield connection
    connection.close()


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder()


@pytest.fixture
def collection_id(conn: sqlite3.Connection) -> str:
    return create_collection(conn, "Test shelf").id


def fixture_bytes(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()
