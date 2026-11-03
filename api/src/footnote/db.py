"""SQLite connection handling and the schema.

One file holds everything: documents, chunks, the FTS5 index, embedding vectors, and
chat history. That keeps ingest transactional (a document and its chunks, FTS rows,
and vectors land together or not at all) and makes a collection trivially portable.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE collections (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    description TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL,
    -- Bumped on every change to the collection's chunks; the in-memory vector cache
    -- compares it to decide whether its matrix is stale.
    revision    INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE documents (
    id            TEXT PRIMARY KEY,
    collection_id TEXT NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
    path          TEXT NOT NULL,
    title         TEXT NOT NULL,
    format        TEXT NOT NULL,
    content_hash  TEXT NOT NULL,
    size_bytes    INTEGER NOT NULL,
    text          TEXT NOT NULL,
    blocks_json   TEXT NOT NULL,
    ingested_at   TEXT NOT NULL,
    UNIQUE (collection_id, path)
);

CREATE TABLE chunks (
    id            INTEGER PRIMARY KEY,
    document_id   TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    collection_id TEXT NOT NULL,
    ordinal       INTEGER NOT NULL,
    start_char    INTEGER NOT NULL,
    end_char      INTEGER NOT NULL,
    heading_path  TEXT NOT NULL,
    page_start    INTEGER,
    page_end      INTEGER,
    text          TEXT NOT NULL,
    embed_hash    TEXT NOT NULL
);
CREATE INDEX chunks_by_collection ON chunks(collection_id);
CREATE INDEX chunks_by_document ON chunks(document_id, ordinal);

CREATE VIRTUAL TABLE chunks_fts USING fts5(
    text, heading_path,
    content='chunks', content_rowid='id',
    tokenize='porter unicode61'
);
CREATE TRIGGER chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text, heading_path)
    VALUES (new.id, new.text, new.heading_path);
END;
CREATE TRIGGER chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text, heading_path)
    VALUES ('delete', old.id, old.text, old.heading_path);
END;

-- Keyed by a hash of the embedded text, not by chunk id, so re-ingesting an edited
-- file only embeds the chunks whose text actually changed.
CREATE TABLE embeddings (
    embed_hash TEXT NOT NULL,
    model      TEXT NOT NULL,
    dim        INTEGER NOT NULL,
    vector     BLOB NOT NULL,
    PRIMARY KEY (embed_hash, model)
);

CREATE TABLE conversations (
    id            TEXT PRIMARY KEY,
    collection_id TEXT NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
    title         TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);
CREATE INDEX conversations_by_collection ON conversations(collection_id, updated_at);

CREATE TABLE messages (
    id              TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role            TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content         TEXT NOT NULL,
    payload_json    TEXT,
    created_at      TEXT NOT NULL
);
CREATE INDEX messages_by_conversation ON messages(conversation_id, created_at);
"""


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def connect(path: Path) -> sqlite3.Connection:
    """Open a connection with the pragmas every caller needs.

    Connections are used from FastAPI's worker threads and the ingest thread, but never
    concurrently: each request or job opens its own. WAL lets readers proceed while the
    single ingest writer holds its transaction.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None, timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def migrate(conn: sqlite3.Connection) -> None:
    (version,) = conn.execute("PRAGMA user_version").fetchone()
    if version == SCHEMA_VERSION:
        return
    if version != 0:
        raise RuntimeError(f"database schema version {version} is newer than this build")
    # executescript() commits any open transaction before it runs, so the BEGIN/COMMIT
    # pair has to live inside the script for the schema to apply atomically.
    conn.executescript(f"BEGIN;\n{SCHEMA}\nPRAGMA user_version = {SCHEMA_VERSION};\nCOMMIT;")


@contextmanager
def transaction(conn: sqlite3.Connection) -> Generator[sqlite3.Connection]:
    """Explicit write transaction. IMMEDIATE takes the write lock up front, so two
    writers fail fast on the busy timeout instead of deadlocking on lock upgrade."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")
