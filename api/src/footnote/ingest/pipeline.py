"""Turn files into searchable chunks, skipping work that has already been done.

Two hashes make re-ingest incremental:

* the SHA-256 of the file bytes: an unchanged file is skipped before it is even parsed;
* the SHA-256 of each chunk's embedding input: when a file *does* change, only chunks
  whose text changed are sent to the embedding model. Editing one paragraph of a long
  document re-embeds one or two chunks, not all of them.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Literal

from footnote.db import transaction, utc_now
from footnote.ingest.chunker import DEFAULT_CHUNKING, Chunk, ChunkingConfig, chunk_document
from footnote.ingest.document import CanonicalDocument, assemble
from footnote.ingest.parsers import parse_document
from footnote.library import bump_revision, encode_heading_path, get_collection, new_id
from footnote.log import get_logger
from footnote.retrieval.embedder import Embedder, Vector
from footnote.retrieval.vectors import to_blob

log = get_logger(__name__)

Stage = Literal["parsing", "chunking", "embedding", "storing"]
ProgressCallback = Callable[[Stage, dict[str, int]], None]


@dataclass(frozen=True, slots=True)
class IngestOutcome:
    status: Literal["added", "updated", "unchanged"]
    document_id: str
    chunks: int
    embedded: int
    reused: int


def content_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def embedding_input(title: str, chunk: Chunk) -> str:
    """What the dense model sees: the document title and section path give a chunk that
    says "Use 4 spaces per indentation level." the context that it is about Python."""
    path = " > ".join(dict.fromkeys((title, *chunk.heading_path)))
    return f"{path}\n\n{chunk.text}"


def embed_hash(model: str, text: str) -> str:
    return hashlib.sha256(f"{model}\x00{text}".encode()).hexdigest()


def ingest_file(
    conn: sqlite3.Connection,
    embedder: Embedder,
    collection_id: str,
    path: str,
    data: bytes,
    *,
    chunking: ChunkingConfig = DEFAULT_CHUNKING,
    on_progress: ProgressCallback | None = None,
) -> IngestOutcome:
    """Parse, chunk, embed, and store one file. Raises the parser's errors unchanged."""
    report = on_progress or _ignore_progress
    get_collection(conn, collection_id)
    digest = content_hash(data)
    existing = conn.execute(
        "SELECT id, content_hash FROM documents WHERE collection_id = ? AND path = ?",
        (collection_id, path),
    ).fetchone()
    if existing is not None and existing["content_hash"] == digest:
        chunks = conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE document_id = ?", (existing["id"],)
        ).fetchone()[0]
        return IngestOutcome("unchanged", existing["id"], chunks, 0, chunks)

    report("parsing", {})
    document = assemble(parse_document(path, data))
    report("chunking", {})
    chunks = chunk_document(document, chunking)

    model = embedder.model_name
    inputs = [embedding_input(document.title, chunk) for chunk in chunks]
    hashes = [embed_hash(model, text) for text in inputs]
    missing = _missing_embeddings(conn, model, hashes)
    report("embedding", {"total": len(missing), "done": 0})
    vectors = _embed_in_batches(embedder, [inputs[i] for i in missing], report)

    report("storing", {})
    document_id = existing["id"] if existing is not None else new_id("doc")
    with transaction(conn):
        _store(conn, collection_id, document_id, path, data, digest, document, chunks, hashes)
        conn.executemany(
            "INSERT OR IGNORE INTO embeddings (embed_hash, model, dim, vector) VALUES (?, ?, ?, ?)",
            [
                (hashes[i], model, int(vector.shape[0]), to_blob(vector))
                for i, vector in zip(missing, vectors, strict=True)
            ],
        )
        bump_revision(conn, collection_id)

    status: Literal["added", "updated"] = "added" if existing is None else "updated"
    log.info(
        "ingested",
        path=path,
        status=status,
        chunks=len(chunks),
        embedded=len(missing),
        reused=len(chunks) - len(missing),
    )
    return IngestOutcome(status, document_id, len(chunks), len(missing), len(chunks) - len(missing))


def _missing_embeddings(conn: sqlite3.Connection, model: str, hashes: Sequence[str]) -> list[int]:
    known: set[str] = set()
    unique = list(dict.fromkeys(hashes))
    for start in range(0, len(unique), 500):
        batch = unique[start : start + 500]
        placeholders = ",".join("?" * len(batch))
        rows = conn.execute(
            f"SELECT embed_hash FROM embeddings WHERE model = ? AND embed_hash IN ({placeholders})",  # noqa: S608
            [model, *batch],
        ).fetchall()
        known.update(r["embed_hash"] for r in rows)
    first_index: dict[str, int] = {}
    for index, value in enumerate(hashes):
        if value not in known:
            first_index.setdefault(value, index)
    return sorted(first_index.values())


def _ignore_progress(_stage: Stage, _counts: dict[str, int]) -> None:
    return None


def _embed_in_batches(
    embedder: Embedder, texts: list[str], report: ProgressCallback, batch_size: int = 64
) -> list[Vector]:
    vectors: list[Vector] = []
    for start in range(0, len(texts), batch_size):
        batch = embedder.embed_passages(texts[start : start + batch_size])
        vectors.extend(batch[row] for row in range(batch.shape[0]))
        report("embedding", {"total": len(texts), "done": len(vectors)})
    return vectors


def _store(
    conn: sqlite3.Connection,
    collection_id: str,
    document_id: str,
    path: str,
    data: bytes,
    digest: str,
    document: CanonicalDocument,
    chunks: list[Chunk],
    hashes: list[str],
) -> None:
    blocks_json = json.dumps([asdict(block) for block in document.blocks])
    conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
    conn.execute(
        """
        INSERT INTO documents (id, collection_id, path, title, format, content_hash,
                               size_bytes, text, blocks_json, ingested_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (id) DO UPDATE SET
            title = excluded.title, format = excluded.format,
            content_hash = excluded.content_hash, size_bytes = excluded.size_bytes,
            text = excluded.text, blocks_json = excluded.blocks_json,
            ingested_at = excluded.ingested_at
        """,
        (
            document_id,
            collection_id,
            path,
            document.title,
            document.format,
            digest,
            len(data),
            document.text,
            blocks_json,
            utc_now(),
        ),
    )
    conn.executemany(
        """
        INSERT INTO chunks (document_id, collection_id, ordinal, start_char, end_char,
                            heading_path, page_start, page_end, text, embed_hash)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                document_id,
                collection_id,
                c.ordinal,
                c.start,
                c.end,
                encode_heading_path(c.heading_path),
                c.page_start,
                c.page_end,
                c.text,
                h,
            )
            for c, h in zip(chunks, hashes, strict=True)
        ],
    )


def prune_missing(conn: sqlite3.Connection, collection_id: str, keep_paths: set[str]) -> list[str]:
    """Delete documents whose path is not in ``keep_paths``; returns the removed paths."""
    rows = conn.execute(
        "SELECT id, path FROM documents WHERE collection_id = ?", (collection_id,)
    ).fetchall()
    removed = [r for r in rows if r["path"] not in keep_paths]
    if removed:
        with transaction(conn):
            conn.executemany("DELETE FROM documents WHERE id = ?", [(r["id"],) for r in removed])
            bump_revision(conn, collection_id)
    return [r["path"] for r in removed]
