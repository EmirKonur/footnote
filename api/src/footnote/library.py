"""Collections and documents: the read/write queries the API and CLI share."""

from __future__ import annotations

import json
import secrets
import sqlite3
from collections.abc import Sequence

from footnote.db import transaction, utc_now
from footnote.schemas import (
    Collection,
    DocumentDetail,
    DocumentSummary,
    Passage,
    ViewerBlock,
)


class NotFoundError(LookupError):
    """The requested collection, document, or conversation does not exist."""


class ConflictError(ValueError):
    """The write would violate a uniqueness rule (for example, a duplicate name)."""


def new_id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(8)}"


# Collections ------------------------------------------------------------------------

_COLLECTION_SELECT = """
    SELECT c.id, c.name, c.description, c.created_at,
           (SELECT COUNT(*) FROM documents d WHERE d.collection_id = c.id) AS document_count,
           (SELECT COUNT(*) FROM chunks k WHERE k.collection_id = c.id) AS chunk_count
    FROM collections c
"""


def _collection(row: sqlite3.Row) -> Collection:
    return Collection(
        id=row["id"],
        name=row["name"],
        description=row["description"],
        created_at=row["created_at"],
        document_count=row["document_count"],
        chunk_count=row["chunk_count"],
    )


def list_collections(conn: sqlite3.Connection) -> list[Collection]:
    rows = conn.execute(_COLLECTION_SELECT + " ORDER BY c.created_at").fetchall()
    return [_collection(r) for r in rows]


def get_collection(conn: sqlite3.Connection, collection_id: str) -> Collection:
    row = conn.execute(_COLLECTION_SELECT + " WHERE c.id = ?", (collection_id,)).fetchone()
    if row is None:
        raise NotFoundError(f"collection {collection_id} not found")
    return _collection(row)


def find_collection_by_name(conn: sqlite3.Connection, name: str) -> Collection | None:
    row = conn.execute(_COLLECTION_SELECT + " WHERE c.name = ?", (name,)).fetchone()
    return _collection(row) if row else None


def create_collection(conn: sqlite3.Connection, name: str, description: str = "") -> Collection:
    collection_id = new_id("col")
    try:
        with transaction(conn):
            conn.execute(
                "INSERT INTO collections (id, name, description, created_at) VALUES (?, ?, ?, ?)",
                (collection_id, name, description, utc_now()),
            )
    except sqlite3.IntegrityError as exc:
        raise ConflictError(f"a collection named {name!r} already exists") from exc
    return get_collection(conn, collection_id)


def update_collection(
    conn: sqlite3.Connection, collection_id: str, name: str, description: str | None
) -> Collection:
    get_collection(conn, collection_id)
    try:
        with transaction(conn):
            conn.execute(
                "UPDATE collections SET name = ?, description = COALESCE(?, description) "
                "WHERE id = ?",
                (name, description, collection_id),
            )
    except sqlite3.IntegrityError as exc:
        raise ConflictError(f"a collection named {name!r} already exists") from exc
    return get_collection(conn, collection_id)


def delete_collection(conn: sqlite3.Connection, collection_id: str) -> None:
    with transaction(conn):
        deleted = conn.execute("DELETE FROM collections WHERE id = ?", (collection_id,)).rowcount
    if not deleted:
        raise NotFoundError(f"collection {collection_id} not found")


def bump_revision(conn: sqlite3.Connection, collection_id: str) -> None:
    conn.execute("UPDATE collections SET revision = revision + 1 WHERE id = ?", (collection_id,))


# Documents --------------------------------------------------------------------------

_DOCUMENT_SELECT = """
    SELECT d.id, d.collection_id, d.path, d.title, d.format, d.size_bytes, d.ingested_at,
           (SELECT COUNT(*) FROM chunks k WHERE k.document_id = d.id) AS chunk_count
    FROM documents d
"""


def _document_summary(row: sqlite3.Row) -> DocumentSummary:
    return DocumentSummary(
        id=row["id"],
        collection_id=row["collection_id"],
        path=row["path"],
        title=row["title"],
        format=row["format"],
        size_bytes=row["size_bytes"],
        chunk_count=row["chunk_count"],
        ingested_at=row["ingested_at"],
    )


def list_documents(conn: sqlite3.Connection, collection_id: str) -> list[DocumentSummary]:
    get_collection(conn, collection_id)
    rows = conn.execute(
        _DOCUMENT_SELECT + " WHERE d.collection_id = ? ORDER BY d.title COLLATE NOCASE",
        (collection_id,),
    ).fetchall()
    return [_document_summary(r) for r in rows]


def get_document(conn: sqlite3.Connection, document_id: str) -> DocumentDetail:
    row = conn.execute(
        """
        SELECT d.*, (SELECT COUNT(*) FROM chunks k WHERE k.document_id = d.id) AS chunk_count
        FROM documents d WHERE d.id = ?
        """,
        (document_id,),
    ).fetchone()
    if row is None:
        raise NotFoundError(f"document {document_id} not found")
    summary = _document_summary(row)
    blocks = [ViewerBlock.model_validate(b) for b in json.loads(row["blocks_json"])]
    return DocumentDetail(**summary.model_dump(), text=row["text"], blocks=blocks)


def delete_document(conn: sqlite3.Connection, document_id: str) -> None:
    with transaction(conn):
        row = conn.execute(
            "SELECT collection_id FROM documents WHERE id = ?", (document_id,)
        ).fetchone()
        if row is None:
            raise NotFoundError(f"document {document_id} not found")
        conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))
        bump_revision(conn, row["collection_id"])


# Chunks -----------------------------------------------------------------------------


def passages_for_chunks(
    conn: sqlite3.Connection,
    chunk_ids: Sequence[int],
    scores: Sequence[float],
    ranks: Sequence[dict[str, int]],
) -> list[Passage]:
    """Hydrate ranked chunk ids into passages, keeping the input order."""
    if not chunk_ids:
        return []
    placeholders = ",".join("?" * len(chunk_ids))
    rows = conn.execute(
        f"""
        SELECT c.id, c.document_id, c.heading_path, c.page_start, c.page_end,
               c.start_char, c.end_char, c.text, d.title, d.path
        FROM chunks c JOIN documents d ON d.id = c.document_id
        WHERE c.id IN ({placeholders})
        """,  # noqa: S608 (placeholders only)
        list(chunk_ids),
    ).fetchall()
    by_id = {r["id"]: r for r in rows}
    passages: list[Passage] = []
    for chunk_id, score, rank in zip(chunk_ids, scores, ranks, strict=True):
        row = by_id.get(chunk_id)
        if row is None:
            continue
        passages.append(
            Passage(
                chunk_id=chunk_id,
                document_id=row["document_id"],
                document_title=row["title"],
                document_path=row["path"],
                heading_path=decode_heading_path(row["heading_path"]),
                page_start=row["page_start"],
                page_end=row["page_end"],
                start=row["start_char"],
                end=row["end_char"],
                text=row["text"],
                score=score,
                ranks=rank,
            )
        )
    return passages


def encode_heading_path(path: Sequence[str]) -> str:
    # Stored as JSON so a heading containing any separator character round-trips; FTS5's
    # tokenizer skips the brackets and quotes.
    return json.dumps(list(path), ensure_ascii=False)


def decode_heading_path(stored: str) -> list[str]:
    decoded: list[str] = json.loads(stored)
    return decoded
