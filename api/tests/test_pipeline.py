from __future__ import annotations

import sqlite3

import pytest

from footnote import library
from footnote.ingest.chunker import ChunkingConfig
from footnote.ingest.parsers import UnsupportedFormatError
from footnote.ingest.pipeline import ingest_file, prune_missing
from footnote.retrieval.bm25 import fts_query, search_bm25
from footnote.retrieval.search import Retriever
from footnote.retrieval.vectors import VectorIndex
from tests.conftest import fixture_bytes
from tests.fakes import HashingEmbedder

SMALL_CHUNKS = ChunkingConfig(target_chars=120, max_chars=200, overlap_chars=40)


def ingest_fixtures(
    conn: sqlite3.Connection, embedder: HashingEmbedder, collection_id: str
) -> None:
    for name in ("field-guide.md", "catalogue.html", "loans.txt"):
        ingest_file(conn, embedder, collection_id, name, fixture_bytes(name), chunking=SMALL_CHUNKS)


def test_ingest_stores_document_chunks_and_vectors(
    conn: sqlite3.Connection, embedder: HashingEmbedder, collection_id: str
) -> None:
    outcome = ingest_file(
        conn,
        embedder,
        collection_id,
        "field-guide.md",
        fixture_bytes("field-guide.md"),
        chunking=SMALL_CHUNKS,
    )

    assert outcome.status == "added"
    assert outcome.chunks > 3
    assert outcome.embedded == outcome.chunks
    document = library.get_document(conn, outcome.document_id)
    assert document.title == "Field Guide to the Reading Room"
    rows = conn.execute(
        "SELECT start_char, end_char, text FROM chunks WHERE document_id = ?",
        (outcome.document_id,),
    ).fetchall()
    assert all(document.text[r["start_char"] : r["end_char"]] == r["text"] for r in rows)
    vectors = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
    assert vectors == outcome.chunks


def test_unchanged_file_is_skipped_without_parsing(
    conn: sqlite3.Connection, embedder: HashingEmbedder, collection_id: str
) -> None:
    data = fixture_bytes("loans.txt")
    first = ingest_file(conn, embedder, collection_id, "loans.txt", data)
    calls_after_first = len(embedder.calls)
    second = ingest_file(conn, embedder, collection_id, "loans.txt", data)

    assert second.status == "unchanged"
    assert second.document_id == first.document_id
    assert (second.chunks, second.embedded, second.reused) == (first.chunks, 0, first.chunks)
    assert len(embedder.calls) == calls_after_first


def test_editing_one_paragraph_re_embeds_only_the_changed_chunks(
    conn: sqlite3.Connection, embedder: HashingEmbedder, collection_id: str
) -> None:
    original = fixture_bytes("field-guide.md")
    first = ingest_file(conn, embedder, collection_id, "guide.md", original, chunking=SMALL_CHUNKS)
    edited = original.replace(
        b"Bindings marked with a red slip", b"Bindings marked with a blue slip"
    )
    embedder.calls.clear()

    second = ingest_file(conn, embedder, collection_id, "guide.md", edited, chunking=SMALL_CHUNKS)

    assert second.status == "updated"
    assert second.document_id == first.document_id
    assert second.chunks == first.chunks
    assert second.embedded == 1
    assert second.reused == first.chunks - 1
    embedded_texts = [text for call in embedder.calls for text in call]
    assert len(embedded_texts) == 1
    assert "blue slip" in embedded_texts[0]
    # The old wording is gone from the full-text index; the new wording is searchable.
    stored = [r[0] for r in conn.execute("SELECT text FROM chunks").fetchall()]
    assert not any("red slip" in text for text in stored)
    assert search_bm25(conn, collection_id, "blue", 5)


def test_update_replaces_fts_rows_instead_of_duplicating_them(
    conn: sqlite3.Connection, embedder: HashingEmbedder, collection_id: str
) -> None:
    ingest_file(conn, embedder, collection_id, "a.txt", b"Zebra crossings are striped.")
    ingest_file(conn, embedder, collection_id, "a.txt", b"Pelican crossings have lights.")

    assert search_bm25(conn, collection_id, "zebra", 5) == []
    hits = search_bm25(conn, collection_id, "pelican", 5)
    assert len(hits) == 1
    fts_rows = conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
    assert fts_rows == conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]


def test_ingest_bumps_the_collection_revision(
    conn: sqlite3.Connection, embedder: HashingEmbedder, collection_id: str
) -> None:
    def revision() -> int:
        return conn.execute(
            "SELECT revision FROM collections WHERE id = ?", (collection_id,)
        ).fetchone()[0]

    before = revision()
    ingest_file(conn, embedder, collection_id, "a.txt", b"One.")
    ingest_file(conn, embedder, collection_id, "a.txt", b"One.")  # unchanged: no bump
    assert revision() == before + 1


def test_prune_removes_documents_missing_from_the_folder(
    conn: sqlite3.Connection, embedder: HashingEmbedder, collection_id: str
) -> None:
    ingest_fixtures(conn, embedder, collection_id)
    removed = prune_missing(conn, collection_id, {"loans.txt"})

    assert sorted(removed) == ["catalogue.html", "field-guide.md"]
    assert [d.path for d in library.list_documents(conn, collection_id)] == ["loans.txt"]
    orphans = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE document_id NOT IN (SELECT id FROM documents)"
    ).fetchone()[0]
    assert orphans == 0


def test_unsupported_files_raise_before_touching_the_database(
    conn: sqlite3.Connection, embedder: HashingEmbedder, collection_id: str
) -> None:
    with pytest.raises(UnsupportedFormatError):
        ingest_file(conn, embedder, collection_id, "deck.key", b"binary")
    assert library.list_documents(conn, collection_id) == []


def test_fts_query_quotes_terms_and_drops_stopwords() -> None:
    assert (
        fts_query("What is the NEAR(maximum) line-length?")
        == '"near" OR "maximum" OR "line" OR "length"'
    )
    assert (
        fts_query('" OR "* AND') == '"or" OR "and"'
    )  # all stopwords: keep them rather than nothing
    assert fts_query("?!") is None


@pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid"])
def test_search_modes_find_the_right_document(
    conn: sqlite3.Connection, embedder: HashingEmbedder, collection_id: str, mode: str
) -> None:
    ingest_fixtures(conn, embedder, collection_id)
    retriever = Retriever(embedder, VectorIndex())
    assert mode in ("bm25", "dense", "hybrid")

    passages = retriever.search(
        conn, collection_id, "how many times can a loan be renewed", 3, mode
    )

    assert passages[0].document_path == "loans.txt"
    assert "renewed twice" in passages[0].text
    assert list(passages[0].ranks) == (["bm25", "dense"] if mode == "hybrid" else [mode])


def test_vector_cache_sees_documents_added_after_the_first_query(
    conn: sqlite3.Connection, embedder: HashingEmbedder, collection_id: str
) -> None:
    retriever = Retriever(embedder, VectorIndex())
    ingest_file(conn, embedder, collection_id, "loans.txt", fixture_bytes("loans.txt"))
    assert all(
        p.document_path == "loans.txt"
        for p in retriever.search(conn, collection_id, "x", 5, "dense")
    )

    ingest_file(conn, embedder, collection_id, "catalogue.html", fixture_bytes("catalogue.html"))
    paths = {
        p.document_path
        for p in retriever.search(conn, collection_id, "shelfmark folio", 5, "dense")
    }
    assert "catalogue.html" in paths


def test_search_is_scoped_to_one_collection(
    conn: sqlite3.Connection, embedder: HashingEmbedder, collection_id: str
) -> None:
    other = library.create_collection(conn, "Other shelf").id
    ingest_file(conn, embedder, other, "loans.txt", fixture_bytes("loans.txt"))
    retriever = Retriever(embedder, VectorIndex())
    assert retriever.search(conn, collection_id, "loan renewed", 5, "hybrid") == []
