"""Build a throwaway index of the bundled corpus for an eval run."""

from __future__ import annotations

import sqlite3
import tempfile
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from footnote import library
from footnote.db import connect, migrate
from footnote.ingest.parsers import SUPPORTED_SUFFIXES
from footnote.ingest.pipeline import ingest_file
from footnote.retrieval.embedder import Embedder

REPO_ROOT = Path(__file__).resolve().parents[4]
EVALS_DIR = REPO_ROOT / "evals"
CORPUS_DIR = EVALS_DIR / "corpus" / "peps"
GOLDEN_PATH = EVALS_DIR / "golden.jsonl"


@dataclass(frozen=True, slots=True)
class EvalIndex:
    conn: sqlite3.Connection
    collection_id: str
    texts: dict[str, str]
    """Canonical text of each document, keyed by its path in the corpus."""
    chunk_count: int
    ingest_seconds: float


@contextmanager
def eval_index(embedder: Embedder, corpus: Path = CORPUS_DIR) -> Generator[EvalIndex]:
    """Ingest ``corpus`` into a temporary database; nothing outlives the block."""
    with tempfile.TemporaryDirectory(prefix="footnote-eval-") as tmp:
        conn = connect(Path(tmp) / "eval.sqlite3")
        try:
            migrate(conn)
            collection = library.create_collection(conn, "eval")
            files = sorted(p for p in corpus.iterdir() if p.suffix.lower() in SUPPORTED_SUFFIXES)
            started = time.perf_counter()
            chunks = 0
            for path in files:
                chunks += ingest_file(
                    conn, embedder, collection.id, path.name, path.read_bytes()
                ).chunks
            elapsed = time.perf_counter() - started
            texts = {
                doc.path: library.get_document(conn, doc.id).text
                for doc in library.list_documents(conn, collection.id)
            }
            yield EvalIndex(conn, collection.id, texts, chunks, elapsed)
        finally:
            conn.close()
