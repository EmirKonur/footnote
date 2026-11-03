"""Time Footnote's exact vector search at several collection sizes.

    uv run python scripts/bench_vectors.py

Builds a throwaway database with N random 384-dimensional unit vectors (the size of
bge-small-en-v1.5), then measures two things through the real VectorIndex: the cold
load of the matrix from SQLite (paid once per collection revision) and the warm query
(paid on every search). Numbers feed docs/adr/0002-vector-storage.md.
"""

from __future__ import annotations

import platform
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from footnote.db import connect, migrate, transaction
from footnote.library import create_collection
from footnote.retrieval.embedder import normalize_rows
from footnote.retrieval.vectors import VectorIndex, to_blob

DIM = 384
SIZES = (1_000, 10_000, 100_000)
QUERIES = 200
POOL = 50


def build(path: Path, size: int) -> str:
    conn = connect(path)
    migrate(conn)
    collection = create_collection(conn, "bench").id
    rng = np.random.default_rng(7)
    vectors = normalize_rows(rng.standard_normal((size, DIM), dtype=np.float32))
    with transaction(conn):
        conn.execute(
            "INSERT INTO documents (id, collection_id, path, title, format, content_hash,"
            " size_bytes, text, blocks_json, ingested_at)"
            " VALUES ('d', ?, 'p', 't', 'text', 'h', 0, '', '[]', '')",
            (collection,),
        )
        conn.executemany(
            "INSERT INTO chunks (document_id, collection_id, ordinal, start_char, end_char,"
            " heading_path, text, embed_hash) VALUES ('d', ?, ?, 0, 0, '[]', '', ?)",
            [(collection, i, f"h{i}") for i in range(size)],
        )
        conn.executemany(
            "INSERT INTO embeddings (embed_hash, model, dim, vector) VALUES (?, 'bench', ?, ?)",
            [(f"h{i}", DIM, to_blob(vectors[i])) for i in range(size)],
        )
    conn.close()
    return collection


def main() -> int:
    print(
        f"{platform.system()} {platform.machine()}, Python {platform.python_version()}, "
        f"numpy {np.__version__}; {QUERIES} queries, top {POOL}"
    )
    columns = ("chunks", "cold load ms", "query p50 ms", "query p95 ms", "matrix MB")
    print("  ".join(f"{c:>12}" for c in columns))
    rng = np.random.default_rng(11)
    for size in SIZES:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bench.sqlite3"
            collection = build(path, size)
            conn = connect(path)
            index = VectorIndex()
            queries = normalize_rows(rng.standard_normal((QUERIES, DIM), dtype=np.float32))
            started = time.perf_counter()
            index.search(conn, collection, "bench", queries[0], POOL)
            cold = (time.perf_counter() - started) * 1000
            timings: list[float] = []
            for query in queries:
                started = time.perf_counter()
                index.search(conn, collection, "bench", query, POOL)
                timings.append((time.perf_counter() - started) * 1000)
            conn.close()
        p95 = statistics.quantiles(timings, n=20)[-1]
        megabytes = size * DIM * 4 / 1e6
        row = (
            f"{size}",
            f"{cold:.1f}",
            f"{statistics.median(timings):.2f}",
            f"{p95:.2f}",
            f"{megabytes:.1f}",
        )
        print("  ".join(f"{cell:>12}" for cell in row))
    return 0


if __name__ == "__main__":
    sys.exit(main())
