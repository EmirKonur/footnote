"""Exact nearest-neighbour search over a collection's chunk vectors.

Vectors live in SQLite next to their chunks; this class keeps one float32 matrix per
collection in memory and answers queries with a single matrix-vector product. See
docs/adr/0002-vector-storage.md for why this beats an ANN index at Footnote's scale.
"""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from footnote.retrieval.embedder import Matrix, Vector


def to_blob(vector: Vector) -> bytes:
    return np.asarray(vector, dtype="<f4").tobytes()


def from_blob(blob: bytes, dim: int) -> Vector:
    vector: Vector = np.frombuffer(blob, dtype="<f4", count=dim).astype(np.float32)
    return vector


@dataclass(frozen=True, slots=True)
class _CollectionMatrix:
    revision: int
    chunk_ids: NDArray[np.int64]
    vectors: Matrix


class VectorIndex:
    def __init__(self) -> None:
        self._matrices: dict[tuple[str, str], _CollectionMatrix] = {}
        self._lock = threading.Lock()

    def search(
        self, conn: sqlite3.Connection, collection_id: str, model: str, query: Vector, limit: int
    ) -> list[tuple[int, float]]:
        """Return up to ``limit`` (chunk_id, cosine similarity) pairs, best first."""
        matrix = self._matrix(conn, collection_id, model)
        if limit <= 0 or len(matrix.chunk_ids) == 0:
            return []
        scores = matrix.vectors @ query
        if len(scores) > limit:
            candidates = np.argpartition(-scores, limit - 1)[:limit]
        else:
            candidates = np.arange(len(scores))
        # argpartition leaves the top slice unordered; sort it, breaking ties by chunk id
        # so results are stable across runs.
        ordered = sorted(
            candidates.tolist(), key=lambda i: (-float(scores[i]), int(matrix.chunk_ids[i]))
        )
        return [(int(matrix.chunk_ids[i]), float(scores[i])) for i in ordered]

    def _matrix(
        self, conn: sqlite3.Connection, collection_id: str, model: str
    ) -> _CollectionMatrix:
        row = conn.execute(
            "SELECT revision FROM collections WHERE id = ?", (collection_id,)
        ).fetchone()
        revision = int(row["revision"]) if row else -1
        key = (collection_id, model)
        with self._lock:
            cached = self._matrices.get(key)
            if cached is not None and cached.revision == revision:
                return cached
        rows = conn.execute(
            """
            SELECT c.id, e.dim, e.vector
            FROM chunks c JOIN embeddings e ON e.embed_hash = c.embed_hash AND e.model = ?
            WHERE c.collection_id = ?
            ORDER BY c.id
            """,
            (model, collection_id),
        ).fetchall()
        if rows:
            ids = np.array([r["id"] for r in rows], dtype=np.int64)
            vectors = np.stack([from_blob(r["vector"], r["dim"]) for r in rows])
        else:
            ids = np.zeros(0, dtype=np.int64)
            vectors = np.zeros((0, 0), dtype=np.float32)
        loaded = _CollectionMatrix(revision, ids, vectors)
        with self._lock:
            self._matrices[key] = loaded
        return loaded

    def forget(self, collection_id: str) -> None:
        with self._lock:
            for key in [k for k in self._matrices if k[0] == collection_id]:
                del self._matrices[key]
