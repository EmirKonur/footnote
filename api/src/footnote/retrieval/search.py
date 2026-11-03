"""BM25-only, dense-only, and hybrid retrieval behind one call."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from footnote.library import passages_for_chunks
from footnote.retrieval.bm25 import search_bm25
from footnote.retrieval.embedder import Embedder
from footnote.retrieval.fusion import DEFAULT_RRF_K, reciprocal_rank_fusion
from footnote.retrieval.vectors import VectorIndex
from footnote.schemas import Passage, SearchMode


@dataclass(slots=True)
class Retriever:
    embedder: Embedder
    index: VectorIndex
    candidate_pool: int = 50
    """How many candidates each retriever contributes before fusion."""
    rrf_k: int = DEFAULT_RRF_K

    def search(
        self,
        conn: sqlite3.Connection,
        collection_id: str,
        query: str,
        k: int,
        mode: SearchMode = "hybrid",
    ) -> list[Passage]:
        pool = max(k, self.candidate_pool)
        rankings: dict[str, list[int]] = {}
        raw_scores: dict[str, dict[int, float]] = {}
        if mode in ("bm25", "hybrid"):
            hits = search_bm25(conn, collection_id, query, pool)
            rankings["bm25"] = [chunk_id for chunk_id, _ in hits]
            raw_scores["bm25"] = dict(hits)
        if mode in ("dense", "hybrid"):
            vector = self.embedder.embed_query(query)
            hits = self.index.search(conn, collection_id, self.embedder.model_name, vector, pool)
            rankings["dense"] = [chunk_id for chunk_id, _ in hits]
            raw_scores["dense"] = dict(hits)

        if mode == "hybrid":
            fused = reciprocal_rank_fusion(rankings, k=self.rrf_k)[:k]
            ids = [f.id for f in fused]
            scores = [f.score for f in fused]
            ranks = [f.ranks for f in fused]
        else:
            ids = rankings[mode][:k]
            scores = [raw_scores[mode][i] for i in ids]
            ranks = [{mode: position} for position in range(1, len(ids) + 1)]
        return passages_for_chunks(conn, ids, scores, ranks)
