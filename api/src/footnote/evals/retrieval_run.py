"""``footnote eval retrieval``: recall@k, MRR, and nDCG for each retrieval mode.

Runs offline: the corpus is ingested with the local embedding model into a temporary
database, every answerable golden question is searched in each configuration, and the
results are written as Markdown and JSON.
"""

from __future__ import annotations

import json
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from footnote.config import Settings
from footnote.evals.corpus import GOLDEN_PATH, EvalIndex, eval_index
from footnote.evals.golden import GoldenItem, Span, load_golden, resolve_spans
from footnote.evals.metrics import (
    RetrievedSpan,
    coverage,
    mean,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)
from footnote.evals.report import check_thresholds, retrieval_markdown
from footnote.ingest.chunker import DEFAULT_CHUNKING
from footnote.retrieval.embedder import FastEmbedder
from footnote.retrieval.search import Retriever
from footnote.retrieval.vectors import VectorIndex
from footnote.schemas import SearchMode

KS = (1, 3, 5, 10)
QUALITY_METRICS = (*(f"recall@{k}" for k in KS), "mrr@10", "ndcg@5", "ndcg@10")


@dataclass(frozen=True, slots=True)
class Variant:
    name: str
    mode: SearchMode
    query_instruction: bool = True
    note: str = ""


VARIANTS = (
    Variant("bm25", "bm25", note="SQLite FTS5 BM25 only"),
    Variant("dense", "dense", note="bge-small-en-v1.5 cosine only"),
    Variant("hybrid", "hybrid", note="BM25 + dense, reciprocal rank fusion (k=60)"),
    Variant(
        "dense-no-instruction",
        "dense",
        query_instruction=False,
        note="dense without the BGE query prefix",
    ),
    Variant(
        "hybrid-no-instruction",
        "hybrid",
        query_instruction=False,
        note="hybrid without the BGE query prefix",
    ),
)


def evaluate_variant(
    retriever: Retriever,
    index: EvalIndex,
    items: list[GoldenItem],
    spans: dict[str, list[Span]],
    mode: SearchMode,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    retriever.search(index.conn, index.collection_id, "warm up", 1, mode)  # loads the model once
    per_question: list[dict[str, Any]] = []
    latencies: list[float] = []
    for item in items:
        started = time.perf_counter()
        passages = retriever.search(index.conn, index.collection_id, item.question, max(KS), mode)
        latencies.append((time.perf_counter() - started) * 1000)
        ranked = [RetrievedSpan(p.document_path, p.start, p.end) for p in passages]
        expected = spans[item.id]
        hits = coverage(ranked, expected)
        row: dict[str, Any] = {"id": item.id, "tags": item.tags}
        for k in KS:
            row[f"recall@{k}"] = recall_at_k(hits, len(expected), k)
        row["mrr@10"] = reciprocal_rank(hits, 10)
        row["ndcg@5"] = ndcg_at_k(hits, len(expected), 5)
        row["ndcg@10"] = ndcg_at_k(hits, len(expected), 10)
        row["first_hit"] = next((rank for rank, hit in enumerate(hits, start=1) if hit), None)
        per_question.append(row)
    summary = {
        metric: round(mean([row[metric] for row in per_question]), 4) for metric in QUALITY_METRICS
    }
    summary["latency_ms_p50"] = round(statistics.median(latencies), 2)
    summary["latency_ms_p95"] = round(statistics.quantiles(latencies, n=20)[-1], 2)
    return summary, per_question


def run_retrieval_eval(
    settings: Settings,
    out_dir: Path,
    fail_under: list[str],
    baseline: Path | None,
    tolerance: float,
) -> int:
    items = [item for item in load_golden(GOLDEN_PATH) if item.answerable]
    embedders = {
        flag: FastEmbedder(
            settings.embedding_model, settings.model_cache_dir, query_instruction=flag
        )
        for flag in (True, False)
    }
    print(f"ingesting corpus with {settings.embedding_model} ...", file=sys.stderr)
    with eval_index(embedders[True]) as index:
        spans = resolve_spans(items, index.texts)
        modes: dict[str, dict[str, float]] = {}
        per_question: dict[str, list[dict[str, Any]]] = {}
        for variant in VARIANTS:
            retriever = Retriever(
                embedders[variant.query_instruction],
                VectorIndex(),
                settings.candidate_pool,
                settings.rrf_k,
            )
            modes[variant.name], per_question[variant.name] = evaluate_variant(
                retriever, index, items, spans, variant.mode
            )
            print(
                f"  {variant.name:<22} recall@5={modes[variant.name]['recall@5']:.3f} "
                f"mrr@10={modes[variant.name]['mrr@10']:.3f}",
                file=sys.stderr,
            )
        machine = f"{platform.system()} {platform.machine()}, Python {platform.python_version()}"
        report: dict[str, Any] = {
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "machine": machine,
            "corpus": {
                "documents": len(index.texts),
                "chunks": index.chunk_count,
                "ingest_seconds": round(index.ingest_seconds, 1),
            },
            "golden": {"answerable_questions": len(items), "path": "evals/golden.jsonl"},
            "settings": {
                "embedding_model": settings.embedding_model,
                "candidate_pool": settings.candidate_pool,
                "rrf_k": settings.rrf_k,
                "chunking": asdict(DEFAULT_CHUNKING),
            },
            "variants": {v.name: v.note for v in VARIANTS},
            "modes": modes,
            "per_question": per_question,
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "retrieval.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (out_dir / "retrieval.md").write_text(retrieval_markdown(report), encoding="utf-8")
    print(f"wrote {out_dir / 'retrieval.md'}", file=sys.stderr)

    failures = check_thresholds(modes, fail_under)
    if baseline is not None:
        failures += regressions(
            modes, json.loads(baseline.read_text(encoding="utf-8"))["modes"], tolerance
        )
    for failure in failures:
        print(f"FAIL {failure}", file=sys.stderr)
    return 1 if failures else 0


def regressions(
    current: dict[str, dict[str, float]], baseline: dict[str, dict[str, float]], tolerance: float
) -> list[str]:
    """Quality metrics that fell more than ``tolerance`` below the baseline. Latency is
    machine-dependent, so it is reported but never gated."""
    found: list[str] = []
    for mode, metrics in baseline.items():
        for metric in QUALITY_METRICS:
            if metric in metrics and mode in current and metric in current[mode]:
                drop = metrics[metric] - current[mode][metric]
                if drop > tolerance:
                    was, now = metrics[metric], current[mode][metric]
                    found.append(f"{mode}.{metric} fell from {was:.3f} to {now:.3f}")
    return found
