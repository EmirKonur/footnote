"""``footnote eval generation``: answer each golden question, then grade it with a judge.

This makes two API calls per question (one answer, one judgement), so it needs
credentials and costs money; it is not part of CI. Retrieval is the same hybrid search
the app uses.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from anthropic import AsyncAnthropic

from footnote.answer.anthropic_stream import AnthropicStreamer, has_credentials
from footnote.answer.generate import GenerationRequest, Streamer, collect_answer
from footnote.config import Settings
from footnote.evals.corpus import GOLDEN_PATH, eval_index
from footnote.evals.golden import GoldenItem, load_golden
from footnote.evals.judge import Judge, JudgeFailure, Verdict, to_judgements
from footnote.evals.metrics import CORRECTNESS_POINTS, mean_defined, score_answer
from footnote.evals.report import check_thresholds, generation_markdown
from footnote.retrieval.embedder import FastEmbedder
from footnote.retrieval.search import Retriever
from footnote.retrieval.vectors import VectorIndex
from footnote.schemas import Answer, Passage

JudgeFn = Callable[[GoldenItem, Answer], Awaitable[Verdict | JudgeFailure]]


async def evaluate_items(
    items: Sequence[GoldenItem],
    retrieve: Callable[[str], list[Passage]],
    streamer: Streamer,
    judge: JudgeFn,
) -> list[dict[str, Any]]:
    """One row per question: generation status, the judge's labels, and scores."""
    rows: list[dict[str, Any]] = []
    for position, item in enumerate(items, start=1):
        answer = await collect_answer(
            streamer, GenerationRequest(item.question, retrieve(item.question))
        )
        row: dict[str, Any] = {
            "id": item.id,
            "answerable": item.answerable,
            "status": answer.status,
        }
        if answer.status != "complete":
            row["error"] = answer.error or answer.refusal_category or answer.status
            rows.append(row)
            continue
        verdict = await judge(item, answer)
        if isinstance(verdict, JudgeFailure):
            row["status"] = "judge_failed"
            row["error"] = verdict.reason
            rows.append(row)
            continue
        scores = score_answer(to_judgements(verdict, answer))
        row.update(
            faithfulness=scores.faithfulness,
            citation_support=scores.citation_support,
            citation_precision=scores.citation_precision,
            correctness=verdict.correctness,
            abstained=verdict.abstained,
            rationale=verdict.rationale,
        )
        rows.append(row)
        print(f"  [{position}/{len(items)}] {item.id}: {verdict.correctness}", file=sys.stderr)
    return rows


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, float | None]:
    """Averages over judged rows.

    Faithfulness and the citation scores only make sense for answers that answered, so
    they are averaged over answerable questions the model did not abstain on.
    Abstention is scored separately in both directions.
    """
    judged = [r for r in rows if "abstained" in r]
    answered = [r for r in judged if r["answerable"] and not r["abstained"]]
    unanswerable = [r for r in judged if not r["answerable"]]
    answerable = [r for r in judged if r["answerable"]]
    correctness = [
        CORRECTNESS_POINTS[r["correctness"]]
        for r in answerable
        if r["correctness"] in CORRECTNESS_POINTS
    ]
    return {
        "faithfulness": mean_defined([r["faithfulness"] for r in answered]),
        "citation_support": mean_defined([r["citation_support"] for r in answered]),
        "citation_precision": mean_defined([r["citation_precision"] for r in answered]),
        "answer_correctness": mean_defined(correctness),
        "abstention_accuracy": mean_defined([1.0 if r["abstained"] else 0.0 for r in unanswerable]),
        "false_abstention_rate": mean_defined([1.0 if r["abstained"] else 0.0 for r in answerable]),
    }


def _write_report(out_dir: Path, report: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "generation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (out_dir / "generation.md").write_text(generation_markdown(report), encoding="utf-8")
    print(f"wrote {out_dir / 'generation.md'}", file=sys.stderr)


async def run_generation_eval(
    settings: Settings, out_dir: Path, fail_under: list[str], limit: int | None
) -> int:
    client = AsyncAnthropic()
    if not has_credentials(client):
        print(
            "the generation eval calls the Anthropic API: set ANTHROPIC_API_KEY first",
            file=sys.stderr,
        )
        return 2
    items = load_golden(GOLDEN_PATH)[:limit] if limit else load_golden(GOLDEN_PATH)
    streamer = AnthropicStreamer(
        client, model=settings.model, effort=settings.effort, max_tokens=settings.max_tokens
    )
    judge = Judge(client, settings.judge_model)
    embedder = FastEmbedder(
        settings.embedding_model,
        settings.model_cache_dir,
        query_instruction=settings.query_instruction,
    )
    with eval_index(embedder) as index:
        retriever = Retriever(embedder, VectorIndex(), settings.candidate_pool, settings.rrf_k)

        def retrieve(question: str) -> list[Passage]:
            return retriever.search(
                index.conn, index.collection_id, question, settings.top_k, "hybrid"
            )

        rows = await evaluate_items(items, retrieve, streamer, judge.judge)

    summary = summarize(rows)
    report: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": settings.model,
        "effort": settings.effort,
        "judge_model": settings.judge_model,
        "questions": len(rows),
        "judged": sum(1 for r in rows if "abstained" in r),
        "failed": sum(1 for r in rows if "abstained" not in r),
        "summary": summary,
        "items": rows,
    }
    _write_report(out_dir, report)
    failures = check_thresholds(summary, fail_under)
    for failure in failures:
        print(f"FAIL {failure}", file=sys.stderr)
    return 1 if failures else 0
