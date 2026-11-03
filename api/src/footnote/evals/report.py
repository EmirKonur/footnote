"""Markdown rendering for eval reports, and threshold checks for CI."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def check_thresholds(results: Mapping[str, Any], specs: list[str]) -> list[str]:
    """Evaluate ``--fail-under`` specs such as ``hybrid.recall@5=0.85``.

    Keys are dotted paths into ``results``; a missing key is an error rather than a
    silent pass, so a typo in CI configuration cannot disable the gate.
    """
    failures: list[str] = []
    for spec in specs:
        key, sep, raw = spec.rpartition("=")
        if not sep or not key:
            raise ValueError(f"--fail-under expects METRIC=VALUE, got {spec!r}")
        floor = float(raw)
        value: Any = results
        for part in key.split("."):
            if not isinstance(value, Mapping) or part not in value:
                raise ValueError(f"--fail-under: no metric named {key!r}")
            value = value[part]  # pyright: ignore[reportUnknownVariableType]
        if value is None or float(value) < floor:  # pyright: ignore[reportUnknownArgumentType]
            failures.append(f"{key} = {value} is below {floor}")
    return failures


def _fmt(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def retrieval_markdown(report: dict[str, Any]) -> str:
    modes: dict[str, dict[str, float]] = report["modes"]
    corpus = report["corpus"]
    lines = [
        "# Retrieval eval",
        "",
        f"Generated {report['generated_at']} on {report['machine']}. "
        f"{corpus['documents']} documents, {corpus['chunks']} chunks "
        f"(ingested in {corpus['ingest_seconds']} s), "
        f"{report['golden']['answerable_questions']} answerable golden questions.",
        "",
        "| Mode | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR@10 | nDCG@5 | nDCG@10 "
        "| p50 ms |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, m in modes.items():
        lines.append(
            f"| {name} | {_fmt(m['recall@1'])} | {_fmt(m['recall@3'])} | {_fmt(m['recall@5'])} | "
            f"{_fmt(m['recall@10'])} | {_fmt(m['mrr@10'])} | {_fmt(m['ndcg@5'])} | "
            f"{_fmt(m['ndcg@10'])} | {_fmt(m['latency_ms_p50'], 1)} |"
        )

    per_question: dict[str, list[dict[str, Any]]] = report["per_question"]
    tags = sorted({tag for rows in per_question.values() for row in rows for tag in row["tags"]})
    lines += [
        "",
        "Recall@5 by question type:",
        "",
        "| Mode | " + " | ".join(tags) + " |",
        "| --- | " + " | ".join("---" for _ in tags) + " |",
    ]
    for name, rows in per_question.items():
        cells: list[str] = []
        for tag in tags:
            tagged = [row["recall@5"] for row in rows if tag in row["tags"]]
            cells.append(f"{sum(tagged) / len(tagged):.3f} (n={len(tagged)})" if tagged else "n/a")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")

    misses = [row for row in per_question.get("hybrid", []) if row["recall@5"] < 1.0]
    lines += ["", f"Hybrid misses at k=5 ({len(misses)}):", ""]
    if misses:
        lines += [
            f"- `{row['id']}`: recall@5 {row['recall@5']:.2f}, first hit at rank "
            f"{row['first_hit'] if row['first_hit'] else '>10'}"
            for row in misses
        ]
    else:
        lines.append("- none")
    lines += ["", "Variants:", ""] + [
        f"- `{name}`: {note}" for name, note in report["variants"].items()
    ]
    return "\n".join(lines) + "\n"


GENERATION_METRICS = (
    ("faithfulness", "Faithfulness"),
    ("citation_support", "Citation support"),
    ("citation_precision", "Citation precision"),
    ("answer_correctness", "Answer correctness"),
    ("abstention_accuracy", "Abstention (unanswerable)"),
    ("false_abstention_rate", "False abstention (answerable)"),
)


def generation_markdown(report: dict[str, Any]) -> str:
    summary: dict[str, float | None] = report["summary"]
    lines = [
        "# Generation eval",
        "",
        f"Generated {report['generated_at']}. Answers by `{report['model']}` (effort "
        f"`{report['effort']}`), judged by `{report['judge_model']}`. "
        f"{report['questions']} questions; {report['judged']} judged, "
        f"{report['failed']} failed to generate or judge.",
        "",
        "| Metric | Score |",
        "| --- | --- |",
    ]
    lines += [f"| {label} | {_fmt(summary.get(key))} |" for key, label in GENERATION_METRICS]
    lines += [
        "",
        "| Question | Status | Faithful | Cite support | Cite precision | Correct | Abstained |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["items"]:
        lines.append(
            f"| `{row['id']}` | {row['status']} | {_fmt(row.get('faithfulness'), 2)} | "
            f"{_fmt(row.get('citation_support'), 2)} | {_fmt(row.get('citation_precision'), 2)} | "
            f"{row.get('correctness') or 'n/a'} | {row.get('abstained', 'n/a')} |"
        )
    return "\n".join(lines) + "\n"
