"""The golden question set, and resolving its expected passages against the corpus.

Expected passages are stored as verbatim quotes, not chunk ids. Relevance is decided by
character overlap between a retrieved chunk and the quote's location in the canonical
document, so the golden set survives any change to the chunker. A quote that cannot be
found fails loudly: a parser change that silently altered the canonical text would
otherwise quietly turn questions into misses.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class ExpectedPassage(BaseModel):
    model_config = ConfigDict(frozen=True)

    doc: str
    quote: str


class GoldenItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    question: str
    answerable: bool
    expected: list[ExpectedPassage]
    reference_answer: str | None
    tags: list[str]


@dataclass(frozen=True, slots=True)
class Span:
    """Where an expected passage sits in its document's canonical text."""

    doc: str
    start: int
    end: int


def load_golden(path: Path) -> list[GoldenItem]:
    items = [
        GoldenItem.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = [item.id for item in items]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise ValueError(f"duplicate golden ids: {sorted(duplicates)}")
    for item in items:
        if item.answerable != bool(item.expected):
            raise ValueError(f"{item.id}: answerable items need expected passages, others none")
    return items


def locate_quote(text: str, quote: str) -> tuple[int, int] | None:
    """Find a quote in canonical text, tolerating differences in whitespace."""
    words = quote.split()
    if not words:
        return None
    pattern = r"\s+".join(re.escape(word) for word in words)
    match = re.search(pattern, text)
    return (match.start(), match.end()) if match else None


def resolve_spans(items: list[GoldenItem], texts: dict[str, str]) -> dict[str, list[Span]]:
    """Map each item id to the spans of its expected passages; raise on any miss."""
    resolved: dict[str, list[Span]] = {}
    missing: list[str] = []
    for item in items:
        spans: list[Span] = []
        for expected in item.expected:
            text = texts.get(expected.doc)
            found = locate_quote(text, expected.quote) if text is not None else None
            if found is None:
                missing.append(f"{item.id}: {expected.doc}: {expected.quote[:60]!r}")
                continue
            spans.append(Span(expected.doc, *found))
        resolved[item.id] = spans
    if missing:
        raise ValueError("golden quotes not found in the corpus:\n  " + "\n  ".join(missing))
    return resolved
