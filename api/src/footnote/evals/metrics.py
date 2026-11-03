"""Retrieval and generation metrics.

Retrieval metrics are passage-level. A query has expected passages E (|E| = m) and a
ranked list of retrieved chunks. A chunk *covers* an expected passage when it overlaps
at least half of the passage's characters. Then:

* recall@k: the fraction of E covered by at least one of the top k chunks;
* reciprocal rank: 1 / (rank of the first chunk that covers anything), 0 if none;
* nDCG@k: a chunk earns gain 1 if it covers an expected passage no earlier chunk
  covered, else 0, discounted by log2(rank + 1); the ideal ranking earns gain 1 at
  each of the first min(k, m) ranks.

Counting only newly covered passages matters because chunks overlap: two adjacent
chunks that both contain the answer must not score as two relevant results.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from footnote.evals.golden import Span

COVER_FRACTION = 0.5


@dataclass(frozen=True, slots=True)
class RetrievedSpan:
    doc: str
    start: int
    end: int


def covers(chunk: RetrievedSpan, expected: Span) -> bool:
    if chunk.doc != expected.doc:
        return False
    overlap = min(chunk.end, expected.end) - max(chunk.start, expected.start)
    return overlap >= COVER_FRACTION * (expected.end - expected.start)


def coverage(ranked: Sequence[RetrievedSpan], expected: Sequence[Span]) -> list[set[int]]:
    """For each rank, the indices of expected passages that chunk covers."""
    return [{i for i, e in enumerate(expected) if covers(chunk, e)} for chunk in ranked]


def recall_at_k(hits: Sequence[set[int]], expected_count: int, k: int) -> float:
    if expected_count == 0:
        raise ValueError("recall is undefined without expected passages")
    found: set[int] = set()
    for hit in hits[:k]:
        found |= hit
    return len(found) / expected_count


def reciprocal_rank(hits: Sequence[set[int]], k: int) -> float:
    for rank, hit in enumerate(hits[:k], start=1):
        if hit:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(hits: Sequence[set[int]], expected_count: int, k: int) -> float:
    if expected_count == 0:
        raise ValueError("nDCG is undefined without expected passages")
    seen: set[int] = set()
    dcg = 0.0
    for rank, hit in enumerate(hits[:k], start=1):
        if hit - seen:
            dcg += 1.0 / math.log2(rank + 1)
        seen |= hit
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(k, expected_count) + 1))
    return dcg / ideal


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


# Generation -------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SegmentJudgement:
    is_claim: bool
    supported: bool
    citations: dict[int, bool]
    """Footnote number -> does that footnote's cited text support this segment?"""


@dataclass(frozen=True, slots=True)
class AnswerScores:
    faithfulness: float | None
    citation_support: float | None
    citation_precision: float | None


def score_answer(segments: Sequence[SegmentJudgement]) -> AnswerScores:
    """Per-answer scores from per-segment judgements.

    * faithfulness: supported claims / claims;
    * citation support: claims with at least one supporting citation / claims;
    * citation precision: supporting citations / citations.

    None means "not applicable" (an answer with no claims, or no citations), which is
    different from 0 and is left out of the averages.
    """
    claims = [s for s in segments if s.is_claim]
    citations = [ok for s in segments for ok in s.citations.values()]
    return AnswerScores(
        faithfulness=(sum(s.supported for s in claims) / len(claims)) if claims else None,
        citation_support=(
            sum(any(s.citations.values()) for s in claims) / len(claims) if claims else None
        ),
        citation_precision=(sum(citations) / len(citations)) if citations else None,
    )


CORRECTNESS_POINTS = {"correct": 1.0, "partially_correct": 0.5, "incorrect": 0.0}


def mean_defined(values: Sequence[float | None]) -> float | None:
    defined = [v for v in values if v is not None]
    return mean(defined) if defined else None
