"""Reciprocal rank fusion (Cormack, Clarke and Buettcher, SIGIR 2009)."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

DEFAULT_RRF_K = 60


@dataclass(frozen=True, slots=True)
class FusedResult:
    id: int
    score: float
    ranks: dict[str, int] = field(default_factory=dict[str, int])
    """1-based rank of this id in each input ranking it appeared in."""


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[int]], k: int = DEFAULT_RRF_K
) -> list[FusedResult]:
    """Fuse ranked id lists: score(d) = sum over rankings of 1 / (k + rank(d)).

    RRF only looks at ranks, which is the point: BM25 scores and cosine similarities
    live on unrelated scales, and normalising them is fragile. ``k`` damps the
    advantage of the very top ranks; 60 is the value from the original paper and is
    what Footnote uses unless the eval suite says otherwise.

    Ties are broken by the best rank the id reached in any input, then by id, so the
    output is deterministic. Scores are summed with ``math.fsum``, which rounds exactly
    once: plain ``+=`` made the result depend on the order of the input rankings in the
    last bit, enough to flip two near-tied results.
    """
    if k < 0:
        raise ValueError("k must be non-negative")
    contributions: dict[int, list[float]] = {}
    ranks: dict[int, dict[str, int]] = {}
    for name, ranking in rankings.items():
        seen: set[int] = set()
        for position, item in enumerate(ranking, start=1):
            if item in seen:
                continue
            seen.add(item)
            contributions.setdefault(item, []).append(1.0 / (k + position))
            ranks.setdefault(item, {})[name] = position
    scores = {item: math.fsum(values) for item, values in contributions.items()}
    ordered = sorted(scores, key=lambda item: (-scores[item], min(ranks[item].values()), item))
    return [FusedResult(item, scores[item], ranks[item]) for item in ordered]
