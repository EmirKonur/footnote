"""An LLM judge for generated answers.

The judge sees the question, the reference answer, the passages the answering model was
given, and the answer split into the same segments the UI renders, each with its
footnotes. It labels every segment, and every footnote on it, so the scores in
``metrics.score_answer`` are computed by Footnote rather than estimated by the model.

The judge call uses structured outputs. That is fine here: the restriction is that
citations and structured outputs cannot share a request, and the judge request has no
citation-enabled documents.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import anthropic
from anthropic import AsyncAnthropic
from pydantic import BaseModel, Field

from footnote.answer.anthropic_stream import FALLBACK_BETA
from footnote.answer.errors import describe_failure
from footnote.evals.golden import GoldenItem
from footnote.evals.metrics import SegmentJudgement
from footnote.schemas import Answer

JUDGE_SYSTEM = """\
You grade answers produced by a retrieval-augmented question answering system. You are \
strict and literal: a claim is supported only if the passage text states it or directly \
entails it, not if it is merely plausible or true in general.

You receive the question, an optional reference answer, the passages the system was \
given, and the answer split into numbered segments. Some segments carry footnotes; each \
footnote quotes the passage text the system cited for that segment.

For every segment, decide:
- is_claim: true if the segment asserts something checkable. Connective text ("According \
to the guide,"), restatements of the question, and statements that the documents do not \
cover something are not claims.
- supported_by_passages: for claims, true if any passage supports it. For non-claims, true.
- citations: for each footnote on the segment, whether that footnote's quoted text by \
itself supports the segment's claim.

Then decide for the whole answer:
- abstained: true if the answer declines to answer, or says the documents do not contain \
the answer, instead of answering.
- correctness: compare with the reference answer. "correct" if it conveys the reference's \
key facts without contradicting them, "partially_correct" if it gets some key facts or \
adds a wrong one alongside right ones, "incorrect" otherwise. Use "not_applicable" when \
there is no reference answer.
- rationale: one or two sentences explaining the correctness label.\
"""


class CitationVerdict(BaseModel):
    footnote: int
    supports: bool


class SegmentVerdict(BaseModel):
    segment: int
    is_claim: bool
    supported_by_passages: bool
    citations: list[CitationVerdict] = Field(default_factory=list[CitationVerdict])


class Verdict(BaseModel):
    segments: list[SegmentVerdict]
    abstained: bool
    correctness: Literal["correct", "partially_correct", "incorrect", "not_applicable"]
    rationale: str


@dataclass(frozen=True, slots=True)
class JudgeFailure:
    reason: str


def render_case(item: GoldenItem, answer: Answer) -> str:
    passages = "\n\n".join(
        f'<passage index="{i}" title="{p.document_title}">\n{p.text}\n</passage>'
        for i, p in enumerate(answer.passages, start=1)
    )
    segments = "\n".join(
        f'<segment index="{i}" footnotes="{_numbers(s.footnotes)}">{s.text}</segment>'
        for i, s in enumerate(answer.segments)
    )
    footnotes = "\n".join(
        f'<footnote n="{f.n}" document="{f.document_title}">{f.cited_text}</footnote>'
        for f in answer.footnotes
    )
    reference = item.reference_answer or "(none: there is no reference answer for this question)"
    return (
        f"<question>{item.question}</question>\n"
        f"<reference_answer>{reference}</reference_answer>\n"
        f"<passages>\n{passages}\n</passages>\n"
        f"<answer_segments>\n{segments}\n</answer_segments>\n"
        f"<footnotes>\n{footnotes or '(none)'}\n</footnotes>"
    )


def _numbers(footnotes: list[int]) -> str:
    return ",".join(str(n) for n in footnotes) or "none"


def to_judgements(verdict: Verdict, answer: Answer) -> list[SegmentJudgement]:
    """Align the verdict with the answer's own segments and footnotes. A segment the
    judge skipped counts as an unsupported claim, and a footnote it skipped as not
    supporting, so omissions can only lower the score."""
    by_index = {s.segment: s for s in verdict.segments}
    judgements: list[SegmentJudgement] = []
    for index, segment in enumerate(answer.segments):
        found = by_index.get(index)
        if found is None:
            judgements.append(
                SegmentJudgement(True, False, dict.fromkeys(segment.footnotes, False))
            )
            continue
        cited = {c.footnote: c.supports for c in found.citations}
        judgements.append(
            SegmentJudgement(
                is_claim=found.is_claim,
                supported=found.supported_by_passages,
                citations={n: cited.get(n, False) for n in segment.footnotes},
            )
        )
    return judgements


class Judge:
    def __init__(self, client: AsyncAnthropic, model: str, *, max_tokens: int = 16000) -> None:
        self._client = client
        self._model = model
        self._max_tokens = max_tokens

    @property
    def model(self) -> str:
        return self._model

    async def judge(self, item: GoldenItem, answer: Answer) -> Verdict | JudgeFailure:
        try:
            response = await self._client.beta.messages.parse(
                model=self._model,
                max_tokens=self._max_tokens,
                system=JUDGE_SYSTEM,
                messages=[{"role": "user", "content": render_case(item, answer)}],
                output_format=Verdict,
                thinking={"type": "adaptive"},
                output_config={"effort": "medium"},
                betas=[FALLBACK_BETA],
                fallbacks="default",
            )
        except anthropic.AnthropicError as exc:
            return JudgeFailure(describe_failure(exc, self._model).message)
        if response.stop_reason == "refusal":
            return JudgeFailure("the judge model declined to grade this case")
        if response.stop_reason == "max_tokens":
            return JudgeFailure("the judge ran out of output tokens")
        if response.parsed_output is None:
            return JudgeFailure("the judge returned no parseable verdict")
        return response.parsed_output
