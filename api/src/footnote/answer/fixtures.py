"""``FOOTNOTE_FAKE_LLM=fixtures``: deterministic answers without a model.

This is a development and test aid, not a model. It picks the retrieved sentences that
share the most words with the question and streams them back as cited text blocks,
using the same SDK event objects the real stream produces. That lets the UI, the
end-to-end test, and the screenshots run the full citation path (stream parsing,
footnote numbering, offset mapping, highlighting) with no API key. The UI labels every
answer produced this way.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass

from anthropic.types.beta import (
    BetaCitationCharLocation,
    BetaCitationsDelta,
    BetaFallbackBlock,
    BetaMessage,
    BetaMessageDeltaUsage,
    BetaRawContentBlockDeltaEvent,
    BetaRawContentBlockStartEvent,
    BetaRawMessageDeltaEvent,
    BetaRawMessageStartEvent,
    BetaRefusalStopDetails,
    BetaStopReason,
    BetaTextBlock,
    BetaTextDelta,
    BetaUsage,
)
from anthropic.types.beta.beta_fallback_info import BetaFallbackInfo
from anthropic.types.beta.beta_fallback_refusal_trigger import BetaFallbackRefusalTrigger
from anthropic.types.beta.beta_raw_message_delta_event import Delta

from footnote.answer.generate import GenerationRequest, StreamEvent
from footnote.answer.prompt import ABSTENTION_PREFIX
from footnote.ingest.chunker import sentence_spans
from footnote.ingest.document import BLOCK_SEPARATOR
from footnote.retrieval.bm25 import fts_query
from footnote.schemas import GenerationMode, Passage

FIXTURE_MODEL = "footnote-fixtures"

# Event builders. The fixture streamer and the test suite both use these so that fake
# streams are made of real SDK types rather than look-alike dicts.


def message_start(model: str) -> BetaRawMessageStartEvent:
    message = BetaMessage(
        id="msg_fixture",
        type="message",
        role="assistant",
        model=model,
        content=[],
        stop_reason=None,
        stop_sequence=None,
        usage=BetaUsage(input_tokens=0, output_tokens=0),
    )
    return BetaRawMessageStartEvent(type="message_start", message=message)


def text_start(index: int, text: str = "") -> BetaRawContentBlockStartEvent:
    block = BetaTextBlock(type="text", text=text, citations=None)
    return BetaRawContentBlockStartEvent(
        type="content_block_start", index=index, content_block=block
    )


def text_delta(index: int, text: str) -> BetaRawContentBlockDeltaEvent:
    return BetaRawContentBlockDeltaEvent(
        type="content_block_delta", index=index, delta=BetaTextDelta(type="text_delta", text=text)
    )


def citation_delta(
    index: int,
    document_index: int,
    passage: Passage,
    start: int,
    end: int,
    cited_text: str | None = None,
) -> BetaRawContentBlockDeltaEvent:
    citation = BetaCitationCharLocation(
        type="char_location",
        cited_text=passage.text[start:end] if cited_text is None else cited_text,
        document_index=document_index,
        document_title=passage.document_title,
        start_char_index=start,
        end_char_index=end,
    )
    return BetaRawContentBlockDeltaEvent(
        type="content_block_delta",
        index=index,
        delta=BetaCitationsDelta(type="citations_delta", citation=citation),
    )


def fallback_start(index: int, from_model: str, to_model: str) -> BetaRawContentBlockStartEvent:
    block = BetaFallbackBlock.model_validate(
        {
            "type": "fallback",
            "from": BetaFallbackInfo(model=from_model).model_dump(),
            "to": BetaFallbackInfo(model=to_model).model_dump(),
            "trigger": BetaFallbackRefusalTrigger(type="refusal", category="cyber").model_dump(),
        }
    )
    return BetaRawContentBlockStartEvent(
        type="content_block_start", index=index, content_block=block
    )


def message_delta(
    stop_reason: BetaStopReason, *, output_tokens: int = 0, refusal_category: str | None = None
) -> BetaRawMessageDeltaEvent:
    details = (
        BetaRefusalStopDetails.model_validate({"type": "refusal", "category": refusal_category})
        if stop_reason == "refusal"
        else None
    )
    return BetaRawMessageDeltaEvent(
        type="message_delta",
        delta=Delta(stop_reason=stop_reason, stop_details=details),
        usage=BetaMessageDeltaUsage(output_tokens=output_tokens),
    )


# The fixture answerer ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Pick:
    document_index: int
    start: int
    end: int


def _terms(text: str) -> set[str]:
    query = fts_query(text) or ""
    return {_stem(term.strip('"')) for term in query.split(" OR ") if len(term.strip('"')) > 1}


def _stem(word: str) -> str:
    # Plural "s" only. Stripping "es" or "ing" as well turned "lines" into "lin", which
    # then never matched "line".
    return word[:-1] if len(word) > 3 and word.endswith("s") and not word.endswith("ss") else word


def pick_sentences(question: str, passages: Sequence[Passage], limit: int = 2) -> list[Pick]:
    """The sentences that cover the most question terms, from the top five passages.

    A sentence qualifies only if it covers at least half of the question's content
    words (and at least two of them for questions longer than one word); otherwise the
    fixture abstains, which is what lets the UI's "not covered" state be exercised.
    Further sentences come only from the document of the best one, so a shared keyword
    can't pull an unrelated document into the answer, and a sentence repeated by two
    overlapping chunks is used once. Picks are returned in document order.
    """
    terms = _terms(question)
    if not terms:
        return []
    needed = max(min(2, len(terms)), (len(terms) + 1) // 2)
    scored: list[tuple[float, Pick]] = []
    for document_index, passage in enumerate(passages[:5]):
        for start, end in _sentences(passage.text):
            sentence = passage.text[start:end]
            if len(sentence) < 25 or "\n" in sentence:
                continue
            words = {_stem(w.lower()) for w in re.findall(r"\w+", sentence)}
            overlap = len(terms & words)
            if overlap >= needed:
                scored.append((overlap - 0.1 * document_index, Pick(document_index, start, end)))
    scored.sort(key=lambda item: -item[0])

    def span(pick: Pick) -> tuple[str, int, int]:
        passage = passages[pick.document_index]
        return passage.document_id, passage.start + pick.start, passage.start + pick.end

    picks: list[Pick] = []
    for _, pick in scored:
        document, start, end = span(pick)
        if picks and document != span(picks[0])[0]:
            continue
        if all(span(p)[2] <= start or span(p)[1] >= end for p in picks):
            picks.append(pick)
        if len(picks) == limit:
            break
    return sorted(picks, key=lambda p: span(p)[1])


def _sentences(text: str) -> list[tuple[int, int]]:
    """Sentence spans within each block of a chunk. Splitting the chunk as a whole would
    glue a heading (which has no full stop) onto the first sentence after it."""
    spans: list[tuple[int, int]] = []
    offset = 0
    for block in text.split(BLOCK_SEPARATOR):
        spans.extend((offset + s, offset + e) for s, e in sentence_spans(block))
        offset += len(block) + len(BLOCK_SEPARATOR)
    return spans


class FixtureStreamer:
    def __init__(self, *, word_delay: float = 0.012) -> None:
        self._word_delay = word_delay

    @property
    def mode(self) -> GenerationMode:
        return "fixtures"

    @property
    def model(self) -> str:
        return FIXTURE_MODEL

    @asynccontextmanager
    async def open(self, request: GenerationRequest) -> AsyncGenerator[AsyncIterator[StreamEvent]]:
        yield self._events(request)

    async def _events(self, request: GenerationRequest) -> AsyncIterator[StreamEvent]:
        yield message_start(FIXTURE_MODEL)
        picks = pick_sentences(request.question, request.passages)
        index = 0
        if not picks:
            yield text_start(index)
            message = (
                f"{ABSTENTION_PREFIX}: no retrieved sentence shares enough words with the "
                "question. (Fixture mode matches words; it does not call a model.)"
            )
            async for event in self._words(index, message):
                yield event
            yield message_delta("end_turn")
            return
        # Plain extractive prose: each sentence is quoted as written, one text block and one
        # footnote per sentence. The footnotes name the source, so there is no lead-in.
        for number, pick in enumerate(picks):
            passage = request.passages[pick.document_index]
            sentence = passage.text[pick.start : pick.end]
            yield text_start(index)
            async for event in self._words(index, sentence if number == 0 else f" {sentence}"):
                yield event
            yield citation_delta(index, pick.document_index, passage, pick.start, pick.end)
            index += 1
        yield message_delta("end_turn")

    async def _words(self, index: int, text: str) -> AsyncIterator[StreamEvent]:
        for piece in re.findall(r"\S+\s*|\s+", text):
            if self._word_delay:
                await asyncio.sleep(self._word_delay)
            yield text_delta(index, piece)
