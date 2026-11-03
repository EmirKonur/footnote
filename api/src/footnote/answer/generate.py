"""Stream an answer: SDK stream events in, browser events out.

``AnswerRun`` consumes raw Messages API stream events from a ``Streamer`` (the real
SDK stream, or the fixture stream) and builds the answer as it goes. Each ``text``
content block becomes one segment; ``citations_delta`` events attach footnotes to the
segment they arrived on. The same code path runs in every mode, so the fixture mode and
the tests exercise exactly what production does.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from typing import Protocol

import anthropic
from anthropic.types.beta import (
    BetaRawContentBlockDeltaEvent,
    BetaRawContentBlockStartEvent,
    BetaRawMessageDeltaEvent,
    BetaRawMessageStartEvent,
)

from footnote.answer.citations import FootnoteBook
from footnote.answer.errors import describe_failure
from footnote.answer.events import (
    DoneEvent,
    ErrorEvent,
    FootnoteEvent,
    NoticeEvent,
    PassagesEvent,
    TextEvent,
)
from footnote.answer.prompt import HistoryTurn
from footnote.log import get_logger
from footnote.schemas import Answer, AnswerStatus, GenerationMode, Passage, Segment

log = get_logger(__name__)

StreamEvent = (
    BetaRawMessageStartEvent
    | BetaRawMessageDeltaEvent
    | BetaRawContentBlockStartEvent
    | BetaRawContentBlockDeltaEvent
)
OutEvent = PassagesEvent | TextEvent | FootnoteEvent | NoticeEvent | ErrorEvent | DoneEvent


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    question: str
    passages: list[Passage]
    history: list[HistoryTurn] = field(default_factory=list[HistoryTurn])


class Streamer(Protocol):
    """Opens one model response as an async iterator of raw stream events."""

    @property
    def mode(self) -> GenerationMode: ...

    @property
    def model(self) -> str: ...

    def open(
        self, request: GenerationRequest
    ) -> AbstractAsyncContextManager[AsyncIterator[StreamEvent]]: ...


@dataclass(slots=True)
class _Builder:
    passages: list[Passage]
    mode: GenerationMode
    model: str
    book: FootnoteBook
    segments: list[Segment] = field(default_factory=list[Segment])
    block_segments: dict[int, int] = field(default_factory=dict[int, int])
    status: AnswerStatus = "complete"
    stop_reason: str | None = None
    refusal_category: str | None = None
    notices: list[str] = field(default_factory=list[str])
    error: str | None = None

    def apply(self, event: StreamEvent) -> list[OutEvent]:
        if isinstance(event, BetaRawMessageStartEvent):
            return self._on_start(event)
        if isinstance(event, BetaRawContentBlockStartEvent):
            return self._on_block_start(event)
        if isinstance(event, BetaRawContentBlockDeltaEvent):
            return self._on_delta(event)
        return self._on_message_delta(event)

    def _on_start(self, event: BetaRawMessageStartEvent) -> list[OutEvent]:
        served_by = str(event.message.model)
        if served_by and served_by != self.model:
            # Sticky fallback routing: the API sends follow-ups straight to the model
            # that rescued an earlier refusal, so message_start already names it.
            self.model = served_by
            return [
                self._notice(f"Answered by {served_by}, the fallback model for this conversation.")
            ]
        return []

    def _on_block_start(self, event: BetaRawContentBlockStartEvent) -> list[OutEvent]:
        block = event.content_block
        if block.type == "text":
            self.block_segments[event.index] = len(self.segments)
            self.segments.append(Segment(text=block.text))
            return (
                [TextEvent(segment=len(self.segments) - 1, text=block.text)] if block.text else []
            )
        if block.type == "fallback":
            self.model = str(block.to.model)
            return [
                self._notice(
                    f"{block.from_.model} declined part of this request; "
                    f"{block.to.model} continued the answer."
                )
            ]
        return []

    def _on_delta(self, event: BetaRawContentBlockDeltaEvent) -> list[OutEvent]:
        index = self.block_segments.get(event.index)
        if index is None:
            return []  # deltas for thinking blocks and other non-text content
        delta = event.delta
        segment = self.segments[index]
        if delta.type == "text_delta":
            self.segments[index] = segment.model_copy(update={"text": segment.text + delta.text})
            return [TextEvent(segment=index, text=delta.text)]
        if delta.type == "citations_delta":
            footnote = self.book.resolve(delta.citation)
            if footnote is None:
                return []
            if footnote.n not in segment.footnotes:
                self.segments[index] = segment.model_copy(
                    update={"footnotes": [*segment.footnotes, footnote.n]}
                )
            return [FootnoteEvent(segment=index, footnote=footnote)]
        return []

    def _on_message_delta(self, event: BetaRawMessageDeltaEvent) -> list[OutEvent]:
        reason = event.delta.stop_reason
        if reason is None:
            return []
        self.stop_reason = reason
        usage = event.usage
        log.info(
            "answer finished",
            stop_reason=reason,
            model=self.model,
            output_tokens=usage.output_tokens,
            cache_read=usage.cache_read_input_tokens,
            cache_write=usage.cache_creation_input_tokens,
        )
        if reason == "refusal":
            details = event.delta.stop_details
            self.refusal_category = details.category if details else None
            self.status = "refused"
            # A refusal can arrive mid-stream; the partial text is not an answer.
            self.segments = []
            return []
        if reason in ("max_tokens", "model_context_window_exceeded"):
            self.status = "truncated"
            return [self._notice("The answer hit the output limit and was cut off.")]
        return []

    def _notice(self, message: str) -> NoticeEvent:
        self.notices.append(message)
        return NoticeEvent(message=message)

    def answer(self) -> Answer:
        return Answer(
            status=self.status,
            segments=[s for s in self.segments if s.text],
            footnotes=self.book.footnotes,
            passages=self.passages,
            model=self.model,
            stop_reason=self.stop_reason,
            refusal_category=self.refusal_category,
            notices=self.notices,
            error=self.error,
            generated_by=self.mode,
        )


class AnswerRun:
    """One answer in progress. Iterate it for events; call ``partial()`` if the client
    goes away before the stream finishes."""

    def __init__(self, streamer: Streamer, request: GenerationRequest) -> None:
        self._streamer = streamer
        self._request = request
        self._builder = _Builder(
            passages=request.passages,
            mode=streamer.mode,
            model=streamer.model,
            book=FootnoteBook(request.passages),
        )

    def partial(self) -> Answer:
        self._builder.status = "stopped"
        return self._builder.answer()

    async def events(self) -> AsyncGenerator[OutEvent]:
        builder = self._builder
        yield PassagesEvent(passages=self._request.passages)
        if not self._request.passages:
            builder.segments = [
                Segment(text="The documents don't cover this: the collection has no documents yet.")
            ]
            yield TextEvent(segment=0, text=builder.segments[0].text)
            yield DoneEvent(answer=builder.answer())
            return
        try:
            async with self._streamer.open(self._request) as stream:
                async for event in stream:
                    for out in builder.apply(event):
                        yield out
        except anthropic.AnthropicError as exc:
            failure = describe_failure(exc, builder.model)
            log.warning("generation failed", code=failure.code, error=str(exc))
            builder.status = "error"
            builder.error = failure.message
            yield ErrorEvent(
                code=failure.code, message=failure.message, retryable=failure.retryable
            )
        yield DoneEvent(answer=builder.answer())


async def collect_answer(streamer: Streamer, request: GenerationRequest) -> Answer:
    """Run a stream to completion and return the final answer (used by the eval suite)."""
    answer: Answer | None = None
    async for event in AnswerRun(streamer, request).events():
        if isinstance(event, DoneEvent):
            answer = event.answer
    assert answer is not None, "AnswerRun always ends with a DoneEvent"
    return answer
