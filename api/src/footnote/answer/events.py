"""Events the answer stream sends to the browser, one SSE message each."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from footnote.schemas import Answer, Footnote, Model, Passage


class PassagesEvent(Model):
    event: Literal["passages"] = "passages"
    passages: list[Passage]


class TextEvent(Model):
    event: Literal["text"] = "text"
    segment: int
    text: str


class FootnoteEvent(Model):
    event: Literal["footnote"] = "footnote"
    segment: int
    footnote: Footnote


class NoticeEvent(Model):
    event: Literal["notice"] = "notice"
    message: str


class ErrorEvent(Model):
    event: Literal["error"] = "error"
    code: str
    message: str
    retryable: bool


class DoneEvent(Model):
    """Always the last event. ``answer`` is authoritative: a client that missed or
    mis-applied a delta can replace its state with it."""

    event: Literal["done"] = "done"
    answer: Answer
    message_id: str | None = None


AnswerEvent = Annotated[
    PassagesEvent | TextEvent | FootnoteEvent | NoticeEvent | ErrorEvent | DoneEvent,
    Field(discriminator="event"),
]


def to_sse(
    event: PassagesEvent | TextEvent | FootnoteEvent | NoticeEvent | ErrorEvent | DoneEvent,
) -> str:
    return f"event: {event.event}\ndata: {event.model_dump_json()}\n\n"
