"""The request Footnote sends to the Messages API, checked against a recording fake.

No network: the fake stands in for ``AsyncAnthropic`` and records the keyword
arguments of ``beta.messages.stream``. These tests pin the parts of the contract that
fail loudly (a 400) or silently (no citations, no cache hits) when they drift.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any, cast

from anthropic import AsyncAnthropic
from anthropic.lib.streaming import ParsedBetaTextEvent

from footnote.answer.anthropic_stream import FALLBACK_BETA, AnthropicStreamer, has_credentials
from footnote.answer.fixtures import (
    citation_delta,
    message_delta,
    message_start,
    text_delta,
    text_start,
)
from footnote.answer.generate import GenerationRequest, collect_answer
from footnote.answer.prompt import SYSTEM_PROMPT, HistoryTurn, build_messages, document_block
from footnote.schemas import Passage

PASSAGE = Passage(
    chunk_id=7,
    document_id="doc_8",
    document_title="PEP 8: Style Guide for Python Code",
    document_path="pep-0008.html",
    heading_path=["PEP 8", "Code Lay-out", "Maximum Line Length"],
    page_start=None,
    page_end=None,
    start=5000,
    end=5044,
    text="Limit all lines to a maximum of 79 characters.",
    score=0.03,
)


class _FakeStream:
    def __init__(self, events: list[object]) -> None:
        self._events = events

    async def __aenter__(self) -> _FakeStream:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        return None

    def __aiter__(self) -> AsyncIterator[object]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[object]:
        for event in self._events:
            yield event


class RecordingClient:
    def __init__(self, events: list[object]) -> None:
        self.calls: list[dict[str, Any]] = []
        self._events = events
        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=self._stream))

    def _stream(self, **kwargs: Any) -> _FakeStream:
        self.calls.append(kwargs)
        return _FakeStream(self._events)


def streamer_for(client: RecordingClient) -> AnthropicStreamer:
    return AnthropicStreamer(
        cast(AsyncAnthropic, client), model="claude-opus-5-5", effort="medium", max_tokens=16000
    )


async def test_request_uses_citations_caching_fallbacks_and_effort() -> None:
    client = RecordingClient([message_start("claude-opus-5-5"), message_delta("end_turn")])
    await collect_answer(
        streamer_for(client), GenerationRequest("How long may a line be?", [PASSAGE])
    )

    (call,) = client.calls
    assert call["model"] == "claude-opus-5-5"
    assert call["max_tokens"] == 16000
    assert call["thinking"] == {"type": "adaptive"}
    assert call["output_config"] == {"effort": "medium"}
    assert call["fallbacks"] == "default"
    assert call["betas"] == [FALLBACK_BETA] == ["server-side-fallback-2026-07-01"]
    for rejected in ("temperature", "top_p", "top_k", "output_format"):
        assert rejected not in call
    assert "format" not in call["output_config"], (
        "structured outputs cannot be combined with citations"
    )
    assert call["system"] == [
        {"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}
    ]

    (turn,) = call["messages"]
    assert turn["role"] == "user"
    document, question = turn["content"]
    assert document["citations"] == {"enabled": True}
    assert document["source"] == {"type": "text", "media_type": "text/plain", "data": PASSAGE.text}
    assert question == {"type": "text", "text": "How long may a line be?"}


async def test_helper_events_are_filtered_and_raw_events_drive_the_answer() -> None:
    helper = ParsedBetaTextEvent(type="text", text="duplicate", snapshot="duplicate")
    events: list[object] = [
        message_start("claude-opus-5-5"),
        text_start(0),
        text_delta(0, "Lines stop at 79."),
        helper,
        citation_delta(0, 0, PASSAGE, 0, len(PASSAGE.text)),
        message_delta("end_turn"),
    ]
    answer = await collect_answer(
        streamer_for(RecordingClient(events)), GenerationRequest("q", [PASSAGE])
    )

    assert answer.text == "Lines stop at 79."
    assert [(f.n, f.start, f.end) for f in answer.footnotes] == [(1, 5000, 5046)]


def test_system_prompt_has_no_per_request_content() -> None:
    # Anything that varies per request (dates, ids) would break the cache prefix.
    assert "{" not in SYSTEM_PROMPT
    # Opus 5.5 caches prefixes of 512 tokens or more. English runs about four characters
    # per token; five is a conservative bound, so this length clears the minimum.
    assert len(SYSTEM_PROMPT) >= 512 * 5


def test_document_block_carries_title_and_location_context() -> None:
    block = document_block(PASSAGE.model_copy(update={"page_start": 3, "page_end": 4}))
    assert block.get("title") == "PEP 8: Style Guide for Python Code"
    assert block.get("context") == (
        "File: pep-0008.html\nSection: PEP 8 > Code Lay-out > Maximum Line Length\nPages: 3-4"
    )


def test_history_is_replayed_as_plain_text_and_starts_with_a_user_turn() -> None:
    history = [
        HistoryTurn("assistant", "orphaned answer from a trimmed window"),
        HistoryTurn("user", "What about tabs?"),
        HistoryTurn("assistant", "Spaces are preferred."),
    ]
    messages = build_messages("And line length?", [PASSAGE], history)

    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert messages[0].get("content") == "What about tabs?"
    assert messages[1].get("content") == "Spaces are preferred."
    # Only the current turn carries documents.
    assert all(isinstance(m.get("content"), str) for m in messages[:-1])


def test_history_is_capped() -> None:
    history = [HistoryTurn("user" if i % 2 == 0 else "assistant", f"turn {i}") for i in range(20)]
    messages = build_messages("latest", [PASSAGE], history)
    assert len(messages) == 7  # six replayed turns plus the question
    assert messages[0].get("content") == "turn 14"


def test_has_credentials_follows_the_sdk() -> None:
    assert has_credentials(AsyncAnthropic(api_key="sk-test"))
    assert not has_credentials(
        cast(AsyncAnthropic, SimpleNamespace(api_key=None, auth_token=None, credentials=None))
    )
