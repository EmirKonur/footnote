from __future__ import annotations

import anthropic
import httpx2
import pytest

from footnote.answer.events import (
    DoneEvent,
    ErrorEvent,
    FootnoteEvent,
    NoticeEvent,
    PassagesEvent,
    TextEvent,
)
from footnote.answer.fixtures import (
    FixtureStreamer,
    citation_delta,
    fallback_start,
    message_delta,
    message_start,
    pick_sentences,
    text_delta,
    text_start,
)
from footnote.answer.generate import (
    AnswerRun,
    GenerationRequest,
    OutEvent,
    StreamEvent,
    collect_answer,
)
from footnote.schemas import Passage
from tests.fakes import ScriptedStreamer

MODEL = "claude-opus-5-5"


def make_passage(
    text: str, *, start: int = 0, document_id: str = "doc_1", title: str = "Loan Policy"
) -> Passage:
    return Passage(
        chunk_id=1,
        document_id=document_id,
        document_title=title,
        document_path="loans.txt",
        heading_path=[title],
        page_start=None,
        page_end=None,
        start=start,
        end=start + len(text),
        text=text,
        score=0.5,
    )


PASSAGE = make_passage(
    "Members may borrow up to six books at once. Reference works stay in the building.", start=40
)


def answer_events(passage: Passage = PASSAGE) -> list[StreamEvent]:
    return [
        message_start(MODEL),
        text_start(0),
        text_delta(0, "You can borrow "),
        text_delta(0, "six books."),
        text_start(1),
        text_delta(1, " Members may borrow up to six books at once."),
        citation_delta(1, 0, passage, 0, 43),
        text_start(2),
        text_delta(2, " Reference works stay put."),
        citation_delta(2, 0, passage, 44, 81),
        message_delta("end_turn", output_tokens=40),
    ]


async def run(
    streamer: ScriptedStreamer | FixtureStreamer,
    question: str = "How many books?",
    passages: list[Passage] | None = None,
) -> list[OutEvent]:
    request = GenerationRequest(question, [PASSAGE] if passages is None else passages)
    return [event async for event in AnswerRun(streamer, request).events()]


async def test_text_blocks_become_segments_with_footnotes() -> None:
    events = await run(ScriptedStreamer(answer_events()))

    assert isinstance(events[0], PassagesEvent)
    done = events[-1]
    assert isinstance(done, DoneEvent)
    answer = done.answer
    assert answer.status == "complete"
    assert answer.stop_reason == "end_turn"
    assert [(s.text, s.footnotes) for s in answer.segments] == [
        ("You can borrow six books.", []),
        (" Members may borrow up to six books at once.", [1]),
        (" Reference works stay put.", [2]),
    ]
    assert [(f.n, f.start, f.end, f.verified) for f in answer.footnotes] == [
        (1, 40, 83, True),
        (2, 84, 121, True),
    ]
    footnote_events = [e for e in events if isinstance(e, FootnoteEvent)]
    assert [(e.segment, e.footnote.n) for e in footnote_events] == [(1, 1), (2, 2)]
    streamed = "".join(e.text for e in events if isinstance(e, TextEvent))
    assert streamed == answer.text


async def test_refusal_discards_partial_text_and_records_the_category() -> None:
    events = [
        message_start(MODEL),
        text_start(0),
        text_delta(0, "Partial words that must not be shown"),
        message_delta("refusal", refusal_category="cyber"),
    ]
    out = await run(ScriptedStreamer(events))
    done = out[-1]
    assert isinstance(done, DoneEvent)
    assert done.answer.status == "refused"
    assert done.answer.refusal_category == "cyber"
    assert done.answer.segments == []


async def test_max_tokens_keeps_the_text_and_flags_truncation() -> None:
    events = [
        message_start(MODEL),
        text_start(0),
        text_delta(0, "A long answer that"),
        message_delta("max_tokens"),
    ]
    out = await run(ScriptedStreamer(events))
    done = out[-1]
    assert isinstance(done, DoneEvent)
    assert done.answer.status == "truncated"
    assert done.answer.text == "A long answer that"
    assert any(isinstance(e, NoticeEvent) and "cut off" in e.message for e in out)


async def test_mid_stream_fallback_is_announced_and_the_answer_continues() -> None:
    events = [
        message_start(MODEL),
        text_start(0),
        text_delta(0, "Start. "),
        fallback_start(1, MODEL, "claude-opus-4-8"),
        text_start(2),
        text_delta(2, "Finished by the fallback."),
        message_delta("end_turn"),
    ]
    out = await run(ScriptedStreamer(events))
    done = out[-1]
    assert isinstance(done, DoneEvent)
    assert done.answer.text == "Start. Finished by the fallback."
    assert done.answer.model == "claude-opus-4-8"
    assert done.answer.notices == [
        f"{MODEL} declined part of this request; claude-opus-4-8 continued the answer."
    ]


async def test_sticky_fallback_model_in_message_start_is_reported() -> None:
    events = [
        message_start("claude-opus-5"),
        text_start(0),
        text_delta(0, "Hi."),
        message_delta("end_turn"),
    ]
    out = await run(ScriptedStreamer(events))
    done = out[-1]
    assert isinstance(done, DoneEvent)
    assert done.answer.model == "claude-opus-5"
    assert "fallback" in done.answer.notices[0]


def _status_error(cls: type[anthropic.APIStatusError], code: int) -> anthropic.APIStatusError:
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    return cls("boom", response=httpx2.Response(code, request=request), body=None)


@pytest.mark.parametrize(
    ("error", "code", "retryable"),
    [
        (_status_error(anthropic.RateLimitError, 429), "rate_limited", True),
        (_status_error(anthropic.AuthenticationError, 401), "auth_failed", False),
        (_status_error(anthropic.NotFoundError, 404), "model_not_found", False),
        (_status_error(anthropic.OverloadedError, 529), "overloaded", True),
        (_status_error(anthropic.InternalServerError, 500), "upstream_error", True),
        (_status_error(anthropic.BadRequestError, 400), "bad_request", False),
        (
            anthropic.APITimeoutError(httpx2.Request("POST", "https://api.anthropic.com")),
            "timeout",
            True,
        ),
        (
            anthropic.APIConnectionError(
                request=httpx2.Request("POST", "https://api.anthropic.com")
            ),
            "connection_failed",
            True,
        ),
    ],
)
async def test_sdk_errors_become_an_error_event_then_done(
    error: anthropic.AnthropicError, code: str, retryable: bool
) -> None:
    streamer = ScriptedStreamer(answer_events(), raise_after=3, error=error)
    out = await run(streamer)

    errors = [e for e in out if isinstance(e, ErrorEvent)]
    assert [(e.code, e.retryable) for e in errors] == [(code, retryable)]
    done = out[-1]
    assert isinstance(done, DoneEvent)
    assert done.answer.status == "error"
    assert done.answer.text == "You can borrow "  # what streamed before the failure is kept
    assert streamer.closed


async def test_no_passages_short_circuits_without_calling_the_model() -> None:
    streamer = ScriptedStreamer(answer_events())
    out = await run(streamer, passages=[])
    done = out[-1]
    assert isinstance(done, DoneEvent)
    assert done.answer.text.startswith("The documents don't cover this")
    assert streamer.requests == []


async def test_partial_marks_an_interrupted_answer_as_stopped() -> None:
    run_ = AnswerRun(ScriptedStreamer(answer_events()), GenerationRequest("q", [PASSAGE]))
    events = run_.events()
    seen = [await anext(events) for _ in range(4)]
    await events.aclose()

    assert any(isinstance(e, TextEvent) for e in seen)
    partial = run_.partial()
    assert partial.status == "stopped"
    # Four events in: passages, two deltas of the first segment, one of the second.
    assert partial.text == "You can borrow six books. Members may borrow up to six books at once."


async def test_fixture_streamer_runs_the_full_citation_path() -> None:
    answer = await collect_answer(
        FixtureStreamer(word_delay=0),
        GenerationRequest("How many books may members borrow at once?", [PASSAGE]),
    )
    assert answer.generated_by == "fixtures"
    assert answer.status == "complete"
    assert len(answer.footnotes) == 1
    footnote = answer.footnotes[0]
    assert footnote.verified
    assert footnote.cited_text == "Members may borrow up to six books at once."
    assert (footnote.start, footnote.end) == (40, 83)


async def test_fixture_streamer_abstains_when_nothing_matches() -> None:
    answer = await collect_answer(
        FixtureStreamer(word_delay=0),
        GenerationRequest("What is the boiling point of mercury?", [PASSAGE]),
    )
    assert answer.footnotes == []
    assert answer.text.startswith("The documents don't cover this")


def test_pick_sentences_prefers_higher_ranked_passages_and_skips_overlap() -> None:
    first = make_passage("Loans can be renewed twice online. Loans are free.", document_id="a")
    second = make_passage("Loans can be renewed twice online at the desk.", document_id="b")
    picks = pick_sentences("Can loans be renewed online?", [first, second], limit=1)
    assert [(p.document_index, p.start) for p in picks] == [(0, 0)]


def test_pick_sentences_does_not_glue_a_heading_onto_the_next_sentence() -> None:
    chunk = make_passage(
        "Loan Policy\n\nMembers may borrow up to six books at once. Reference works stay."
    )
    (pick,) = pick_sentences("How many books may members borrow at once?", [chunk], limit=1)
    assert chunk.text[pick.start : pick.end] == "Members may borrow up to six books at once."


async def test_fixture_answers_are_the_cited_sentences_with_no_scaffolding() -> None:
    answer = await collect_answer(
        FixtureStreamer(word_delay=0),
        GenerationRequest("How many books may members borrow at once?", [PASSAGE]),
    )
    assert [(s.text, s.footnotes) for s in answer.segments] == [
        ("Members may borrow up to six books at once.", [1])
    ]


def test_extra_sentences_come_from_the_best_documents_own_text_in_reading_order() -> None:
    style = "Lines must stay short. Limit every line of Python code to 79 characters."
    best = make_passage(style, start=500, document_id="style")
    # The same document again (an overlapping chunk), plus an earlier, weaker sentence.
    overlap = make_passage(
        "Python code lines in docstrings wrap at 72. " + style, start=456, document_id="style"
    )
    # An unrelated document that happens to share the question's words.
    other = make_passage("Python line counts of code vary by editor.", document_id="editor")
    picks = pick_sentences("How long may a line of Python code be?", [best, other, overlap])

    spans = [(p.document_index, p.start, p.end) for p in picks]
    texts = [[best, other, overlap][i].text[s:e] for i, s, e in spans]
    assert texts == [
        "Python code lines in docstrings wrap at 72.",
        "Limit every line of Python code to 79 characters.",
    ]
    assert all(i != 1 for i, _, _ in spans), "no sentence from the unrelated document"
