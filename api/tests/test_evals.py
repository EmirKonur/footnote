from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from anthropic import AsyncAnthropic

from footnote.answer.fixtures import (
    citation_delta,
    message_delta,
    message_start,
    text_delta,
    text_start,
)
from footnote.answer.generate import GenerationRequest, StreamEvent
from footnote.evals.corpus import CORPUS_DIR, GOLDEN_PATH
from footnote.evals.generation_run import evaluate_items, summarize
from footnote.evals.golden import GoldenItem, Span, load_golden, locate_quote, resolve_spans
from footnote.evals.judge import (
    CitationVerdict,
    Judge,
    JudgeFailure,
    SegmentVerdict,
    Verdict,
    to_judgements,
)
from footnote.evals.metrics import (
    RetrievedSpan,
    SegmentJudgement,
    coverage,
    covers,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    score_answer,
)
from footnote.evals.report import check_thresholds
from footnote.evals.retrieval_run import regressions
from footnote.ingest.document import assemble
from footnote.ingest.parsers import parse_document
from footnote.schemas import Answer, Footnote, Passage, Segment
from tests.fakes import ScriptedStreamer

# Retrieval metrics --------------------------------------------------------------------

EXPECTED = [Span("a.md", 100, 200), Span("b.md", 0, 50)]


def chunk(doc: str, start: int, end: int) -> RetrievedSpan:
    return RetrievedSpan(doc, start, end)


def test_cover_needs_half_the_passage_in_the_same_document() -> None:
    passage = Span("a.md", 100, 200)
    assert covers(chunk("a.md", 150, 400), passage)  # exactly half
    assert not covers(chunk("a.md", 151, 400), passage)
    assert covers(chunk("a.md", 0, 1000), passage)
    assert not covers(chunk("b.md", 100, 200), passage)


def test_hand_computed_retrieval_metrics() -> None:
    ranked = [
        chunk("c.md", 0, 10),  # rank 1: miss
        chunk("a.md", 90, 210),  # rank 2: covers passage 0
        chunk("a.md", 120, 260),  # rank 3: covers passage 0 again (overlapping chunk)
        chunk("b.md", 0, 40),  # rank 4: covers passage 1
    ]
    hits = coverage(ranked, EXPECTED)
    assert hits == [set(), {0}, {0}, {1}]

    assert recall_at_k(hits, 2, 1) == 0.0
    assert recall_at_k(hits, 2, 3) == 0.5
    assert recall_at_k(hits, 2, 4) == 1.0
    assert reciprocal_rank(hits, 10) == 0.5
    assert reciprocal_rank(hits, 1) == 0.0
    # Gains: rank 2 and rank 4 cover something new; rank 3 is a duplicate and earns nothing.
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    ideal = 1 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(hits, 2, 10) == pytest.approx(dcg / ideal)
    assert ndcg_at_k(hits, 2, 4) == pytest.approx(dcg / ideal)
    assert ndcg_at_k([{0}], 1, 5) == 1.0


def test_metrics_reject_questions_without_expected_passages() -> None:
    with pytest.raises(ValueError, match="undefined"):
        recall_at_k([], 0, 5)
    with pytest.raises(ValueError, match="undefined"):
        ndcg_at_k([], 0, 5)


def test_quotes_are_found_across_whitespace_differences() -> None:
    text = "Limit all lines\nto a maximum   of 79 characters."
    found = locate_quote(text, "lines to a maximum of 79")
    assert found is not None
    assert text[found[0] : found[1]] == "lines\nto a maximum   of 79"
    assert locate_quote(text, "80 characters") is None


def test_every_golden_quote_resolves_against_the_bundled_corpus() -> None:
    items = load_golden(GOLDEN_PATH)
    texts = {
        path.name: assemble(parse_document(path.name, path.read_bytes())).text
        for path in CORPUS_DIR.iterdir()
    }
    spans = resolve_spans(items, texts)

    assert 30 <= len(items) <= 50
    assert sum(not item.answerable for item in items) >= 5
    for item in items:
        assert len(spans[item.id]) == len(item.expected)
        assert (item.reference_answer is not None) == item.answerable


def test_missing_quotes_fail_loudly(tmp_path: Path) -> None:
    golden = tmp_path / "golden.jsonl"
    golden.write_text(
        '{"id": "x", "question": "q", "answerable": true, "reference_answer": "r", "tags": [],'
        ' "expected": [{"doc": "a.md", "quote": "not in the text"}]}\n'
    )
    with pytest.raises(ValueError, match="not found"):
        resolve_spans(load_golden(golden), {"a.md": "something else entirely"})


# Generation metrics -------------------------------------------------------------------


def test_hand_computed_answer_scores() -> None:
    segments = [
        SegmentJudgement(is_claim=False, supported=True, citations={}),  # "According to PEP 8,"
        SegmentJudgement(is_claim=True, supported=True, citations={1: True, 2: False}),
        SegmentJudgement(is_claim=True, supported=False, citations={3: False}),
        SegmentJudgement(is_claim=True, supported=True, citations={}),  # supported but uncited
    ]
    scores = score_answer(segments)
    assert scores.faithfulness == pytest.approx(2 / 3)
    assert scores.citation_support == pytest.approx(1 / 3)
    assert scores.citation_precision == pytest.approx(1 / 3)


def test_scores_are_undefined_rather_than_zero_without_claims() -> None:
    scores = score_answer([SegmentJudgement(is_claim=False, supported=True, citations={})])
    assert (scores.faithfulness, scores.citation_support, scores.citation_precision) == (
        None,
        None,
        None,
    )


def test_summary_separates_abstention_from_answer_quality() -> None:
    rows: list[dict[str, Any]] = [
        {
            "answerable": True,
            "abstained": False,
            "faithfulness": 1.0,
            "citation_support": 1.0,
            "citation_precision": 0.5,
            "correctness": "correct",
        },
        {
            "answerable": True,
            "abstained": False,
            "faithfulness": 0.5,
            "citation_support": 0.0,
            "citation_precision": None,
            "correctness": "partially_correct",
        },
        {
            "answerable": True,
            "abstained": True,
            "faithfulness": None,
            "citation_support": None,
            "citation_precision": None,
            "correctness": "incorrect",
        },
        {
            "answerable": False,
            "abstained": True,
            "faithfulness": None,
            "citation_support": None,
            "citation_precision": None,
            "correctness": "not_applicable",
        },
        {
            "answerable": False,
            "abstained": False,
            "faithfulness": 0.0,
            "citation_support": 0.0,
            "citation_precision": 0.0,
            "correctness": "not_applicable",
        },
        {"answerable": True, "status": "refused"},  # never judged: excluded everywhere
    ]
    summary = summarize(rows)
    assert summary["faithfulness"] == pytest.approx(0.75)  # mean of 1.0 and 0.5
    assert summary["citation_support"] == pytest.approx(0.5)
    assert summary["citation_precision"] == pytest.approx(0.5)  # the None is left out
    assert summary["answer_correctness"] == pytest.approx(0.5)  # (1 + 0.5 + 0) / 3
    assert summary["abstention_accuracy"] == pytest.approx(0.5)
    assert summary["false_abstention_rate"] == pytest.approx(1 / 3)


def _answer(segments: list[Segment], footnotes: list[Footnote]) -> Answer:
    return Answer(
        status="complete",
        segments=segments,
        footnotes=footnotes,
        passages=[],
        generated_by="anthropic",
    )


def _footnote(n: int) -> Footnote:
    return Footnote(
        n=n,
        document_id="d",
        document_title="T",
        document_path="t.md",
        chunk_id=1,
        heading_path=[],
        page=None,
        start=0,
        end=5,
        cited_text="quote",
        verified=True,
    )


def test_judge_omissions_can_only_lower_the_score() -> None:
    answer = _answer(
        [Segment(text="Claim one.", footnotes=[1, 2]), Segment(text="Claim two.", footnotes=[3])],
        [_footnote(1), _footnote(2), _footnote(3)],
    )
    verdict = Verdict(
        segments=[
            SegmentVerdict(segment=0, is_claim=True, supported_by_passages=True, citations=[])
        ],
        abstained=False,
        correctness="correct",
        rationale="",
    )
    judgements = to_judgements(verdict, answer)
    assert judgements[0].citations == {1: False, 2: False}  # footnotes the judge skipped
    assert judgements[1] == SegmentJudgement(True, False, {3: False})  # segment it skipped


# The judge against a mocked client ----------------------------------------------------

ITEM = GoldenItem(
    id="q1",
    question="How long can lines be?",
    answerable=True,
    expected=[],
    reference_answer="79 characters.",
    tags=[],
)


class FakeParseClient:
    def __init__(self, response: object) -> None:
        self.calls: list[dict[str, Any]] = []
        self._response = response
        self.beta = SimpleNamespace(messages=SimpleNamespace(parse=self._parse))

    async def _parse(self, **kwargs: Any) -> object:
        self.calls.append(kwargs)
        return self._response


async def test_judge_returns_the_parsed_verdict_and_sends_no_citations() -> None:
    verdict = Verdict(segments=[], abstained=False, correctness="correct", rationale="Matches.")
    client = FakeParseClient(SimpleNamespace(stop_reason="end_turn", parsed_output=verdict))
    judge = Judge(cast(AsyncAnthropic, client), "claude-opus-5-5")
    answer = _answer([Segment(text="Lines stop at 79.", footnotes=[1])], [_footnote(1)])

    assert await judge.judge(ITEM, answer) == verdict
    (call,) = client.calls
    assert call["output_format"] is Verdict
    assert call["model"] == "claude-opus-5-5"
    assert call["fallbacks"] == "default"
    content = call["messages"][0]["content"]
    assert isinstance(content, str), "judge input is plain text: no citation-enabled documents"
    assert '<segment index="0" footnotes="1">Lines stop at 79.</segment>' in content
    assert "<reference_answer>79 characters.</reference_answer>" in content


@pytest.mark.parametrize(
    ("response", "reason"),
    [
        (SimpleNamespace(stop_reason="refusal", parsed_output=None), "declined"),
        (SimpleNamespace(stop_reason="max_tokens", parsed_output=None), "output tokens"),
        (SimpleNamespace(stop_reason="end_turn", parsed_output=None), "no parseable"),
    ],
)
async def test_judge_failures_are_reported_not_scored(response: object, reason: str) -> None:
    judge = Judge(cast(AsyncAnthropic, FakeParseClient(response)), "claude-opus-5-5")
    result = await judge.judge(ITEM, _answer([], []))
    assert isinstance(result, JudgeFailure)
    assert reason in result.reason


async def test_evaluate_items_end_to_end_with_fakes() -> None:
    passage = Passage(
        chunk_id=1,
        document_id="d8",
        document_title="PEP 8",
        document_path="pep-0008.html",
        heading_path=["PEP 8"],
        page_start=None,
        page_end=None,
        start=0,
        end=46,
        text="Limit all lines to a maximum of 79 characters.",
        score=1.0,
    )

    def script(request: GenerationRequest) -> list[StreamEvent]:
        return [
            message_start("claude-opus-5-5"),
            text_start(0),
            text_delta(0, "79 characters."),
            citation_delta(0, 0, request.passages[0], 0, 46),
            message_delta("end_turn"),
        ]

    judged: list[Answer] = []

    async def judge(item: GoldenItem, answer: Answer) -> Verdict | JudgeFailure:
        judged.append(answer)
        if item.id == "broken":
            return JudgeFailure("the judge returned no parseable verdict")
        return Verdict(
            segments=[
                SegmentVerdict(
                    segment=0,
                    is_claim=True,
                    supported_by_passages=True,
                    citations=[CitationVerdict(footnote=1, supports=True)],
                )
            ],
            abstained=False,
            correctness="correct",
            rationale="ok",
        )

    items = [ITEM, ITEM.model_copy(update={"id": "broken"})]
    rows = await evaluate_items(items, lambda _q: [passage], ScriptedStreamer(script), judge)

    assert rows[0] | {"rationale": None} == {
        "id": "q1",
        "answerable": True,
        "status": "complete",
        "faithfulness": 1.0,
        "citation_support": 1.0,
        "citation_precision": 1.0,
        "correctness": "correct",
        "abstained": False,
        "rationale": None,
    }
    assert rows[1]["status"] == "judge_failed"
    assert judged[0].footnotes[0].cited_text == passage.text


# CI gates -------------------------------------------------------------------------------


def test_fail_under_specs() -> None:
    results = {"hybrid": {"recall@5": 0.86, "mrr@10": 0.70}}
    assert check_thresholds(results, ["hybrid.recall@5=0.85"]) == []
    assert check_thresholds(results, ["hybrid.mrr@10=0.75"]) == [
        "hybrid.mrr@10 = 0.7 is below 0.75"
    ]
    with pytest.raises(ValueError, match="no metric named"):
        check_thresholds(results, ["hybird.recall@5=0.8"])
    with pytest.raises(ValueError, match="METRIC=VALUE"):
        check_thresholds(results, ["hybrid.recall@5"])


def test_regressions_ignore_latency_and_respect_tolerance() -> None:
    baseline = {"hybrid": {"recall@5": 0.90, "mrr@10": 0.70, "latency_ms_p50": 1.0}}
    current = {"hybrid": {"recall@5": 0.89, "mrr@10": 0.60, "latency_ms_p50": 50.0}}
    assert regressions(current, baseline, tolerance=0.02) == [
        "hybrid.mrr@10 fell from 0.700 to 0.600"
    ]
