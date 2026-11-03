from __future__ import annotations

from anthropic.types.beta import BetaCitationCharLocation, BetaCitationPageLocation

from footnote.answer.citations import FootnoteBook
from footnote.schemas import Passage


def passage(
    text: str,
    *,
    start: int,
    document_id: str = "doc_a",
    chunk_id: int = 1,
    pages: tuple[int | None, int | None] = (None, None),
) -> Passage:
    return Passage(
        chunk_id=chunk_id,
        document_id=document_id,
        document_title="Style Guide",
        document_path="style.md",
        heading_path=["Style Guide", "Layout"],
        page_start=pages[0],
        page_end=pages[1],
        start=start,
        end=start + len(text),
        text=text,
        score=1.0,
    )


def cite(document_index: int, start: int, end: int, quote: str) -> BetaCitationCharLocation:
    return BetaCitationCharLocation(
        type="char_location",
        cited_text=quote,
        document_index=document_index,
        document_title="Style Guide",
        start_char_index=start,
        end_char_index=end,
    )


TEXT = "Use 4 spaces per indentation level. Limit all lines to 79 characters. "


def test_offsets_are_shifted_into_document_coordinates() -> None:
    book = FootnoteBook([passage(TEXT, start=1000)])
    footnote = book.resolve(cite(0, 36, 71, "Limit all lines to 79 characters. "))

    assert footnote is not None
    assert footnote.verified
    # The trailing space the API includes after a sentence is trimmed off the highlight.
    assert (footnote.start, footnote.end) == (1036, 1069)
    assert footnote.cited_text == "Limit all lines to 79 characters."
    assert footnote.heading_path == ["Style Guide", "Layout"]


def test_wrong_offsets_are_repaired_from_the_quote() -> None:
    book = FootnoteBook([passage(TEXT, start=500)])
    footnote = book.resolve(cite(0, 0, 10, "Limit all lines to 79 characters."))

    assert footnote is not None
    assert footnote.verified
    assert (footnote.start, footnote.end) == (536, 569)


def test_a_quote_that_is_not_in_the_passage_is_unverified_and_unlinked() -> None:
    book = FootnoteBook([passage(TEXT, start=0)])
    footnote = book.resolve(cite(0, 0, 20, "Tabs are preferred over spaces."))

    assert footnote is not None
    assert not footnote.verified
    assert footnote.start is None
    assert footnote.end is None
    assert footnote.cited_text == "Tabs are preferred over spaces."


def test_whitespace_differences_still_verify() -> None:
    text = "Line one of the rule,\nline two of the rule."
    book = FootnoteBook([passage(text, start=0)])
    footnote = book.resolve(cite(0, 0, len(text), "Line one of the rule, line two of the rule."))
    assert footnote is not None
    assert footnote.verified


def test_same_span_reuses_the_number_and_new_spans_count_up() -> None:
    passages = [
        passage(TEXT, start=0),
        passage("Another rule here.", start=300, document_id="doc_b", chunk_id=2),
    ]
    book = FootnoteBook(passages)

    first = book.resolve(cite(0, 0, 35, "Use 4 spaces per indentation level."))
    again = book.resolve(cite(0, 0, 35, "Use 4 spaces per indentation level."))
    second = book.resolve(cite(1, 0, 18, "Another rule here."))

    assert first is not None
    assert again is not None
    assert second is not None
    assert (first.n, again.n, second.n) == (1, 1, 2)
    assert len(book.footnotes) == 2


def test_overlapping_chunks_citing_the_same_text_share_a_footnote() -> None:
    # Chunk overlap means two passages can contain the same sentence at different local offsets.
    first_chunk = passage(TEXT, start=0, chunk_id=1)
    second_chunk = passage(TEXT[36:], start=36, chunk_id=2)
    book = FootnoteBook([first_chunk, second_chunk])

    a = book.resolve(cite(0, 36, 69, "Limit all lines to 79 characters."))
    b = book.resolve(cite(1, 0, 33, "Limit all lines to 79 characters."))
    assert a is not None
    assert b is not None
    assert a.n == b.n == 1


def test_single_page_chunks_report_their_page() -> None:
    book = FootnoteBook(
        [passage(TEXT, start=0, pages=(4, 4)), passage(TEXT, start=900, pages=(4, 5))]
    )
    on_page = book.resolve(cite(0, 0, 35, "Use 4 spaces per indentation level."))
    spanning = book.resolve(cite(1, 0, 35, "Use 4 spaces per indentation level."))
    assert on_page is not None
    assert spanning is not None
    assert (on_page.page, spanning.page) == (4, None)


def test_out_of_range_and_foreign_citation_types_are_dropped() -> None:
    book = FootnoteBook([passage(TEXT, start=0)])
    assert book.resolve(cite(3, 0, 5, "Use 4")) is None
    page_citation = BetaCitationPageLocation(
        type="page_location",
        cited_text="x",
        document_index=0,
        document_title="t",
        start_page_number=1,
        end_page_number=2,
    )
    assert book.resolve(page_citation) is None
    assert book.footnotes == []
