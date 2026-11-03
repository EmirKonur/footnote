from __future__ import annotations

import pytest

from footnote.ingest.document import Block, ParsedDocument, assemble
from footnote.ingest.parsers import (
    DocumentParseError,
    UnsupportedFormatError,
    parse_document,
    parse_pdf,
)
from tests.conftest import fixture_bytes
from tests.fakes import build_pdf


def kinds(parsed: ParsedDocument) -> list[tuple[str, int, str]]:
    return [(b.kind, b.level, b.text) for b in parsed.blocks]


def test_markdown_keeps_structure_and_drops_markup() -> None:
    parsed = parse_document("field-guide.md", fixture_bytes("field-guide.md"))

    assert parsed.title == "Field Guide to the Reading Room"
    assert parsed.format == "markdown"
    blocks = kinds(parsed)
    assert blocks[0] == ("heading", 1, "Field Guide to the Reading Room")
    assert ("heading", 2, "Handling manuscripts") in blocks
    assert ("heading", 3, "Fragile bindings") in blocks
    # Soft line breaks inside a paragraph become spaces.
    assert (
        "paragraph",
        0,
        "Wash your hands before touching any manuscript. Cotton gloves are not required for "
        "paper, because they reduce dexterity and cause more tears than they prevent.",
    ) in blocks
    # Ordered items keep their numbers; bullets do not get a marker.
    assert ("list_item", 0, "1. Support the spine with foam wedges.") in blocks
    assert ("list_item", 0, "2. Turn pages from the upper corner.") in blocks
    assert ("list_item", 0, "Pencils only, never ink.") in blocks
    assert ("row", 0, "Weekdays | 9:00 to 17:00") in blocks
    assert ("quote", 0, "Closed on public holidays, without exception.") in blocks
    assert ("code", 0, "def is_open(hour: int) -> bool:\n    return 9 <= hour < 17") in blocks
    assert all("ignored front matter" not in text for _, _, text in blocks)


def test_html_reads_main_content_only() -> None:
    parsed = parse_document("catalogue.html", fixture_bytes("catalogue.html"))
    blocks = kinds(parsed)

    assert parsed.title == "Catalogue Notes"
    assert blocks[0] == ("heading", 1, "Catalogue Notes")
    assert ("paragraph", 0, "Every item receives a shelfmark before it is catalogued.") in blocks
    # A list item with a nested list contributes its own text, then the nested item.
    assert ("list_item", 0, "Prefix with the room letter.") in blocks
    assert ("list_item", 0, "Room A holds folios.") in blocks
    assert ("term", 0, "Folio") in blocks
    assert ("code", 0, "A-1912-004\nA-1913-011") in blocks
    assert ("row", 0, "B | Maps") in blocks
    text = " ".join(t for _, _, t in blocks)
    assert "Home" not in text
    assert "never indexed" not in text
    assert "Copyright notice" not in text


def test_text_recognises_underlined_headings() -> None:
    parsed = parse_document("loans.txt", fixture_bytes("loans.txt"))

    assert parsed.title == "Loan Policy"
    assert kinds(parsed) == [
        ("heading", 1, "Loan Policy"),
        (
            "paragraph",
            0,
            "Members may borrow up to six books at once. Reference works stay in the building.",
        ),
        ("heading", 2, "Renewals"),
        ("paragraph", 0, "A loan can be renewed twice,\nonline or at the desk."),
        (
            "paragraph",
            0,
            "Overdue items accrue no fines, but borrowing is paused until they come back.",
        ),
    ]


def test_pdf_tracks_pages_outline_headings_and_hyphenation() -> None:
    data = build_pdf(
        [
            [
                "Introduction",
                "Ledgers are kept for every reader who enters the build-",
                "ing after hours.",
                "",
                "A second paragraph on page one.",
            ],
            ["Retention", "Ledgers are retained for seven years."],
        ],
        title="Ledger Rules",
        outline=[("Introduction", 0), ("Retention", 1)],
    )
    parsed = parse_pdf(data, "fallback")

    assert parsed.title == "Ledger Rules"
    assert [(b.kind, b.text, b.page) for b in parsed.blocks] == [
        ("heading", "Introduction", 1),
        ("paragraph", "Ledgers are kept for every reader who enters the building after hours.", 1),
        ("paragraph", "A second paragraph on page one.", 1),
        ("heading", "Retention", 2),
        ("paragraph", "Ledgers are retained for seven years.", 2),
    ]


def test_corrupt_pdf_is_a_parse_error() -> None:
    with pytest.raises(DocumentParseError):
        parse_document("broken.pdf", b"%PDF-1.4 this is not really a pdf")


def test_unknown_extension_is_rejected() -> None:
    with pytest.raises(UnsupportedFormatError):
        parse_document("slides.pptx", b"")


def test_title_falls_back_to_filename() -> None:
    parsed = parse_document("notes/meeting_minutes-2026.txt", b"no headings here")
    assert parsed.title == "meeting minutes 2026"


def test_assemble_places_blocks_with_exact_offsets() -> None:
    parsed = ParsedDocument(
        "t",
        "text",
        [Block("heading", "Title", level=1), Block("paragraph", "  "), Block("paragraph", "Body.")],
    )
    doc = assemble(parsed)

    assert doc.text == "Title\n\nBody."
    assert [(b.start, b.end) for b in doc.blocks] == [(0, 5), (7, 12)]
    assert [doc.block_text(b) for b in doc.blocks] == ["Title", "Body."]
