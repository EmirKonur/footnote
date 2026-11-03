from __future__ import annotations

from itertools import pairwise

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from footnote.ingest.chunker import ChunkingConfig, chunk_document, sentence_spans
from footnote.ingest.document import Block, BlockKind, CanonicalDocument, ParsedDocument, assemble


def doc(*blocks: Block) -> CanonicalDocument:
    return assemble(ParsedDocument("Doc", "text", list(blocks)))


def h(level: int, text: str, page: int | None = None) -> Block:
    return Block("heading", text, level=level, page=page)


def p(text: str, page: int | None = None) -> Block:
    return Block("paragraph", text, page=page)


SMALL = ChunkingConfig(target_chars=60, max_chars=100, overlap_chars=30)


def test_sentence_spans_skip_abbreviations_and_exclude_gaps() -> None:
    text = "Use e.g. spaces. Then stop!  Next one (really).\nLast"
    spans = sentence_spans(text)
    assert [text[s:e] for s, e in spans] == [
        "Use e.g. spaces.",
        "Then stop!",
        "Next one (really).",
        "Last",
    ]


def test_every_heading_starts_a_new_chunk() -> None:
    document = doc(h(1, "Guide"), p("Alpha one."), h(2, "Part B"), p("Beta two."))
    chunks = chunk_document(document, SMALL)

    assert [c.text for c in chunks] == ["Guide\n\nAlpha one.", "Part B\n\nBeta two."]
    assert [c.heading_path for c in chunks] == [("Guide",), ("Guide", "Part B")]


def test_consecutive_headings_join_the_content_that_follows() -> None:
    document = doc(h(1, "Guide"), h(2, "Layout"), h(3, "Indentation"), p("Use four spaces."))
    (chunk,) = chunk_document(document, SMALL)

    assert chunk.text == "Guide\n\nLayout\n\nIndentation\n\nUse four spaces."
    assert chunk.heading_path == ("Guide", "Layout", "Indentation")


def test_heading_path_pops_back_to_the_right_level() -> None:
    document = doc(
        h(1, "A"), h(2, "B"), p("x" * 10 + "."), h(3, "C"), p("y" * 10 + "."), h(2, "D"), p("z.")
    )
    paths = [c.heading_path for c in chunk_document(document, SMALL)]
    assert paths == [("A", "B"), ("A", "B", "C"), ("A", "D")]


def test_overlap_repeats_trailing_sentences_within_a_section() -> None:
    sentences = [f"Sentence number {i} is here." for i in range(6)]  # 27 chars each
    document = doc(h(1, "S"), p(" ".join(sentences)))
    chunks = chunk_document(document, SMALL)

    assert len(chunks) > 1
    for previous, current in pairwise(chunks):
        assert current.start < previous.end, "consecutive chunks in one section overlap"
        overlap = document.text[current.start : previous.end]
        assert len(overlap) <= SMALL.overlap_chars
        assert overlap.startswith("Sentence number"), "overlap starts on a sentence boundary"


def test_no_overlap_across_a_heading() -> None:
    document = doc(h(1, "One"), p("First section text."), h(1, "Two"), p("Second section text."))
    first, second = chunk_document(document, SMALL)
    assert second.start > first.end
    assert second.text.startswith("Two")


def test_zero_overlap_produces_disjoint_chunks() -> None:
    config = ChunkingConfig(target_chars=60, max_chars=100, overlap_chars=0)
    document = doc(p(" ".join(f"Sentence number {i} is here." for i in range(8))))
    chunks = chunk_document(document, config)
    assert all(b.start >= a.end for a, b in pairwise(chunks))


def test_overlap_never_pushes_a_chunk_past_max_chars() -> None:
    # The sentence after the boundary is long, so a full overlap would overflow.
    config = ChunkingConfig(target_chars=50, max_chars=90, overlap_chars=60)
    document = doc(p("Short opener here. Another short one. " + "Long " * 16 + "ending."))
    chunks = chunk_document(document, config)
    assert all(len(c.text) <= config.max_chars for c in chunks)


def test_oversized_sentence_is_split_at_whitespace() -> None:
    words = " ".join(["manuscript"] * 30)  # 329 chars, no sentence boundary
    chunks = chunk_document(doc(p(words + ".")), SMALL)

    assert all(len(c.text) <= SMALL.max_chars for c in chunks)
    assert all(not c.text.startswith(" ") and not c.text.endswith(" ") for c in chunks)
    assert " ".join(c.text for c in chunks) == words + "."


def test_long_code_blocks_split_on_line_boundaries() -> None:
    code = "\n".join(f"line_{i:02d} = {i}" for i in range(20))  # 13 chars a line
    chunks = chunk_document(doc(Block("code", code)), SMALL)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.text) <= SMALL.max_chars
        assert all(line.startswith("line_") for line in chunk.text.split("\n"))


def test_pages_cover_the_blocks_in_the_chunk() -> None:
    document = doc(p("On page three.", page=3), p("Still three.", page=3), p("Now four.", page=4))
    (chunk,) = chunk_document(
        document, ChunkingConfig(target_chars=200, max_chars=300, overlap_chars=0)
    )
    assert (chunk.page_start, chunk.page_end) == (3, 4)


def test_empty_document_has_no_chunks() -> None:
    assert chunk_document(doc(), SMALL) == []


def test_config_validation() -> None:
    with pytest.raises(ValueError, match="target_chars"):
        ChunkingConfig(target_chars=200, max_chars=100)
    with pytest.raises(ValueError, match="overlap_chars"):
        ChunkingConfig(target_chars=50, max_chars=100, overlap_chars=100)


# Properties ---------------------------------------------------------------------------

_PROSE_KINDS: tuple[BlockKind, ...] = ("paragraph", "list_item", "quote")
_words = st.text(alphabet="abcdefghij", min_size=1, max_size=12)


def _sentence(words: list[str]) -> str:
    return " ".join(words).capitalize() + "."


def _heading(level: int, title: str) -> Block:
    return Block("heading", title.capitalize(), level=level)


def _prose(kind: BlockKind, sentences: list[str]) -> Block:
    return Block(kind, " ".join(sentences))


def _code(lines: list[str]) -> Block:
    return Block("code", "\n".join(lines))


def _config(target: int, extra: int, overlap: int) -> ChunkingConfig:
    return ChunkingConfig(target, target + extra, min(overlap, target + extra - 1))


_sentences = st.lists(
    st.lists(_words, min_size=1, max_size=12).map(_sentence), min_size=1, max_size=8
)
_block = st.one_of(
    st.builds(_heading, st.integers(1, 4), _words),
    st.builds(_prose, st.sampled_from(_PROSE_KINDS), _sentences),
    st.builds(_code, st.lists(_words, min_size=1, max_size=15)),
)
_configs = st.builds(_config, st.integers(40, 300), st.integers(0, 300), st.integers(0, 200))


@settings(max_examples=150, deadline=None)
@given(st.lists(_block, max_size=25), _configs)
def test_chunks_are_exact_slices_that_cover_every_unit(
    blocks: list[Block], config: ChunkingConfig
) -> None:
    document = doc(*blocks)
    chunks = chunk_document(document, config)

    covered = [False] * len(document.text)
    for chunk in chunks:
        assert chunk.text == document.text[chunk.start : chunk.end]
        assert len(chunk.text) <= config.max_chars
        assert chunk.text == chunk.text.strip()
        for i in range(chunk.start, chunk.end):
            covered[i] = True
    for i, char in enumerate(document.text):
        if not char.isspace():
            assert covered[i], f"character {i} ({char!r}) is not in any chunk"

    starts = [c.start for c in chunks]
    assert starts == sorted(starts)
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


@settings(max_examples=150, deadline=None)
@given(st.lists(_block, max_size=25), _configs)
def test_chunks_never_straddle_a_heading(blocks: list[Block], config: ChunkingConfig) -> None:
    document = doc(*blocks)
    heading_starts = [b.start for b in document.blocks if b.kind == "heading"]
    for chunk in chunk_document(document, config):
        inside = [s for s in heading_starts if chunk.start < s < chunk.end]
        # A heading may appear inside a chunk only as part of a run of headings that
        # opens it (no content between the chunk start and that heading).
        for start in inside:
            before = document.text[chunk.start : start]
            content_before = [
                b for b in document.blocks if b.kind != "heading" and chunk.start <= b.start < start
            ]
            assert not content_before, f"content precedes a heading inside a chunk: {before!r}"
