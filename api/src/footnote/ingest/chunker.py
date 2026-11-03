"""Structure-aware chunking.

A chunk is always an exact slice ``text[start:end]`` of the canonical document. That is
the property everything downstream relies on: a citation offset inside a chunk plus the
chunk's ``start`` is an offset in the document the viewer renders.

The packer works on *units*: sentences for prose, line groups for oversized code, and
whole blocks otherwise. It

* starts a new chunk at every heading, so a chunk never straddles two sections
  (consecutive headings with nothing between them are kept together with the content
  that follows),
* prefers to end a chunk at a block boundary once it reaches ``target_chars``,
* never exceeds ``max_chars`` (single units longer than that are split at whitespace),
* repeats up to ``overlap_chars`` of trailing sentences at the start of the next chunk,
  but only when the next chunk continues the same section.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from footnote.ingest.document import CanonicalDocument, PlacedBlock


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    target_chars: int = 900
    max_chars: int = 1400
    overlap_chars: int = 220

    def __post_init__(self) -> None:
        if not 0 < self.target_chars <= self.max_chars:
            raise ValueError("need 0 < target_chars <= max_chars")
        if not 0 <= self.overlap_chars < self.max_chars:
            raise ValueError("need 0 <= overlap_chars < max_chars")


DEFAULT_CHUNKING = ChunkingConfig()


@dataclass(frozen=True, slots=True)
class Chunk:
    ordinal: int
    start: int
    end: int
    text: str
    heading_path: tuple[str, ...]
    page_start: int | None
    page_end: int | None


@dataclass(frozen=True, slots=True)
class _Unit:
    start: int
    end: int
    section: int
    is_heading: bool
    ends_block: bool
    page: int | None


def chunk_document(
    doc: CanonicalDocument, config: ChunkingConfig = DEFAULT_CHUNKING
) -> list[Chunk]:
    units, section_paths = _units(doc, config)
    chunks: list[Chunk] = []
    for ordinal, (first, last) in enumerate(_pack(units, config)):
        members = units[first:last]
        start, end = members[0].start, members[-1].end
        content_sections = [u.section for u in members if not u.is_heading]
        section = content_sections[0] if content_sections else members[-1].section
        pages = [u.page for u in members if u.page is not None]
        chunks.append(
            Chunk(
                ordinal=ordinal,
                start=start,
                end=end,
                text=doc.text[start:end],
                heading_path=section_paths[section],
                page_start=min(pages) if pages else None,
                page_end=max(pages) if pages else None,
            )
        )
    return chunks


def _units(
    doc: CanonicalDocument, config: ChunkingConfig
) -> tuple[list[_Unit], list[tuple[str, ...]]]:
    units: list[_Unit] = []
    section_paths: list[tuple[str, ...]] = [()]
    stack: list[tuple[int, str]] = []
    section = 0
    for block in doc.blocks:
        if block.kind == "heading":
            while stack and stack[-1][0] >= block.level:
                stack.pop()
            stack.append((block.level, doc.block_text(block)))
            section_paths.append(tuple(title for _, title in stack))
            section = len(section_paths) - 1
            units.append(_Unit(block.start, block.end, section, True, True, block.page))
            continue
        spans = _split_block(doc, block, config.max_chars)
        for index, (start, end) in enumerate(spans):
            units.append(_Unit(start, end, section, False, index == len(spans) - 1, block.page))
    return units, section_paths


def _split_block(
    doc: CanonicalDocument, block: PlacedBlock, max_chars: int
) -> list[tuple[int, int]]:
    text = doc.block_text(block)
    if block.kind == "code":
        local = [(0, len(text))] if len(text) <= max_chars else _line_groups(text, max_chars)
    else:
        local = sentence_spans(text)
    spans: list[tuple[int, int]] = []
    for start, end in local:
        spans.extend(_hard_split(text, start, end, max_chars))
    return [(block.start + s, block.start + e) for s, e in spans]


# A sentence ends at . ! or ? (optionally followed by a closing quote or bracket),
# followed by whitespace and something that can start a sentence. The lookbehind skips
# the abbreviations that show up most in technical prose.
_SENTENCE_BOUNDARY = re.compile(
    r"(?<!\be\.g)(?<!\bi\.e)(?<!\betc)(?<!\bvs)(?<!\bcf)(?<!\bMr)(?<!\bDr)(?<!\bNo)"
    r"[.!?][\"'”’)\]]?(?=\s+[\"'“‘(\[A-Z0-9])"  # noqa: RUF001 (curly quotes are intended)
)


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Split prose into sentence spans that exclude the whitespace between them."""
    spans: list[tuple[int, int]] = []
    start = _skip_space(text, 0)
    for match in _SENTENCE_BOUNDARY.finditer(text):
        end = match.end()
        if end > start:
            spans.append((start, end))
        start = _skip_space(text, end)
    if start < len(text) and text[start:].strip():
        spans.append((start, len(text.rstrip())))
    return spans


def _skip_space(text: str, index: int) -> int:
    while index < len(text) and text[index].isspace():
        index += 1
    return index


def _line_groups(text: str, max_chars: int) -> list[tuple[int, int]]:
    groups: list[tuple[int, int]] = []
    start = 0
    cursor = 0
    for line in text.splitlines(keepends=True):
        if cursor + len(line) - start > max_chars and cursor > start:
            groups.append((start, cursor - 1 if text[cursor - 1] == "\n" else cursor))
            start = cursor
        cursor += len(line)
    if cursor > start:
        groups.append((start, cursor))
    return [(s, e) for s, e in groups if text[s:e].strip()]


def _hard_split(text: str, start: int, end: int, max_chars: int) -> list[tuple[int, int]]:
    """Cut a single over-long unit at the last whitespace before the limit."""
    pieces: list[tuple[int, int]] = []
    while end - start > max_chars:
        cut = text.rfind(" ", start + 1, start + max_chars)
        if cut <= start:
            cut = start + max_chars
        pieces.append((start, cut))
        start = _skip_space(text, cut)
    if end > start:
        pieces.append((start, end))
    return pieces


def _pack(units: list[_Unit], config: ChunkingConfig) -> list[tuple[int, int]]:
    """Group units into chunks, returned as half-open unit index ranges."""
    ranges: list[tuple[int, int]] = []
    first = 0
    first_new = 0
    while first < len(units):
        last = max(first, first_new) + 1
        while last < len(units):
            candidate = units[last]
            has_content = any(not u.is_heading for u in units[first:last])
            if candidate.is_heading and has_content:
                break
            if candidate.end - units[first].start > config.max_chars:
                break
            length = units[last - 1].end - units[first].start
            if length >= config.target_chars and units[last - 1].ends_block and has_content:
                break
            last += 1
        ranges.append((first, last))
        if last >= len(units):
            break
        first_new = last
        first = _overlap_start(units, first, last, config)
    return ranges


def _overlap_start(units: list[_Unit], first: int, last: int, config: ChunkingConfig) -> int:
    if units[last].is_heading:
        return last
    start = last
    while start - 1 > first:
        candidate = units[start - 1]
        if candidate.is_heading or candidate.section != units[last].section:
            break
        if units[last - 1].end - candidate.start > config.overlap_chars:
            break
        if units[last].end - candidate.start > config.max_chars:
            break
        start -= 1
    return start
