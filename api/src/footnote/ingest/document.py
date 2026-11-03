"""The format-independent document model.

Every parser emits a flat list of structural blocks. ``assemble`` joins them into one
canonical text and records where each block landed. Chunks, citations, and the viewer
all address the document through offsets into that canonical text, so a highlight in
the browser and a citation from the model are the same pair of integers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

BlockKind = Literal["heading", "paragraph", "list_item", "code", "quote", "row", "term"]
DocFormat = Literal["markdown", "html", "pdf", "text"]

BLOCK_SEPARATOR = "\n\n"


@dataclass(frozen=True, slots=True)
class Block:
    kind: BlockKind
    text: str
    level: int = 0
    """Heading level 1-6; 0 for every other kind."""
    page: int | None = None
    """1-based page number, for formats that have pages."""


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    title: str
    format: DocFormat
    blocks: list[Block]


@dataclass(frozen=True, slots=True)
class PlacedBlock:
    kind: BlockKind
    start: int
    end: int
    level: int
    page: int | None


@dataclass(frozen=True, slots=True)
class CanonicalDocument:
    title: str
    format: DocFormat
    text: str
    blocks: tuple[PlacedBlock, ...]

    def block_text(self, block: PlacedBlock) -> str:
        return self.text[block.start : block.end]


def assemble(parsed: ParsedDocument) -> CanonicalDocument:
    """Join blocks with a blank line between them, dropping blocks with no text."""
    parts: list[str] = []
    placed: list[PlacedBlock] = []
    cursor = 0
    for block in parsed.blocks:
        text = block.text.strip("\n")
        if not text.strip():
            continue
        if parts:
            parts.append(BLOCK_SEPARATOR)
            cursor += len(BLOCK_SEPARATOR)
        parts.append(text)
        placed.append(PlacedBlock(block.kind, cursor, cursor + len(text), block.level, block.page))
        cursor += len(text)
    return CanonicalDocument(parsed.title, parsed.format, "".join(parts), tuple(placed))
