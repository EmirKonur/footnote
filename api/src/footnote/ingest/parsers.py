"""Format parsers: bytes in, structural blocks out.

Each parser keeps only what the chunker and the viewer need: block boundaries, heading
levels, and page numbers. Inline formatting is dropped on purpose. The canonical text
is what the model cites and what the viewer highlights, so it has to read as prose,
not as markup.
"""

from __future__ import annotations

import io
import re
from collections.abc import Iterator
from pathlib import PurePosixPath

from markdown_it import MarkdownIt
from markdown_it.token import Token
from pypdf import PdfReader
from pypdf.errors import PdfReadError
from selectolax.lexbor import LexborHTMLParser, LexborNode

from footnote.ingest.document import Block, BlockKind, ParsedDocument


class UnsupportedFormatError(ValueError):
    """The file extension is not one Footnote can ingest."""


class DocumentParseError(ValueError):
    """The file has a supported extension but its contents could not be read."""


SUPPORTED_SUFFIXES = (".md", ".markdown", ".html", ".htm", ".pdf", ".txt", ".text")


def parse_document(filename: str, data: bytes) -> ParsedDocument:
    suffix = PurePosixPath(filename).suffix.lower()
    fallback_title = _title_from_filename(filename)
    if suffix in (".md", ".markdown"):
        return parse_markdown(_decode(data), fallback_title)
    if suffix in (".html", ".htm"):
        return parse_html(_decode(data), fallback_title)
    if suffix == ".pdf":
        return parse_pdf(data, fallback_title)
    if suffix in (".txt", ".text"):
        return parse_text(_decode(data), fallback_title)
    raise UnsupportedFormatError(
        f"{filename}: unsupported file type (expected one of {', '.join(SUPPORTED_SUFFIXES)})"
    )


def _decode(data: bytes) -> str:
    return data.decode("utf-8-sig", errors="replace").replace("\r\n", "\n").replace("\r", "\n")


def _title_from_filename(filename: str) -> str:
    stem = PurePosixPath(filename).stem
    return re.sub(r"[-_]+", " ", stem).strip() or filename


def _collapse(text: str) -> str:
    return " ".join(text.split())


# Markdown ---------------------------------------------------------------------------

_FRONT_MATTER = re.compile(r"\A---\n.*?\n---\n", re.DOTALL)
_markdown = MarkdownIt("commonmark").enable("table")


def parse_markdown(source: str, fallback_title: str) -> ParsedDocument:
    tokens = _markdown.parse(_FRONT_MATTER.sub("", source))
    blocks: list[Block] = []
    containers: list[str] = []
    ordered_counters: list[int] = []
    pending_marker = ""
    row_cells: list[str] | None = None

    for index, token in enumerate(tokens):
        kind = token.type
        if kind in ("bullet_list_open", "blockquote_open"):
            containers.append(kind)
        elif kind == "ordered_list_open":
            containers.append(kind)
            ordered_counters.append(int(str(token.attrGet("start") or 1)))
        elif kind in ("bullet_list_close", "blockquote_close"):
            containers.pop()
        elif kind == "ordered_list_close":
            containers.pop()
            ordered_counters.pop()
        elif kind == "list_item_open" and containers and containers[-1] == "ordered_list_open":
            pending_marker = f"{ordered_counters[-1]}. "
            ordered_counters[-1] += 1
        elif kind == "heading_open":
            text = _inline_text(tokens[index + 1])
            blocks.append(Block("heading", text, level=int(token.tag[1])))
        elif kind == "paragraph_open":
            text = pending_marker + _inline_text(tokens[index + 1])
            pending_marker = ""
            blocks.append(Block(_markdown_paragraph_kind(containers), text))
        elif kind in ("fence", "code_block"):
            blocks.append(Block("code", token.content.rstrip("\n")))
        elif kind == "tr_open":
            row_cells = []
        elif kind == "inline" and row_cells is not None:
            row_cells.append(_inline_text(token))
        elif kind == "tr_close" and row_cells is not None:
            blocks.append(Block("row", " | ".join(row_cells)))
            row_cells = None

    return ParsedDocument(_first_heading(blocks) or fallback_title, "markdown", blocks)


def _markdown_paragraph_kind(containers: list[str]) -> BlockKind:
    if any(c in ("bullet_list_open", "ordered_list_open") for c in containers):
        return "list_item"
    if "blockquote_open" in containers:
        return "quote"
    return "paragraph"


def _inline_text(token: Token) -> str:
    parts: list[str] = []
    for child in token.children or []:
        if child.type in ("text", "code_inline"):
            parts.append(child.content)
        elif child.type == "softbreak":
            parts.append(" ")
        elif child.type == "hardbreak":
            parts.append("\n")
        elif child.type == "image":
            parts.append(child.content)
    return "".join(parts).strip()


def _first_heading(blocks: list[Block]) -> str | None:
    return next((b.text for b in blocks if b.kind == "heading" and b.level == 1), None)


# HTML -------------------------------------------------------------------------------

_SKIP_TAGS = frozenset(
    {
        "script",
        "style",
        "noscript",
        "template",
        "svg",
        "nav",
        "header",
        "footer",
        "aside",
        "form",
        "button",
        "iframe",
        "head",
        "select",
        "-comment",
    }
)
_HEADINGS = {f"h{n}": n for n in range(1, 7)}
_LEAF_KINDS: dict[str, BlockKind] = {
    **dict.fromkeys(_HEADINGS, "heading"),
    "p": "paragraph",
    "pre": "code",
    "li": "list_item",
    "dt": "term",
    "dd": "paragraph",
    "blockquote": "quote",
    "figcaption": "paragraph",
    "caption": "paragraph",
    "summary": "paragraph",
    "tr": "row",
}
_BLOCK_TAGS = frozenset(
    {
        *_LEAF_KINDS,
        "div",
        "section",
        "article",
        "main",
        "ul",
        "ol",
        "dl",
        "table",
        "thead",
        "tbody",
        "tfoot",
        "figure",
        "details",
        "body",
        "html",
        "address",
        "fieldset",
        "hr",
    }
)


def parse_html(source: str, fallback_title: str) -> ParsedDocument:
    tree = LexborHTMLParser(source)
    root = tree.css_first("main") or tree.css_first("article") or tree.body
    blocks: list[Block] = []
    if root is not None:
        _walk_html(root, blocks, context=None)
    title_node = tree.css_first("title")
    title = _collapse(title_node.text()) if title_node is not None else ""
    return ParsedDocument(title or _first_heading(blocks) or fallback_title, "html", blocks)


def _walk_html(node: LexborNode, out: list[Block], context: BlockKind | None) -> None:
    loose: list[str] = []

    def flush_loose() -> None:
        text = _collapse("".join(loose))
        loose.clear()
        if text:
            out.append(Block(context or "paragraph", text))

    for child in node.iter(include_text=True):
        tag = child.tag or ""
        if tag == "-text":
            loose.append(child.text_content or "")
        elif tag in _SKIP_TAGS:
            continue
        elif tag not in _BLOCK_TAGS:
            loose.append(child.text(deep=True))
        else:
            flush_loose()
            _emit_html_block(child, tag, out, context)
    flush_loose()


def _emit_html_block(
    node: LexborNode, tag: str, out: list[Block], context: BlockKind | None
) -> None:
    kind = _LEAF_KINDS.get(tag)
    if tag == "tr":
        cells = [_collapse(cell.text(deep=True)) for cell in node.css("td, th")]
        out.append(Block("row", " | ".join(c for c in cells if c)))
    elif tag == "pre":
        out.append(Block("code", (node.text(deep=True) or "").strip("\n")))
    elif kind is not None and not _has_block_descendant(node):
        text = _collapse(node.text(deep=True))
        level = _HEADINGS.get(tag, 0)
        # A paragraph inside a list item or a quote keeps the container's kind.
        effective = context if kind == "paragraph" and context is not None else kind
        out.append(Block(effective, text, level=level))
    else:
        inner = kind if kind in ("list_item", "quote") else context
        _walk_html(node, out, inner)


def _has_block_descendant(node: LexborNode) -> bool:
    descendants = node.traverse(include_text=False)
    next(descendants, None)  # traverse() yields the node itself first
    return any((descendant.tag or "") in _BLOCK_TAGS for descendant in descendants)


# PDF --------------------------------------------------------------------------------

# "build-\ning after" becomes "building\nafter": the word is rejoined but the line
# break stays, because line lengths are what the paragraph heuristic reads.
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w+)[ \t]*")
_SENTENCE_END = re.compile(r"[.!?:;]['\")\]]?$")


def parse_pdf(data: bytes, fallback_title: str) -> ParsedDocument:
    try:
        reader = PdfReader(io.BytesIO(data))
        pages = [page.extract_text() or "" for page in reader.pages]
    except (PdfReadError, ValueError, KeyError) as exc:
        raise DocumentParseError(f"could not read PDF: {exc}") from exc

    outline = _pdf_outline(reader)
    blocks: list[Block] = []
    for page_number, raw in enumerate(pages, start=1):
        headings = [(title, level) for title, page, level in outline if page == page_number]
        paragraphs = _pdf_paragraphs(raw)
        for title, level in headings:
            match = next((i for i, p in enumerate(paragraphs) if p.startswith(title)), None)
            if match is None:
                blocks.append(Block("heading", title, level=level, page=page_number))
                continue
            for paragraph in paragraphs[:match]:
                blocks.append(Block("paragraph", paragraph, page=page_number))
            blocks.append(Block("heading", title, level=level, page=page_number))
            rest = paragraphs[match][len(title) :].strip()
            paragraphs = ([rest] if rest else []) + paragraphs[match + 1 :]
        blocks.extend(Block("paragraph", p, page=page_number) for p in paragraphs)

    metadata_title = (reader.metadata.title if reader.metadata else None) or ""
    title = metadata_title.strip() or _first_heading(blocks) or fallback_title
    return ParsedDocument(title, "pdf", blocks)


def _pdf_outline(reader: PdfReader) -> list[tuple[str, int, int]]:
    """Flatten the bookmark tree into (title, 1-based page, level) triples."""
    entries: list[tuple[str, int, int]] = []

    def visit(items: list[object], level: int) -> None:
        for item in items:
            if isinstance(item, list):
                visit(item, level + 1)  # pyright: ignore[reportUnknownArgumentType]
                continue
            try:
                page = reader.get_destination_page_number(item)  # pyright: ignore[reportArgumentType]
                title = str(getattr(item, "title", "")).strip()
            except (KeyError, ValueError, AttributeError):
                continue
            if title and page is not None:
                entries.append((title, page + 1, min(level, 6)))

    try:
        visit(list(reader.outline), 1)
    except (PdfReadError, KeyError, ValueError):
        return []
    return entries


def _pdf_paragraphs(raw: str) -> list[str]:
    """Rebuild paragraphs from extracted lines.

    PDF text extraction yields one line per printed line. A blank line always ends a
    paragraph; so does a line that ends in terminal punctuation well short of the column
    width, which is how most typeset paragraphs end once the extractor has dropped the
    vertical gap between them.
    """
    text = _HYPHEN_BREAK.sub(r"\1\2\n", raw.replace("\r", ""))
    paragraphs: list[str] = []
    for chunk in re.split(r"\n\s*\n", text):
        lines = [line.strip() for line in chunk.split("\n") if line.strip()]
        if not lines:
            continue
        lengths = sorted(len(line) for line in lines)
        full_width = lengths[min(len(lengths) - 1, int(len(lengths) * 0.9))]
        current: list[str] = []
        for line in lines:
            current.append(line)
            if _SENTENCE_END.search(line) and len(line) < 0.75 * full_width:
                paragraphs.append(" ".join(current))
                current = []
        if current:
            paragraphs.append(" ".join(current))
    return paragraphs


# Plain text -------------------------------------------------------------------------

_UNDERLINE = re.compile(r"^(=+|-+|~+)$")


def parse_text(source: str, fallback_title: str) -> ParsedDocument:
    blocks: list[Block] = []
    for chunk in re.split(r"\n\s*\n", source):
        lines = chunk.strip("\n").split("\n")
        if not chunk.strip():
            continue
        blocks.extend(_text_blocks(lines))
    return ParsedDocument(_first_heading(blocks) or fallback_title, "text", blocks)


def _is_underlined(lines: list[str], i: int) -> bool:
    if i + 1 >= len(lines) or not lines[i].strip():
        return False
    rule = lines[i + 1].strip()
    return bool(_UNDERLINE.match(rule)) and len(rule) >= max(3, len(lines[i].strip()) // 2)


def _text_blocks(lines: list[str]) -> Iterator[Block]:
    """Recognise setext-style headings (a line underlined with = or -), which is how
    most plain-text documents mark sections."""
    buffer: list[str] = []
    i = 0
    while i < len(lines):
        if _is_underlined(lines, i):
            if buffer:
                yield Block("paragraph", "\n".join(buffer).strip())
                buffer = []
            level = 1 if lines[i + 1].strip().startswith("=") else 2
            yield Block("heading", lines[i].strip(), level=level)
            i += 2
            continue
        buffer.append(lines[i].rstrip())
        i += 1
    if buffer and "\n".join(buffer).strip():
        yield Block("paragraph", "\n".join(buffer).strip())
