"""Map API citations onto Footnote's documents and check them.

A ``char_location`` citation says: in document N of the request, characters
[start, end) support this text. Document N is passage N, whose text is the chunk
``text[chunk.start:chunk.end]`` of a canonical document. So the location in the
viewer is ``chunk.start + start``.

Footnote does not take the offsets on trust. It compares ``cited_text`` with the stored
text at those offsets. If they disagree it searches the passage for the quote, and if
the quote is not there at all the footnote is shown as unverified: the quote is
displayed, but it does not link to a location it cannot vouch for.
"""

from __future__ import annotations

from collections.abc import Sequence

from anthropic.types.beta import BetaCitationCharLocation
from anthropic.types.beta.beta_text_citation import BetaTextCitation

from footnote.log import get_logger
from footnote.schemas import Footnote, Passage

log = get_logger(__name__)


def _squash(text: str) -> str:
    return " ".join(text.split())


class FootnoteBook:
    """Numbers citations in order of first appearance and reuses a number when two
    citations point at the same span."""

    def __init__(self, passages: Sequence[Passage]) -> None:
        self._passages = list(passages)
        self._numbers: dict[tuple[str, int | None, int | None, str], int] = {}
        self.footnotes: list[Footnote] = []

    def resolve(self, citation: BetaTextCitation) -> Footnote | None:
        if not isinstance(citation, BetaCitationCharLocation):
            # Footnote only sends plain-text documents, so other location types mean the
            # request changed shape without this mapper being updated.
            log.warning("unexpected citation type", type=citation.type)
            return None
        if not 0 <= citation.document_index < len(self._passages):
            log.warning("citation document index out of range", index=citation.document_index)
            return None
        passage = self._passages[citation.document_index]
        span = _locate(passage.text, citation)
        start = passage.start + span[0] if span else None
        end = passage.start + span[1] if span else None
        key = (passage.document_id, start, end, "" if span else _squash(citation.cited_text))
        number = self._numbers.get(key)
        if number is not None:
            return self.footnotes[number - 1]
        footnote = Footnote(
            n=len(self.footnotes) + 1,
            document_id=passage.document_id,
            document_title=passage.document_title,
            document_path=passage.document_path,
            chunk_id=passage.chunk_id,
            heading_path=passage.heading_path,
            page=_page(passage),
            start=start,
            end=end,
            cited_text=citation.cited_text.strip(),
            verified=span is not None,
        )
        self._numbers[key] = footnote.n
        self.footnotes.append(footnote)
        return footnote


def _locate(text: str, citation: BetaCitationCharLocation) -> tuple[int, int] | None:
    start = max(0, min(citation.start_char_index, len(text)))
    end = max(start, min(citation.end_char_index, len(text)))
    quote = citation.cited_text
    if _squash(text[start:end]) == _squash(quote) and quote.strip():
        # Trim the whitespace the API includes after a sentence so highlights end on
        # the last character of the quote.
        while end > start and text[end - 1].isspace():
            end -= 1
        while start < end and text[start].isspace():
            start += 1
        return start, end
    stripped = quote.strip()
    if not stripped:
        return None
    found = text.find(stripped)
    if found >= 0:
        return found, found + len(stripped)
    return None


def _page(passage: Passage) -> int | None:
    # A chunk that spans a page break can't say which page a sentence inside it is on,
    # so only single-page chunks report a page.
    if passage.page_start is not None and passage.page_start == passage.page_end:
        return passage.page_start
    return None
