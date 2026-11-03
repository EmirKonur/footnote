"""Test doubles: a deterministic embedder, scripted model streams, and a PDF builder."""

from __future__ import annotations

import hashlib
import io
import re
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager

import numpy as np
from pypdf import PdfReader, PdfWriter

from footnote.answer.generate import GenerationRequest, StreamEvent
from footnote.retrieval.embedder import Matrix, Vector, normalize_rows
from footnote.schemas import GenerationMode


class HashingEmbedder:
    """Bag-of-words feature hashing. Similar texts get similar vectors, which is all the
    retrieval tests need, and it never touches the network."""

    def __init__(self, dim: int = 256) -> None:
        self._dim = dim
        self.calls: list[list[str]] = []

    @property
    def model_name(self) -> str:
        return f"test-hashing-{self._dim}"

    def _vector(self, text: str) -> Vector:
        vector = np.zeros(self._dim, dtype=np.float32)
        for token in re.findall(r"\w+", text.lower()):
            bucket = int(hashlib.blake2b(token.encode(), digest_size=4).hexdigest(), 16)
            vector[bucket % self._dim] += 1.0
        return vector

    def embed_passages(self, texts: Sequence[str]) -> Matrix:
        self.calls.append(list(texts))
        return normalize_rows(np.stack([self._vector(t) for t in texts]))

    def embed_query(self, text: str) -> Vector:
        return normalize_rows(self._vector(text)[None, :])[0]


Script = Sequence[StreamEvent] | Callable[[GenerationRequest], Sequence[StreamEvent]]


class ScriptedStreamer:
    """Replays SDK stream events, optionally raising part way. ``events`` may be a
    function of the request, for scripts that cite whatever passages were retrieved."""

    def __init__(
        self,
        events: Script,
        *,
        raise_after: int | None = None,
        error: BaseException | None = None,
        model: str = "claude-opus-5-5",
    ) -> None:
        self._script = events
        self._raise_after = raise_after
        self._error = error
        self._model = model
        self.requests: list[GenerationRequest] = []
        self.closed = False

    @property
    def mode(self) -> GenerationMode:
        return "anthropic"

    @property
    def model(self) -> str:
        return self._model

    @asynccontextmanager
    async def open(self, request: GenerationRequest) -> AsyncGenerator[AsyncIterator[StreamEvent]]:
        self.requests.append(request)
        try:
            script = self._script
            yield self._iterate(list(script(request) if callable(script) else script))
        finally:
            self.closed = True

    async def _iterate(self, events: list[StreamEvent]) -> AsyncIterator[StreamEvent]:
        for position, event in enumerate(events):
            if self._raise_after is not None and position == self._raise_after and self._error:
                raise self._error
            yield event


def build_pdf(
    pages: Sequence[Sequence[str]],
    *,
    title: str | None = None,
    outline: Sequence[tuple[str, int]] = (),
) -> bytes:
    """Write a minimal text PDF: one Helvetica text object per page, one line per entry
    (an empty string leaves a blank line). ``outline`` adds (title, 0-based page)
    bookmarks."""
    objects: list[bytes] = []

    def add(body: bytes) -> int:
        objects.append(body)
        return len(objects)

    font = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    pages_index = add(b"")  # placeholder, filled in once the kids are known
    kids: list[int] = []
    for lines in pages:
        escaped = [
            line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)") for line in lines
        ]
        ops = ["BT", "/F1 11 Tf", "14 TL", "72 740 Td"]
        for line in escaped:
            ops.append(f"({line}) Tj T*" if line else "T*")
        ops.append("ET")
        stream = "\n".join(ops).encode("latin-1")
        content = add(b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream))
        kids.append(
            add(
                b"<< /Type /Page /Parent %d 0 R /MediaBox [0 0 612 792] "
                b"/Resources << /Font << /F1 %d 0 R >> >> /Contents %d 0 R >>"
                % (pages_index, font, content)
            )
        )
    objects[pages_index - 1] = b"<< /Type /Pages /Kids [%s] /Count %d >>" % (
        b" ".join(b"%d 0 R" % k for k in kids),
        len(kids),
    )
    catalog = add(b"<< /Type /Catalog /Pages %d 0 R >>" % pages_index)

    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets: list[int] = []
    for number, body in enumerate(objects, start=1):
        offsets.append(out.tell())
        out.write(b"%d 0 obj\n%s\nendobj\n" % (number, body))
    xref = out.tell()
    out.write(b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1))
    for offset in offsets:
        out.write(b"%010d 00000 n \n" % offset)
    out.write(
        b"trailer\n<< /Size %d /Root %d 0 R >>\nstartxref\n%d\n%%%%EOF\n"
        % (len(objects) + 1, catalog, xref)
    )

    writer = PdfWriter(clone_from=PdfReader(io.BytesIO(out.getvalue())))
    for heading, page in outline:
        writer.add_outline_item(heading, page)
    if title:
        writer.add_metadata({"/Title": title})
    final = io.BytesIO()
    writer.write(final)
    return final.getvalue()
