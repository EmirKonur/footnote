"""Lexical retrieval through SQLite FTS5's built-in BM25."""

from __future__ import annotations

import re
import sqlite3

# Question words and glue. FTS5's IDF already discounts common terms, but dropping these
# keeps "what does the PEP say about X" from matching every chunk that contains "the".
_STOPWORDS = frozenset(
    """
    a an and are as at be by can could do does for from had has have how i if in into is it
    its me my not of on or should so than that the their them then there these they this to
    was we were what when where which who why will with would you your about
    """.split()  # noqa: SIM905 (a word list reads better than sixty quoted strings)
)
_TOKEN = re.compile(r"\w+", re.UNICODE)

# Weight of each FTS column, in declaration order (text, heading_path). Heading matches
# count, but less than body matches, so a section titled "Imports" doesn't outrank the
# paragraph that actually answers an import question.
_COLUMN_WEIGHTS = (1.0, 0.5)


def fts_query(text: str) -> str | None:
    """Turn free text into an FTS5 OR-query of quoted terms (quoting disables operators)."""
    tokens = [t.lower() for t in _TOKEN.findall(text)]
    terms = [t for t in tokens if t not in _STOPWORDS] or tokens
    unique = list(dict.fromkeys(terms))
    if not unique:
        return None
    return " OR ".join(f'"{term}"' for term in unique)


def search_bm25(
    conn: sqlite3.Connection, collection_id: str, query: str, limit: int
) -> list[tuple[int, float]]:
    """Return up to ``limit`` (chunk_id, score) pairs, best first; higher is better."""
    match = fts_query(query)
    if match is None or limit <= 0:
        return []
    weights = ", ".join(str(w) for w in _COLUMN_WEIGHTS)
    rows = conn.execute(
        f"""
        SELECT chunks.id AS id, bm25(chunks_fts, {weights}) AS score
        FROM chunks_fts JOIN chunks ON chunks.id = chunks_fts.rowid
        WHERE chunks_fts MATCH ? AND chunks.collection_id = ?
        ORDER BY score, chunks.id
        LIMIT ?
        """,  # noqa: S608 (weights are module constants, not input)
        (match, collection_id, limit),
    ).fetchall()
    # FTS5 reports BM25 as a negative number where more negative means more relevant.
    return [(int(r["id"]), -float(r["score"])) for r in rows]
