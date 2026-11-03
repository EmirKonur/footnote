# ADR 0001: One SQLite file for documents, the search index, vectors, and chat

Status: accepted

## Context

Footnote needs to store four kinds of data: documents with their canonical text and
structure, chunks with a full-text index, an embedding per chunk, and chat history. It
runs on a laptop for one person or a small team, and the "try it in under a minute" goal
rules out anything that needs a separate server to be provisioned first.

Ingest has to be atomic per document. If a re-ingest replaces a document's chunks, the
FTS rows and vectors for the old chunks must disappear in the same step. Otherwise
search returns passages whose offsets point into text that no longer exists, and every
citation built on them is wrong.

## Decision

Everything lives in one SQLite database (`data/footnote.sqlite3`), in WAL mode:

- `documents` holds the canonical text and the block structure as JSON.
- `chunks` holds offsets into that text, the heading path, and pages.
- `chunks_fts` is an FTS5 external-content table kept in sync by triggers.
- `embeddings` stores float32 vectors as BLOBs, keyed by a hash of the embedded text.
- `conversations` and `messages` hold chat history, with each answer stored as JSON.

A document, its chunks, their FTS rows, and new vectors are written in a single
`BEGIN IMMEDIATE` transaction. Ingest runs on one worker thread, so there is one writer.

## Alternatives considered

**Postgres with pgvector.** It handles more concurrent writers and scales further, but it
adds a service to install and run before the first question can be asked. Footnote's
scale (see ADR 0002) does not need it.

**A dedicated vector database (Qdrant, Chroma, Weaviate) next to a relational store.**
That means two systems to keep consistent. Atomic replacement of a document's chunks
would need a two-phase write or a reconciliation job, which is exactly the failure mode
described above.

**Files on disk (JSON per document, a pickled index).** Simple until a crash during
re-ingest leaves the index and the text out of step.

## Consequences

- A collection is one file. Backing it up, or shipping a pre-built index, is a copy.
- Embeddings keyed by content hash are shared across documents and collections: a
  paragraph that appears in two files is embedded once.
- SQLite allows one writer at a time. That is fine for one ingest worker plus many
  readers, but a multi-tenant deployment with heavy concurrent uploads would need a
  different store.
- FTS5's BM25 cannot be tuned beyond column weights (no custom k1 or b), which the
  eval suite accepts as a fixed baseline.
