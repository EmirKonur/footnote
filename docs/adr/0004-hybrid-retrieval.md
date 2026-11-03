# ADR 0004: Hybrid retrieval: FTS5 BM25 plus local BGE embeddings, fused with RRF

Status: accepted

## Context

Questions about documents come in two shapes. Lookups use the document's own words
("What is the maximum line length?"), and lexical matching handles them well.
Paraphrases do not ("How long can a line of Python code be?"), and they need semantic
matching. The eval suite tags its golden questions this way, so the trade-off can be
measured instead of argued.

Retrieval-only mode (no API key) is a first-class feature, so embeddings cannot depend
on a paid API.

## Decision

- **Lexical:** SQLite FTS5 with the `porter unicode61` tokenizer. Queries are rewritten
  to an OR of quoted terms with a short stopword list removed. Quoting matters: it stops
  user text from being parsed as FTS5 operators (`NEAR`, `*`, `AND`). The body column
  is weighted 1.0 and the heading path 0.5.
- **Dense:** `BAAI/bge-small-en-v1.5` through fastembed's ONNX runtime, run locally (a
  64 MB download on first use, then no network). Passages are embedded with their
  document title and section path prefixed, so a chunk like "Use 4 spaces per
  indentation level." still carries the context that it is about Python layout. Queries
  get BGE's retrieval instruction prefix.
- **Fusion:** reciprocal rank fusion with k = 60 over the top 50 candidates from each
  retriever. Scores are summed with `math.fsum` (a property test found that plain `+=`
  made the result depend on input order).

## Results

From `make eval` (38 answerable golden questions, 782 chunks, Apple Silicon laptop):

| Mode | Recall@5 | MRR@10 | nDCG@10 |
| --- | --- | --- | --- |
| BM25 | 0.684 | 0.587 | 0.593 |
| Dense | 0.868 | 0.639 | 0.656 |
| Hybrid | 0.868 | 0.701 | 0.703 |

Hybrid's recall@5 matches dense, but it ranks the right passage higher: MRR rises from
0.639 to 0.701, and recall@1 from 0.500 to 0.579. By question type, hybrid reaches
1.000 recall@5 on lookups (BM25 0.857, dense 0.929) and trails dense on paraphrases
(0.737 against 0.789). The paraphrase gap is one question, `pep405-marker`: dense ranks
the right chunk third, BM25 doesn't retrieve it in its top ten at all, and chunks that
both retrievers liked push it out of hybrid's top ten. The reverse happens on
`pep668-override`, a lookup, which dense ranks seventh, BM25 first, and hybrid second.
The query instruction prefix is worth 0.026 recall@5 for dense alone (0.868 against
0.842).

## Alternatives considered

**Weighted score fusion** (normalise BM25 and cosine, then add). BM25 scores are
unbounded and depend on the query, so normalisation is fragile; RRF needs only ranks.

**A cross-encoder reranker.** It would likely help the paraphrase misses, but it adds a
second model and roughly 10x the per-query latency. It is on the roadmap, gated on the
eval showing a gain.

**API embeddings.** Better models exist, but they would make retrieval-only mode
require a key, and they would put a network call on the ingest path.

## Consequences

- Retrieval works offline and is reproducible in CI, where the eval runs on every push.
- The paraphrase gap is visible in the committed report rather than hidden in an
  average.
- Changing the embedding model invalidates stored vectors. They are keyed by model
  name, so a new model re-embeds on the next ingest without corrupting the old ones.
