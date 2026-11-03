# ADR 0002: Exact search with numpy over vectors stored in SQLite

Status: accepted

## Context

Dense retrieval needs nearest neighbours of a query vector among a collection's chunk
vectors (384 dimensions, bge-small-en-v1.5). Collections are personal or team document
sets: the bundled PEP corpus is 15 documents and 782 chunks, and a large shelf might
hold tens of thousands of chunks.

Two properties matter more than raw speed. The search should be exact, because the eval
suite compares retrieval modes and an approximate index would add its own recall loss to
the numbers. And vectors should live in the same transaction as their chunks (ADR 0001).

## Decision

Vectors are stored as little-endian float32 BLOBs in the `embeddings` table. On the first
dense query for a collection, `VectorIndex` loads them into one normalised float32
matrix and caches it, tagged with the collection's `revision` counter. Every ingest or
delete bumps the revision, and the next query reloads. A query is one matrix-vector
product plus `argpartition` for the top candidates.

Measured with `make bench` (synthetic unit vectors, 384 dimensions, 200 queries, top 50)
on an Apple Silicon laptop, macOS arm64, Python 3.12.4, numpy 2.5.4:

| Chunks | Cold load | Query p50 | Query p95 | Matrix size |
| --- | --- | --- | --- | --- |
| 1,000 | 3.9 ms | 0.05 ms | 0.06 ms | 1.5 MB |
| 10,000 | 30.2 ms | 0.22 ms | 0.28 ms | 15.4 MB |
| 100,000 | 396.1 ms | 3.49 ms | 4.29 ms | 153.6 MB |

Embedding the query with the local model costs about 4 to 5 ms (the dense p50 in the
retrieval eval is 5.2 ms per query), so up to around 100,000 chunks the vector math is
not the bottleneck.

## Alternatives considered

**sqlite-vec.** Keeps search inside SQL, which is attractive. But it is a loadable
extension, and Python's `sqlite3` module can only load extensions when the interpreter
was built with `enable_load_extension` support. Several common Python builds disable it,
so "works on a fresh clone" would depend on how Python was installed. At this scale its
brute-force scan is not faster than numpy.

**FAISS or hnswlib (HNSW).** Approximate indexes pay off at millions of vectors. Below
that they add a native dependency, an index file that has to be kept in step with
SQLite, and a recall/latency knob that would muddy the eval comparison.

**LanceDB or Chroma.** Both bring their own storage, which means two sources of truth
(see ADR 0001).

## Consequences

- Results are exact and deterministic. Ties are broken by chunk id, so the eval report
  is reproducible.
- Memory grows linearly: about 1.5 KB per chunk, so 150 MB at 100,000 chunks per
  collection kept warm.
- The cold load after an ingest is O(n), about 0.4 s at 100,000 chunks. Past a few
  hundred thousand chunks, the cache would need to become incremental, or the search
  would move to an ANN index. That threshold is on the roadmap rather than built ahead
  of need.
