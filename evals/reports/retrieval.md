# Retrieval eval

Generated 2026-11-03T16:00:49+00:00 on Darwin arm64, Python 3.12.4. 15 documents, 782 chunks (ingested in 39.5 s), 38 answerable golden questions.

| Mode | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR@10 | nDCG@5 | nDCG@10 | p50 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bm25 | 0.526 | 0.605 | 0.684 | 0.789 | 0.587 | 0.561 | 0.593 | 0.6 |
| dense | 0.500 | 0.711 | 0.868 | 0.947 | 0.639 | 0.630 | 0.656 | 5.2 |
| hybrid | 0.579 | 0.789 | 0.868 | 0.947 | 0.701 | 0.680 | 0.703 | 6.2 |
| dense-no-instruction | 0.526 | 0.658 | 0.842 | 0.947 | 0.640 | 0.622 | 0.656 | 4.5 |
| hybrid-no-instruction | 0.553 | 0.789 | 0.868 | 0.921 | 0.687 | 0.671 | 0.688 | 4.9 |

Recall@5 by question type:

| Mode | lookup | multi-passage | paraphrase |
| --- | --- | --- | --- |
| bm25 | 0.857 (n=14) | 0.800 (n=5) | 0.526 (n=19) |
| dense | 0.929 (n=14) | 1.000 (n=5) | 0.789 (n=19) |
| hybrid | 1.000 (n=14) | 1.000 (n=5) | 0.737 (n=19) |
| dense-no-instruction | 0.929 (n=14) | 1.000 (n=5) | 0.737 (n=19) |
| hybrid-no-instruction | 1.000 (n=14) | 1.000 (n=5) | 0.737 (n=19) |

Hybrid misses at k=5 (5):

- `pep343-protocol`: recall@5 0.00, first hit at rank >10
- `pep405-marker`: recall@5 0.00, first hit at rank >10
- `pep484-any`: recall@5 0.00, first hit at rank 8
- `pep557-mutable-default`: recall@5 0.00, first hit at rank 10
- `pep668-marker`: recall@5 0.00, first hit at rank 10

Variants:

- `bm25`: SQLite FTS5 BM25 only
- `dense`: bge-small-en-v1.5 cosine only
- `hybrid`: BM25 + dense, reciprocal rank fusion (k=60)
- `dense-no-instruction`: dense without the BGE query prefix
- `hybrid-no-instruction`: hybrid without the BGE query prefix
