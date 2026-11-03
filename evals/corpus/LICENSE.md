# Corpus license

The files in `peps/` are Python Enhancement Proposals from <https://peps.python.org>,
fetched on 2026-11-03 by `fetch_peps.py`. The script keeps each page's article body and
strips markup that carries no text (syntax highlighting spans, the table of contents,
permalinks, attributes other than `href`). The prose is not edited.

Each PEP states its own license in its Copyright section, and `fetch_peps.py` refuses to
bundle a PEP whose statement is anything else:

| PEP | Statement |
| --- | --- |
| 8, 20, 257, 343, 405, 484, 492, 498, 557, 572 | "This document has been placed in the public domain." |
| 1, 621, 634, 668, 703 | "This document is placed in the public domain or under the CC0-1.0-Universal license, whichever is more permissive." |

The golden question set in `../golden.jsonl` and the reports in `../reports/` are part of
this repository and fall under its MIT license.
