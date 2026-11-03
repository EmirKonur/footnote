"""Rebuild the bundled PEP corpus from peps.python.org.

Run from the repo root:

    uv run --project api python evals/corpus/fetch_peps.py

Each page is reduced to its article body: syntax-highlighting spans, the table of
contents, header permalinks, and attributes are dropped so the files stay small and
diff cleanly. The prose itself is not edited. The script refuses to write a PEP whose
copyright section does not place it in the public domain (or CC0), because the
license note in LICENSE.md depends on that.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

from selectolax.lexbor import LexborHTMLParser

PEPS = [1, 8, 20, 257, 343, 405, 484, 492, 498, 557, 572, 621, 634, 668, 703]
OUT = Path(__file__).parent / "peps"
DROP = "details, a.headerlink, a.footnote-reference, script, style, hr, nav, .admonition-title"
KEEP_ATTRS = {"href"}


def fetch(number: int) -> str:
    url = f"https://peps.python.org/pep-{number:04d}/"
    request = urllib.request.Request(url, headers={"User-Agent": "footnote-corpus/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310 (fixed https host)
        return response.read().decode("utf-8")


def clean(html: str, number: int) -> str:
    tree = LexborHTMLParser(html)
    article = tree.css_first("section#pep-content")
    if article is None:
        raise SystemExit(f"PEP {number}: no #pep-content section")
    copyright_text = " ".join(node.text() for node in article.css("section#copyright p")).lower()
    if "public domain" not in copyright_text and "cc0" not in copyright_text:
        raise SystemExit(f"PEP {number}: not public domain / CC0, refusing to bundle")
    for node in article.css(DROP):
        node.decompose()
    for span in article.css("span, abbr"):
        span.unwrap()
    for empty in article.css("span"):
        empty.decompose()
    for node in article.traverse():
        for name in list(node.attributes):
            if name not in KEEP_ATTRS:
                del node.attrs[name]
    title_node = article.css_first("h1")
    title = title_node.text(strip=True) if title_node else f"PEP {number}"
    body = article.html or ""
    source = f"https://peps.python.org/pep-{number:04d}/"
    return (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        f'<title>{title}</title>\n<link rel="canonical" href="{source}">\n'
        f"</head>\n<body>\n<main>\n{body}\n</main>\n</body>\n</html>\n"
    )


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for number in PEPS:
        cleaned = clean(fetch(number), number)
        path = OUT / f"pep-{number:04d}.html"
        path.write_text(cleaned, encoding="utf-8")
        print(f"{path.name}: {len(cleaned) // 1024} KiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
