"""``footnote`` command line: serve the API, ingest folders, search, and run evals."""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path

from footnote import library
from footnote.config import Settings
from footnote.db import connect, migrate
from footnote.evals.corpus import CORPUS_DIR, EVALS_DIR
from footnote.ingest.parsers import SUPPORTED_SUFFIXES
from footnote.ingest.pipeline import ingest_file, prune_missing
from footnote.log import configure_logging
from footnote.retrieval.embedder import FastEmbedder
from footnote.retrieval.search import Retriever
from footnote.retrieval.vectors import VectorIndex

DEMO_COLLECTION = "Python PEPs"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="footnote", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    serve = commands.add_parser("serve", help="run the HTTP API")
    serve.add_argument("--reload", action="store_true", help="restart on code changes")

    ingest = commands.add_parser("ingest", help="ingest a folder into a collection")
    ingest.add_argument("folder", type=Path)
    ingest.add_argument("--collection", required=True, help="collection name (created if missing)")
    ingest.add_argument(
        "--prune", action="store_true", help="remove documents no longer in the folder"
    )

    commands.add_parser("seed", help=f"ingest the bundled PEP corpus as {DEMO_COLLECTION!r}")

    search = commands.add_parser("search", help="print ranked passages for a query")
    search.add_argument("query")
    search.add_argument("--collection", required=True)
    search.add_argument("--mode", choices=["bm25", "dense", "hybrid"], default="hybrid")
    search.add_argument("-k", type=int, default=5)

    evals = commands.add_parser("eval", help="run the evaluation suite")
    eval_kind = evals.add_subparsers(dest="suite", required=True)
    retrieval = eval_kind.add_parser("retrieval", help="offline retrieval metrics (no API key)")
    generation = eval_kind.add_parser("generation", help="LLM-judged answer metrics (needs a key)")
    for sub in (retrieval, generation):
        sub.add_argument("--out", type=Path, default=EVALS_DIR / "reports")
        sub.add_argument(
            "--fail-under",
            action="append",
            default=[],
            metavar="METRIC=VALUE",
            help="exit 1 if METRIC (e.g. hybrid.recall@5) is below VALUE; repeatable",
        )
    retrieval.add_argument(
        "--baseline",
        type=Path,
        help="exit 1 if any metric drops more than --tolerance below this report",
    )
    retrieval.add_argument("--tolerance", type=float, default=0.02)
    generation.add_argument("--limit", type=int, help="only judge the first N golden questions")

    args = parser.parse_args(argv)
    settings = Settings()
    configure_logging(settings.log_level, settings.log_format)

    if args.command == "serve":
        return _serve(settings, reload=args.reload)
    if args.command == "ingest":
        return _ingest(settings, args.folder, args.collection, prune=args.prune)
    if args.command == "seed":
        return _ingest(settings, CORPUS_DIR, DEMO_COLLECTION, prune=False)
    if args.command == "search":
        return _search(settings, args.query, args.collection, args.mode, args.k)
    if args.suite == "retrieval":
        from footnote.evals.retrieval_run import run_retrieval_eval

        return run_retrieval_eval(
            settings, args.out, args.fail_under, args.baseline, args.tolerance
        )
    from footnote.evals.generation_run import run_generation_eval

    return asyncio.run(run_generation_eval(settings, args.out, args.fail_under, args.limit))


def _serve(settings: Settings, *, reload: bool) -> int:
    import uvicorn

    uvicorn.run(
        "footnote.server.app:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
        reload=reload,
        log_level=settings.log_level.lower(),
    )
    return 0


def _ingest(settings: Settings, folder: Path, name: str, *, prune: bool) -> int:
    if not folder.is_dir():
        print(f"not a folder: {folder}", file=sys.stderr)
        return 2
    conn = connect(settings.db_path)
    migrate(conn)
    collection = library.find_collection_by_name(conn, name) or library.create_collection(
        conn, name
    )
    embedder = FastEmbedder(
        settings.embedding_model,
        settings.model_cache_dir,
        query_instruction=settings.query_instruction,
    )
    files = sorted(
        p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    )
    counts = {"added": 0, "updated": 0, "unchanged": 0, "failed": 0}
    for path in files:
        relative = path.relative_to(folder).as_posix()
        try:
            outcome = ingest_file(conn, embedder, collection.id, relative, path.read_bytes())
        except ValueError as exc:
            counts["failed"] += 1
            print(f"  failed     {relative}: {exc}", file=sys.stderr)
            continue
        counts[outcome.status] += 1
        print(
            f"  {outcome.status:<10} {relative} "
            f"({outcome.chunks} chunks, {outcome.embedded} embedded, {outcome.reused} reused)"
        )
    if prune:
        for removed in prune_missing(
            conn, collection.id, {p.relative_to(folder).as_posix() for p in files}
        ):
            print(f"  removed    {removed}")
    summary = ", ".join(f"{n} {status}" for status, n in counts.items() if n)
    print(f"{collection.name}: {summary or 'no supported files found'}")
    return 1 if counts["failed"] else 0


def _search(settings: Settings, query: str, name: str, mode: str, k: int) -> int:
    conn = connect(settings.db_path)
    migrate(conn)
    collection = library.find_collection_by_name(conn, name)
    if collection is None:
        print(f"no collection named {name!r}", file=sys.stderr)
        return 2
    embedder = FastEmbedder(
        settings.embedding_model,
        settings.model_cache_dir,
        query_instruction=settings.query_instruction,
    )
    retriever = Retriever(embedder, VectorIndex(), settings.candidate_pool, settings.rrf_k)
    assert mode in ("bm25", "dense", "hybrid")
    for rank, passage in enumerate(retriever.search(conn, collection.id, query, k, mode), start=1):
        where = " > ".join(passage.heading_path) or passage.document_title
        ranks = " ".join(f"{name}#{r}" for name, r in passage.ranks.items())
        print(f"{rank}. {where}  [{ranks}]")
        print(f"   {' '.join(passage.text.split())[:220]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
