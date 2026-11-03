# Contributing

## Setup

Python 3.12 with [uv](https://docs.astral.sh/uv/), Node 23 or later, pnpm 9.

```sh
make install
make seed            # optional: the bundled PEP corpus as a collection
make dev             # API :4410 with reload, web :4411
```

`make help` lists every target. `FOOTNOTE_FAKE_LLM=fixtures make dev` gives cited
answers without an API key (labeled as fixtures in the UI).

## Before opening a pull request

```sh
make check           # lint, strict types, unit and integration tests, production build
make e2e             # when the UI or an API response shape changed
make eval-gate       # when parsing, chunking, or retrieval changed
```

If a change is supposed to move retrieval quality, run `make eval` and commit the new
`evals/reports/retrieval.{md,json}` in the same pull request, with a sentence on why the
numbers moved. Don't edit the golden set to make a change look better; if a label is
wrong, fix it in a separate commit that says so.

## Conventions

- Python: ruff (lint and format), pyright in strict mode, Pydantic models at the
  boundaries. Tests live in `api/tests`, one file per module area.
- TypeScript: `strict` with `noUncheckedIndexedAccess`; logic that can be pure lives in
  `web/src/lib` and is covered by Vitest.
- Tests never call the Anthropic API. Use `tests/fakes.py` (`ScriptedStreamer`, the
  event builders in `footnote.answer.fixtures`) to script a stream.
- Offsets are always into a document's canonical text. Anything that changes the
  canonical text (a parser change) changes every stored offset, so it needs a re-ingest
  and an eval run.
- Comments explain why. Commit messages say what changed and why, in the imperative.
