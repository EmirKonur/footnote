# The eval suite

Everything lives under `evals/`:

- `corpus/peps/`: 15 public-domain PEPs as cleaned HTML (license in
  `corpus/LICENSE.md`, rebuild with `corpus/fetch_peps.py`);
- `golden.jsonl`: 46 questions: 38 answerable, each with verbatim expected quotes and a
  reference answer, and 8 the corpus cannot answer;
- `reports/`: the committed retrieval report (`retrieval.md`, `retrieval.json`).

The code is in `api/src/footnote/evals/`. The design is in
[ADR 0005](adr/0005-eval-design.md).

## Retrieval (offline, runs in CI)

```sh
make eval            # or: cd api && uv run footnote eval retrieval
```

This ingests the corpus into a temporary database with the local embedding model (about
40 s on an Apple Silicon laptop; it downloads 64 MB the first time) and searches every
answerable question in five configurations: `bm25`, `dense`, `hybrid`, and dense and
hybrid without the BGE query prefix. It writes `evals/reports/retrieval.md` and
`retrieval.json`.

The metrics, per question, then averaged:

- **recall@k**: the fraction of expected passages covered by the top k chunks. A chunk
  covers a passage when it overlaps at least half of the quote's characters.
- **MRR@10**: one over the rank of the first chunk that covers anything, or 0.
- **nDCG@k**: chunks that cover a new expected passage earn gain 1, discounted by
  log2(rank + 1); duplicates earn nothing, so overlapping chunks are not double counted.

### Gating CI

```sh
make eval-gate
```

This is the same run with `--baseline evals/reports/retrieval.json --tolerance 0.03
--fail-under hybrid.recall@5=0.8`. It exits 1 if any quality metric of any mode falls
more than 0.03 below the committed report, or if hybrid recall@5 drops under 0.8.
Latency is reported but never gated. `--fail-under` takes any dotted metric path and
can be repeated; a path that doesn't exist is an error, so a typo can't switch the gate
off.

When a change is meant to move the numbers, rerun `make eval` and commit the new
report together with the change.

## Generation (needs a key, costs money)

```sh
export ANTHROPIC_API_KEY=...
make eval-generation                      # all 46 questions
cd api && uv run footnote eval generation --limit 5 --fail-under faithfulness=0.9
```

For each question it runs the app's own hybrid retrieval and answer path
(`FOOTNOTE_MODEL`, `FOOTNOTE_EFFORT`), then asks the judge (`FOOTNOTE_JUDGE_MODEL`,
default `claude-opus-5-5`) to label the answer. That is two API calls per question. It
writes `evals/reports/generation.md` and `generation.json`.

The judge labels every answer segment (is it a claim, is it supported by the passages)
and every footnote (does its quote support its segment), and says whether the answer
abstained and how it compares with the reference. Footnote computes:

| Metric | Definition | Averaged over |
| --- | --- | --- |
| Faithfulness | supported claims / claims | answerable questions that were answered |
| Citation support | claims with at least one supporting footnote / claims | same |
| Citation precision | supporting footnotes / footnotes | same |
| Answer correctness | correct 1, partially correct 0.5, incorrect 0 | answerable questions |
| Abstention (unanswerable) | share of unanswerable questions the answer declined | unanswerable questions |
| False abstention | share of answerable questions the answer declined | answerable questions |

Answers that were refused or failed, and cases the judge could not grade (refusal,
`max_tokens`, unparseable output), are reported per question and left out of the
averages. They are counted in the report header so they can't disappear quietly.

The judge runs with structured outputs (`client.beta.messages.parse` with a Pydantic
schema). That is allowed because the judge request carries no citation-enabled
documents; citations and structured outputs cannot share a request.

The generation metrics, the judge's error handling, and the end-to-end loop are tested
against a mocked client in `api/tests/test_evals.py`. No real API call is made in the
test suite. The generation eval has not been run for this repository's committed
reports.
