# ADR 0005: A golden set of quotes, and a judge that labels segments

Status: accepted

## Context

The eval suite has to answer two questions: did retrieval find the right passage, and
is the answer grounded in what was retrieved? Both need to survive changes to the
system they measure. A golden set keyed to chunk ids breaks the moment the chunker
changes, and that is precisely when the eval matters most.

## Decision

**Expected passages are verbatim quotes.** Each golden item lists `(document, quote)`
pairs. At eval time, each quote is located in the document's canonical text (tolerating
whitespace differences). A retrieved chunk *covers* a quote when it overlaps at least
half of the quote's characters. A quote that cannot be found fails the run, because a
silent miss would turn a parser regression into a lower score instead of an error.

**Retrieval metrics are passage-level.** recall@k counts expected passages covered in
the top k; MRR is the reciprocal rank of the first covering chunk; nDCG gives gain only
to chunks that cover a passage no earlier chunk covered. Without that last rule, two
overlapping chunks containing the same sentence would score as two relevant results.

**The judge labels segments, Footnote computes the scores.** The answer is split into
the same segments the UI renders, each with its footnotes. The judge (structured
output, `claude-opus-5-5` by default) labels each segment (is it a claim, is it
supported by the passages) and each footnote (does its quote support that segment), plus
whether the answer abstained and how it compares to the reference answer. Faithfulness,
citation support, citation precision, correctness, and abstention are then computed in
code. Anything the judge skips counts against the answer.

**CI gates retrieval only.** The retrieval eval needs no key and runs on every push. It
fails on an absolute floor (`hybrid.recall@5 >= 0.8`) and on any quality metric
dropping more than 0.03 below the committed report. The tolerance absorbs small
cross-platform differences in ONNX output.

## Alternatives considered

**Chunk-id relevance labels.** Brittle, as described above.

**LLM-judged retrieval relevance.** It costs money and adds noise to a measurement that
can be exact.

**A single 1 to 10 "quality" score from the judge.** Not actionable: it doesn't say
whether a bad answer came from a missing passage, an unsupported claim, or a citation
pointing at the wrong sentence.

## Consequences

- The golden set lists what the author judged to be the supporting passage, not every
  passage that supports the answer. A question answerable from two places can score as
  a miss when retrieval finds the other one. `pep343-protocol` is one such case in the
  current report; it is left as a miss rather than relabelled after seeing the results.
- Generation metrics depend on the judge model and cost two API calls per question, so
  they are run by hand (`make eval-generation`) and their reports are not committed.
- 38 answerable questions is small: one question moves recall@5 by 0.026. The tolerance
  in CI is just above that.
