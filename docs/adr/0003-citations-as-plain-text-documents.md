# ADR 0003: Native citations, with each passage sent as a plain-text document

Status: accepted

## Context

Every claim in an answer needs a footnote that opens the source at the exact passage.
There are four ways to get there with the Claude API:

1. Ask the model to write `[1]`-style markers and parse them out of the text.
2. Send each retrieved chunk as a `document` block with a plain-text source and
   `citations: {"enabled": true}`. The response then carries `char_location`
   citations: a document index plus character offsets into the text that was sent.
3. Send chunks as `document` blocks with a custom-content source (a list of text
   blocks). Citations come back as `content_block_location`, which gives block indices.
4. Send chunks as `search_result` blocks, the content type the API docs suggest for
   RAG. Citations come back as `search_result_location`, carrying the `source` string
   Footnote supplied plus block indices.

## Decision

Option 2. Each passage becomes one plain-text `document` block. Its `title` is the
document's title, and its `context` (which is shown to the model but never cited) holds
the file path, the section path, and the page range.

The mapping back to the viewer is integer addition. A chunk is the slice
`text[start:end]` of the canonical document, and the API reports offsets into the
string Footnote sent, so `chunk.start + start_char_index` is the position in the
document the viewer renders. Before showing a footnote, Footnote compares the API's
`cited_text` with the stored text at those offsets. If they differ, it searches the
passage for the quote; if the quote is not there at all, the footnote is marked
unverified and does not link anywhere.

## Alternatives considered

**Regex-parsed markers (option 1).** The model's markers say which passage, never which
sentence, and nothing checks that the marked passage supports the claim. A footnote that
highlights a whole 900-character chunk is not something a reader can verify at a glance.

**`search_result` blocks (option 4).** These fit RAG well, and a citation that carries
Footnote's own `source` id keeps working if the documents are reordered. Two things weighed against them
here. First, they have no `context` field, so section and page metadata would have to
go into the title or into a content block, and a content block becomes citable text.
Second, the minimal citable unit is a whole content block. To cite a single sentence,
Footnote would have to pre-split every chunk into sentence blocks and track their
offsets. Plain-text documents get sentence-level chunking from the API, with character
offsets that need no extra bookkeeping. The API docs list `search_result` support on
fewer platforms than `document` citations, which matters for anyone who later points
the client at Bedrock or Foundry.

**Custom-content documents (option 3).** These give full control over citation
granularity, at the price of owning the sentence splitter. They remain the fallback if
the API's sentence chunking turns out to handle code-heavy passages badly; the eval
suite's citation-precision metric is what would show that.

## Consequences

- Citations cannot be combined with structured outputs (`output_config.format`), so the
  answer is free text. The eval judge, which does need JSON, runs as a separate request
  with no citation-enabled documents.
- All documents in a request must have citations enabled, or none. Footnote enables
  them on all.
- Earlier answers are replayed in later turns as plain text without their citations:
  their document indices referred to an earlier request's documents, and API version
  2026-10-21 rejects `citations` on text in user turns.
- A citation whose quote cannot be found is shown, but marked unverified and unlinked.
  Footnote never invents a location.
