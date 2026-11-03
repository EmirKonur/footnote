import { describe, expect, it } from "vitest";

import {
  answerToMarkdown,
  applyEvent,
  emptyDraft,
  isAbstention,
  layoutAnswer,
  sourceLabel,
  viewerHref,
  type AnswerDraft,
} from "@/lib/answer";
import type { Answer, AnswerEvent, Footnote } from "@/lib/types";

const footnote = (n: number, overrides: Partial<Footnote> = {}): Footnote => ({
  n,
  document_id: "doc_8",
  document_title: "PEP 8",
  document_path: "pep-0008.html",
  chunk_id: 3,
  heading_path: ["PEP 8", "Code Lay-out", "Maximum Line Length"],
  page: null,
  start: 100,
  end: 146,
  cited_text: "Limit all lines to a maximum of 79 characters.",
  verified: true,
  ...overrides,
});

const answer = (overrides: Partial<Answer> = {}): Answer => ({
  status: "complete",
  segments: [
    { text: "Lines stop at ", footnotes: [] },
    { text: "79 characters", footnotes: [1] },
    { text: ", and docstrings at 72.", footnotes: [2] },
  ],
  footnotes: [footnote(1), footnote(2, { cited_text: "...limited to 72 characters.", start: 200, end: 228 })],
  passages: [],
  model: "claude-opus-5-5",
  stop_reason: "end_turn",
  refusal_category: null,
  notices: [],
  error: null,
  generated_by: "anthropic",
  ...overrides,
});

describe("layoutAnswer", () => {
  it("places each footnote marker right after the segment it cites", () => {
    expect(layoutAnswer(answer().segments)).toEqual([
      {
        kind: "paragraph",
        inlines: [
          { type: "text", text: "Lines stop at 79 characters" },
          { type: "ref", n: 1 },
          { type: "text", text: ", and docstrings at 72." },
          { type: "ref", n: 2 },
        ],
      },
    ]);
  });

  it("keeps markers attached across paragraph and list structure", () => {
    const blocks = layoutAnswer([
      { text: "Two rules apply:\n\n- Use spaces", footnotes: [1] },
      { text: "\n- Never mix tabs", footnotes: [2] },
      { text: "\n\nThat is all.", footnotes: [] },
    ]);
    expect(blocks.map((b) => b.kind)).toEqual(["paragraph", "list", "paragraph"]);
    expect(blocks[1]).toEqual({
      kind: "list",
      ordered: false,
      items: [
        [{ type: "text", text: "Use spaces" }, { type: "ref", n: 1 }],
        [{ type: "text", text: "Never mix tabs" }, { type: "ref", n: 2 }],
      ],
    });
  });

  it("attaches a marker that trails a paragraph break to the previous block", () => {
    const blocks = layoutAnswer([{ text: "First paragraph.\n\n", footnotes: [3] }, { text: "Second.", footnotes: [] }]);
    expect(blocks).toEqual([
      { kind: "paragraph", inlines: [{ type: "text", text: "First paragraph." }, { type: "ref", n: 3 }] },
      { kind: "paragraph", inlines: [{ type: "text", text: "Second." }] },
    ]);
  });

  it("parses inline code and ordered lists", () => {
    const blocks = layoutAnswer([{ text: "Steps:\n1. Run `python -m venv env`\n2. Activate it", footnotes: [] }]);
    expect(blocks[1]).toEqual({
      kind: "list",
      ordered: true,
      items: [
        [{ type: "text", text: "Run " }, { type: "code", text: "python -m venv env" }],
        [{ type: "text", text: "Activate it" }],
      ],
    });
  });

  it("moves a marker out of a code fence instead of printing it as code", () => {
    const blocks = layoutAnswer([
      { text: "Example:\n\n```python\nwith open(p) as f:\n    data = f.read()", footnotes: [4] },
      { text: "\n```\n\nDone.", footnotes: [] },
    ]);
    expect(blocks[1]).toEqual({ kind: "code", text: "with open(p) as f:\n    data = f.read()", refs: [4] });
    expect(blocks[2]).toEqual({ kind: "paragraph", inlines: [{ type: "text", text: "Done." }] });
  });
});

describe("applyEvent", () => {
  const replay = (events: AnswerEvent[]): AnswerDraft => events.reduce(applyEvent, emptyDraft);

  it("builds segments and footnotes as deltas arrive", () => {
    const draft = replay([
      { event: "passages", passages: [] },
      { event: "text", segment: 0, text: "Lines stop at " },
      { event: "text", segment: 1, text: "79" },
      { event: "text", segment: 1, text: " characters" },
      { event: "footnote", segment: 1, footnote: footnote(1) },
      { event: "footnote", segment: 1, footnote: footnote(1) },
    ]);
    expect(draft.phase).toBe("writing");
    expect(draft.segments).toEqual([
      { text: "Lines stop at ", footnotes: [] },
      { text: "79 characters", footnotes: [1] },
    ]);
    expect(draft.footnotes).toHaveLength(1);
  });

  it("lets the final answer replace whatever was streamed", () => {
    const final = answer();
    const draft = replay([
      { event: "text", segment: 0, text: "garbled partial" },
      { event: "done", answer: final, message_id: "msg_1" },
    ]);
    expect(draft.phase).toBe("done");
    expect(draft.segments).toEqual(final.segments);
    expect(draft.messageId).toBe("msg_1");
  });

  it("records errors without dropping text", () => {
    const draft = replay([
      { event: "text", segment: 0, text: "Partial" },
      { event: "error", code: "rate_limited", message: "Slow down", retryable: true },
    ]);
    expect(draft.error).toEqual({ code: "rate_limited", message: "Slow down", retryable: true });
    expect(draft.segments[0]?.text).toBe("Partial");
  });
});

describe("answerToMarkdown", () => {
  it("writes Markdown footnotes with the cited passage and its source", () => {
    expect(answerToMarkdown("How long can lines be?", answer())).toBe(
      [
        "> How long can lines be?",
        "",
        "Lines stop at 79 characters[^1], and docstrings at 72.[^2]",
        "",
        '[^1]: "Limit all lines to a maximum of 79 characters." (PEP 8, Code Lay-out › Maximum Line Length; pep-0008.html)',
        '[^2]: "...limited to 72 characters." (PEP 8, Code Lay-out › Maximum Line Length; pep-0008.html)',
        "",
      ].join("\n"),
    );
  });

  it("flags unverified footnotes", () => {
    const text = answerToMarkdown("q", answer({ footnotes: [footnote(1, { verified: false })], segments: [{ text: "x", footnotes: [1] }] }));
    expect(text).toContain("pep-0008.html) (unverified)");
  });
});

describe("helpers", () => {
  it("does not repeat the document title in the source label", () => {
    expect(sourceLabel(footnote(1, { page: 4 }))).toBe("PEP 8, Code Lay-out › Maximum Line Length, p. 4");
    expect(sourceLabel(footnote(1, { heading_path: [] }))).toBe("PEP 8");
  });

  it("links only verified footnotes to the viewer", () => {
    expect(viewerHref("col_1", footnote(2), "conv_9")).toBe("/c/col_1/doc/doc_8?start=100&end=146&n=2&from=conv_9");
    expect(viewerHref("col_1", footnote(2, { verified: false }))).toBeNull();
  });

  it("recognises an abstention only when nothing was cited", () => {
    const declined = answer({ segments: [{ text: "The documents don't cover this.", footnotes: [] }], footnotes: [] });
    expect(isAbstention(declined)).toBe(true);
    expect(isAbstention(answer())).toBe(false);
  });
});
