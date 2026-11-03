import type { Answer, AnswerEvent, Footnote, Passage, Segment } from "@/lib/types";

/* Streaming state ------------------------------------------------------------------ */

export type Phase = "retrieving" | "reading" | "writing" | "done";

export interface AnswerDraft {
  phase: Phase;
  passages: Passage[];
  segments: Segment[];
  footnotes: Footnote[];
  notices: string[];
  error: { code: string; message: string; retryable: boolean } | null;
  /** Set by the final `done` event; once present it is the source of truth. */
  final: Answer | null;
  messageId: string | null;
}

export const emptyDraft: AnswerDraft = {
  phase: "retrieving",
  passages: [],
  segments: [],
  footnotes: [],
  notices: [],
  error: null,
  final: null,
  messageId: null,
};

/** Fold one server event into the draft. Pure, so the stream can be replayed in tests. */
export function applyEvent(draft: AnswerDraft, event: AnswerEvent): AnswerDraft {
  switch (event.event) {
    case "passages":
      return { ...draft, phase: "reading", passages: event.passages };
    case "text": {
      const segments = [...draft.segments];
      while (segments.length <= event.segment) segments.push({ text: "", footnotes: [] });
      const current = segments[event.segment]!;
      segments[event.segment] = { ...current, text: current.text + event.text };
      return { ...draft, phase: "writing", segments };
    }
    case "footnote": {
      const segments = [...draft.segments];
      while (segments.length <= event.segment) segments.push({ text: "", footnotes: [] });
      const current = segments[event.segment]!;
      if (!current.footnotes.includes(event.footnote.n)) {
        segments[event.segment] = { ...current, footnotes: [...current.footnotes, event.footnote.n] };
      }
      const known = draft.footnotes.some((f) => f.n === event.footnote.n);
      const footnotes = known ? draft.footnotes : [...draft.footnotes, event.footnote];
      return { ...draft, segments, footnotes };
    }
    case "notice":
      return { ...draft, notices: [...draft.notices, event.message] };
    case "error":
      return { ...draft, error: { code: event.code, message: event.message, retryable: event.retryable } };
    case "done":
      return {
        ...draft,
        phase: "done",
        final: event.answer,
        messageId: event.message_id,
        passages: event.answer.passages,
        segments: event.answer.segments,
        footnotes: event.answer.footnotes,
        notices: event.answer.notices,
      };
  }
}

/* Layout: answer text into blocks with footnote references ------------------------ */

export type Inline =
  | { type: "text"; text: string }
  | { type: "code"; text: string }
  | { type: "ref"; n: number };

export type AnswerBlock =
  | { kind: "paragraph"; inlines: Inline[] }
  | { kind: "list"; ordered: boolean; items: Inline[][] }
  | { kind: "code"; text: string; refs: number[] };

// Private-use characters mark footnote positions while the text is split into blocks;
// they cannot occur in model output that survives the API's JSON encoding unchanged.
const REF_OPEN = "";
const REF_CLOSE = "";
const REF_PATTERN = new RegExp(`${REF_OPEN}(\\d+)${REF_CLOSE}`, "g");
const LIST_ITEM = /^\s*(?:([-*+])|(\d+)[.)])\s+/;
const MARKERS_AFTER_SPACE = new RegExp(`(\\s+)((?:${REF_OPEN}\\d+${REF_CLOSE})+)`, "g");

/**
 * Turn segments into renderable blocks. Footnote markers follow the segment they cite,
 * which means a marker can land mid-paragraph, at a list item's end, or (rarely) inside
 * a code fence, where it is moved to just after the block.
 */
export function layoutAnswer(segments: Segment[]): AnswerBlock[] {
  let raw = "";
  for (const segment of segments) {
    raw += segment.text;
    for (const n of segment.footnotes) raw += `${REF_OPEN}${n}${REF_CLOSE}`;
  }
  // A marker hugs the text it cites: pull it back over any whitespace before it, so a
  // segment that ends in "\n\n" doesn't push its footnote onto the next paragraph.
  const source = raw.replace(MARKERS_AFTER_SPACE, "$2$1");
  const blocks: AnswerBlock[] = [];
  const fence = /```[^\n]*\n([\s\S]*?)(?:```|$)/g;
  let cursor = 0;
  for (const match of source.matchAll(fence)) {
    blocks.push(...proseBlocks(source.slice(cursor, match.index)));
    const body = match[1] ?? "";
    const refs = [...body.matchAll(REF_PATTERN)].map((m) => Number(m[1]));
    blocks.push({ kind: "code", text: body.replace(REF_PATTERN, "").replace(/\n$/, ""), refs });
    cursor = match.index + match[0].length;
  }
  blocks.push(...proseBlocks(source.slice(cursor)));
  return blocks;
}

function proseBlocks(text: string): AnswerBlock[] {
  const blocks: AnswerBlock[] = [];
  for (const chunk of text.split(/\n{2,}/)) {
    const lines = chunk.split("\n").filter((line) => line.trim() !== "");
    if (lines.length === 0) continue;
    // A lone marker left between blocks belongs to the block before it.
    if (lines.every((line) => line.replace(REF_PATTERN, "").trim() === "")) {
      const refs = parseInlines(lines.join(""));
      const last = blocks.at(-1);
      if (last?.kind === "paragraph") last.inlines.push(...refs);
      else if (last?.kind === "list") last.items.at(-1)?.push(...refs);
      else blocks.push({ kind: "paragraph", inlines: refs });
      continue;
    }
    let paragraph: string[] = [];
    let list: { ordered: boolean; items: string[] } | null = null;
    const flushParagraph = () => {
      if (paragraph.length) blocks.push({ kind: "paragraph", inlines: parseInlines(paragraph.join(" ")) });
      paragraph = [];
    };
    const flushList = () => {
      if (list) blocks.push({ kind: "list", ordered: list.ordered, items: list.items.map(parseInlines) });
      list = null;
    };
    for (const line of lines) {
      const marker = LIST_ITEM.exec(line);
      if (marker) {
        flushParagraph();
        const ordered = marker[2] !== undefined;
        if (list && list.ordered !== ordered) flushList();
        list ??= { ordered, items: [] };
        list.items.push(line.slice(marker[0].length));
      } else if (list && /^\s+/.test(line)) {
        list.items[list.items.length - 1] += ` ${line.trim()}`;
      } else {
        flushList();
        paragraph.push(line.trim());
      }
    }
    flushParagraph();
    flushList();
  }
  return blocks;
}

export function parseInlines(text: string): Inline[] {
  const inlines: Inline[] = [];
  const token = new RegExp(`\`([^\`]+)\`|${REF_OPEN}(\\d+)${REF_CLOSE}`, "g");
  let cursor = 0;
  for (const match of text.matchAll(token)) {
    if (match.index > cursor) inlines.push({ type: "text", text: text.slice(cursor, match.index) });
    if (match[1] !== undefined) inlines.push({ type: "code", text: match[1] });
    else inlines.push({ type: "ref", n: Number(match[2]) });
    cursor = match.index + match[0].length;
  }
  if (cursor < text.length) inlines.push({ type: "text", text: text.slice(cursor) });
  return inlines;
}

/* Labels and export ----------------------------------------------------------------- */

export function sourceLabel(footnote: Pick<Footnote, "document_title" | "heading_path" | "page">): string {
  // The first heading is usually the document title again; don't print it twice.
  const path = footnote.heading_path.filter((h, i) => !(i === 0 && h === footnote.document_title));
  const parts = [footnote.document_title, ...(path.length ? [path.join(" › ")] : [])];
  if (footnote.page !== null) parts.push(`p. ${footnote.page}`);
  return parts.join(", ");
}

export function isAbstention(answer: Pick<Answer, "status" | "footnotes" | "segments">): boolean {
  if (answer.status !== "complete" || answer.footnotes.length > 0) return false;
  const text = answer.segments.map((s) => s.text).join("").trim();
  return text.startsWith("The documents don't cover this");
}

/** The answer as Markdown with footnotes, ready to paste into notes or a pull request. */
export function answerToMarkdown(question: string, answer: Answer): string {
  const lines = [`> ${question.replace(/\n/g, "\n> ")}`, ""];
  if (answer.status === "retrieval_only") {
    answer.passages.forEach((p, i) => {
      lines.push(`${i + 1}. ${sourceLabel({ ...p, page: p.page_start })}`, "");
      lines.push(`   > ${p.text.replace(/\s+/g, " ").trim()}`, "");
    });
    return lines.join("\n").trimEnd() + "\n";
  }
  let body = "";
  for (const segment of answer.segments) {
    body += segment.text + segment.footnotes.map((n) => `[^${n}]`).join("");
  }
  lines.push(body.trim(), "");
  for (const footnote of answer.footnotes) {
    const quote = footnote.cited_text.replace(/\s+/g, " ").trim();
    const flag = footnote.verified ? "" : " (unverified)";
    lines.push(`[^${footnote.n}]: "${quote}" (${sourceLabel(footnote)}; ${footnote.document_path})${flag}`);
  }
  return lines.join("\n").trimEnd() + "\n";
}

/** Where a verified footnote should open the viewer, or null if it can't be located. */
export function viewerHref(collectionId: string, footnote: Footnote, from?: string): string | null {
  if (!footnote.verified || footnote.start === null || footnote.end === null) return null;
  const params = new URLSearchParams({ start: String(footnote.start), end: String(footnote.end) });
  params.set("n", String(footnote.n));
  if (from) params.set("from", from);
  return `/c/${collectionId}/doc/${footnote.document_id}?${params.toString()}`;
}

export function passageHref(collectionId: string, passage: Passage, from?: string): string {
  const params = new URLSearchParams({ start: String(passage.start), end: String(passage.end) });
  if (from) params.set("from", from);
  return `/c/${collectionId}/doc/${passage.document_id}?${params.toString()}`;
}
