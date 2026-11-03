"use client";

import { Check, Copy } from "lucide-react";
import Link from "next/link";
import { useId, useRef, useState, type ReactNode } from "react";

import {
  answerToMarkdown,
  isAbstention,
  layoutAnswer,
  passageHref,
  sourceLabel,
  viewerHref,
  type AnswerDraft,
  type Inline,
} from "@/lib/answer";
import type { Answer, Footnote, Passage } from "@/lib/types";

export interface NoteFocus {
  turn: string;
  n: number;
}

interface FootnoteContext {
  collectionId: string;
  conversationId: string | null;
  turn: string;
  footnotes: Footnote[];
  active: NoteFocus | null;
  onFocusNote: (focus: NoteFocus | null) => void;
}

export function noteAnchor(turn: string, n: number): string {
  return `note-${turn}-${n}`;
}

/** A superscript reference. Hover or focus shows the cited passage; activating it opens
 * the source at that passage (or, if the citation could not be verified, the rail entry). */
export function FootnoteRef({ n, context, afterRef = false }: {
  n: number;
  context: FootnoteContext;
  /** True when another reference directly precedes this one, as in ¹,². */
  afterRef?: boolean;
}) {
  const cardId = useId();
  const anchor = useRef<HTMLElement>(null);
  const [open, setOpen] = useState(false);
  const [align, setAlign] = useState<"start" | "end">("start");
  const footnote = context.footnotes.find((f) => f.n === n);
  const active = context.active?.turn === context.turn && context.active.n === n;
  const show = (value: boolean) => {
    if (value && anchor.current) {
      // Open toward whichever side has room, so the card never runs off the column.
      const own = anchor.current.getBoundingClientRect();
      const column = anchor.current.closest(".answer__body")?.getBoundingClientRect();
      setAlign(column && own.left > column.left + column.width * 0.55 ? "end" : "start");
    }
    setOpen(value);
    context.onFocusNote(value ? { turn: context.turn, n } : null);
  };
  const href = footnote ? viewerHref(context.collectionId, footnote, context.conversationId ?? undefined) : null;
  const label = footnote ? `Footnote ${n}: ${sourceLabel(footnote)}` : `Footnote ${n}`;
  const handlers = {
    onMouseEnter: () => show(true),
    onMouseLeave: () => show(false),
    onFocus: () => show(true),
    onBlur: () => show(false),
    "aria-describedby": open ? cardId : undefined,
    "aria-label": label,
  };
  return (
    <sup className="ref" data-active={active} ref={anchor}>
      {afterRef && <span className="ref__sep">,</span>}
      {href ? (
        <Link href={href} {...handlers}>
          {n}
        </Link>
      ) : (
        <a href={`#${noteAnchor(context.turn, n)}`} {...handlers}>
          {n}
        </a>
      )}
      {open && footnote && (
        <span className="ref-card" role="tooltip" id={cardId} data-align={align}>
          <blockquote>{footnote.cited_text}</blockquote>
          <span className="meta">
            {sourceLabel(footnote)}
            {!footnote.verified && " · unverified quote"}
          </span>
        </span>
      )}
    </sup>
  );
}

function renderInlines(inlines: Inline[], context: FootnoteContext): ReactNode[] {
  return inlines.map((inline, index) => {
    if (inline.type === "text") return inline.text;
    if (inline.type === "code") return <code key={index}>{inline.text}</code>;
    const afterRef = inlines[index - 1]?.type === "ref";
    return <FootnoteRef key={index} n={inline.n} context={context} afterRef={afterRef} />;
  });
}

export function AnswerBody({ segments, context, streaming }: {
  segments: Answer["segments"];
  context: FootnoteContext;
  streaming: boolean;
}) {
  const blocks = layoutAnswer(segments);
  return (
    <div className="answer__body" data-streaming={streaming}>
      {blocks.map((block, index) => {
        if (block.kind === "paragraph") return <p key={index}>{renderInlines(block.inlines, context)}</p>;
        if (block.kind === "code") {
          return (
            <div key={index}>
              <pre>
                <code>{block.text}</code>
              </pre>
              {block.refs.map((n, i) => (
                <FootnoteRef key={n} n={n} context={context} afterRef={i > 0} />
              ))}
            </div>
          );
        }
        const List = block.ordered ? "ol" : "ul";
        return (
          <List key={index}>
            {block.items.map((item, i) => (
              <li key={i}>{renderInlines(item, context)}</li>
            ))}
          </List>
        );
      })}
      {streaming && blocks.length === 0 && <p />}
    </div>
  );
}

export function PassageList({ passages, collectionId, conversationId }: {
  passages: Passage[];
  collectionId: string;
  conversationId: string | null;
}) {
  if (passages.length === 0) return <p className="meta">No passages matched.</p>;
  return (
    <ol className="passages" aria-label="Ranked passages">
      {passages.map((passage, index) => {
        const ranks = Object.entries(passage.ranks)
          .map(([name, rank]) => `${name === "bm25" ? "BM25" : "dense"} #${rank}`)
          .join(" · ");
        const path = passage.heading_path.filter((h, i) => !(i === 0 && h === passage.document_title));
        return (
          <li key={passage.chunk_id} className="passage">
            <span className="passage__rank">{index + 1}</span>
            <div>
              <div className="passage__where">
                <Link href={passageHref(collectionId, passage, conversationId ?? undefined)}>
                  {passage.document_title}
                </Link>
                {path.length > 0 && <span className="meta">{path.join(" › ")}</span>}
              </div>
              <p className="meta">{ranks}</p>
              <p className="passage__text">{passage.text}</p>
            </div>
          </li>
        );
      })}
    </ol>
  );
}

function CopyButton({ question, answer }: { question: string; answer: Answer }) {
  const [copied, setCopied] = useState(false);
  async function copy() {
    try {
      await navigator.clipboard.writeText(answerToMarkdown(question, answer));
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1600);
    } catch {
      setCopied(false);
    }
  }
  return (
    <button type="button" className="button button--quiet" onClick={copy} title="Copy the answer with its footnotes">
      {copied ? <Check className="icon" aria-hidden /> : <Copy className="icon" aria-hidden />}
      {copied ? "Copied" : "Copy as Markdown"}
    </button>
  );
}

const STATUS_TAGS: Partial<Record<Answer["status"], { text: string; tone?: "danger" }>> = {
  truncated: { text: "Cut off at the output limit" },
  stopped: { text: "Stopped" },
  refused: { text: "Declined", tone: "danger" },
  error: { text: "Failed", tone: "danger" },
};

export function AnswerFooter({ question, answer }: { question: string; answer: Answer }) {
  const status = STATUS_TAGS[answer.status];
  const abstained = isAbstention(answer);
  return (
    <div className="answer__toolbar">
      {answer.generated_by === "fixtures" && (
        <span className="tag tag--accent" title="FOOTNOTE_FAKE_LLM=fixtures: sentences picked by word overlap, no model call">
          Fixture answer · no model call
        </span>
      )}
      {abstained && <span className="tag">Not in these documents</span>}
      {status && <span className={`tag${status.tone ? " tag--danger" : ""}`}>{status.text}</span>}
      {answer.status !== "refused" && answer.status !== "error" && <CopyButton question={question} answer={answer} />}
      {answer.model && answer.generated_by === "anthropic" && <span className="meta">{answer.model}</span>}
      {answer.footnotes.length > 0 && <span className="meta">{answer.footnotes.length} cited passages</span>}
    </div>
  );
}

export function StreamStatus({ draft }: { draft: AnswerDraft }) {
  if (draft.phase === "retrieving") return <p className="meta" role="status">Searching the collection…</p>;
  if (draft.phase === "reading") {
    return (
      <p className="meta" role="status">
        Reading {draft.passages.length} passages…
      </p>
    );
  }
  return null;
}
