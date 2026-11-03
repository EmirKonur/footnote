"use client";

import { RotateCcw } from "lucide-react";
import Link from "next/link";

import {
  AnswerBody,
  AnswerFooter,
  PassageList,
  StreamStatus,
  noteAnchor,
  type NoteFocus,
} from "@/components/answer-view";
import { sourceLabel, viewerHref, type AnswerDraft } from "@/lib/answer";
import type { Answer, Footnote } from "@/lib/types";

interface TurnProps {
  turn: string;
  question: string;
  answer: Answer | null;
  draft: AnswerDraft | null;
  collectionId: string;
  conversationId: string | null;
  active: NoteFocus | null;
  onFocusNote: (focus: NoteFocus | null) => void;
  onRetry: (question: string) => void;
}

export function Turn(props: TurnProps) {
  const { turn, question, answer, draft, collectionId, conversationId } = props;
  const footnotes = answer?.footnotes ?? draft?.footnotes ?? [];
  const context = {
    collectionId,
    conversationId,
    turn,
    footnotes,
    active: props.active,
    onFocusNote: props.onFocusNote,
  };

  return (
    <article className="turn" aria-label={question}>
      <h2 className="question">{question}</h2>
      <div className="answer">
        {draft && !answer && (
          <>
            <StreamStatus draft={draft} />
            {draft.phase === "writing" && <AnswerBody segments={draft.segments} context={context} streaming />}
            {draft.notices.map((notice) => (
              <p key={notice} className="callout callout--accent">
                {notice}
              </p>
            ))}
          </>
        )}
        {answer?.status === "retrieval_only" && (
          <>
            <p className="callout">
              Retrieval only: written answers are off, so these are the passages a model would have been given,
              best first. Set <code>ANTHROPIC_API_KEY</code> to get written, cited answers.
            </p>
            <PassageList passages={answer.passages} collectionId={collectionId} conversationId={conversationId} />
            <AnswerFooter question={question} answer={answer} />
          </>
        )}
        {answer?.status === "refused" && (
          <p className="callout callout--danger">
            The model declined to answer this request
            {answer.refusal_category ? ` (safety category: ${answer.refusal_category})` : ""}. Any partial text
            was discarded. Rephrasing the question sometimes helps.
          </p>
        )}
        {answer && answer.status !== "retrieval_only" && answer.status !== "refused" && (
          <>
            {answer.segments.length > 0 && <AnswerBody segments={answer.segments} context={context} streaming={false} />}
            {answer.notices.map((notice) => (
              <p key={notice} className="callout callout--accent">
                {notice}
              </p>
            ))}
            {answer.status === "error" && (
              <div className="callout callout--danger" role="alert">
                <p>{answer.error ?? "The answer failed."}</p>
                <button type="button" className="button" onClick={() => props.onRetry(question)}>
                  <RotateCcw className="icon" aria-hidden />
                  Ask again
                </button>
              </div>
            )}
            <AnswerFooter question={question} answer={answer} />
          </>
        )}
      </div>
    </article>
  );
}

export function FootnoteRail({ turn, footnotes, collectionId, conversationId, active, onFocusNote, retrievalOnly }: {
  turn: string | null;
  retrievalOnly: boolean;
  footnotes: Footnote[];
  collectionId: string;
  conversationId: string | null;
  active: NoteFocus | null;
  onFocusNote: (focus: NoteFocus | null) => void;
}) {
  return (
    <aside className="rail" aria-label="Footnotes">
      <div className="rail__sticky">
        <p className="label">Footnotes</p>
        {footnotes.length === 0 && retrievalOnly && (
          <p className="meta">
            In retrieval-only mode an answer is the ranked list of passages in the thread. Footnotes appear here
            once the API has Anthropic credentials.
          </p>
        )}
        {footnotes.length === 0 && !retrievalOnly && (
          <p className="meta">
            Sources for the latest answer gather here. Hover a superscript to read its passage; click it to open
            the document at that line.
          </p>
        )}
        {footnotes.length > 0 && (
          <ol className="notes">
            {footnotes.map((footnote) => {
              const href = viewerHref(collectionId, footnote, conversationId ?? undefined);
              const isActive = active?.turn === turn && active.n === footnote.n;
              return (
                <li
                  key={footnote.n}
                  id={turn ? noteAnchor(turn, footnote.n) : undefined}
                  className="note"
                  data-active={isActive}
                  onMouseEnter={() => turn && onFocusNote({ turn, n: footnote.n })}
                  onMouseLeave={() => onFocusNote(null)}
                >
                  <span className="note__n">{footnote.n}</span>
                  <div>
                    <p className="note__quote">{footnote.cited_text}</p>
                    <div className="note__source">
                      {href ? <Link href={href}>{sourceLabel(footnote)}</Link> : <span className="meta">{sourceLabel(footnote)}</span>}
                      {!footnote.verified && (
                        <span className="tag tag--danger" title="The quoted text was not found at the cited location">
                          unverified
                        </span>
                      )}
                    </div>
                  </div>
                </li>
              );
            })}
          </ol>
        )}
      </div>
    </aside>
  );
}
