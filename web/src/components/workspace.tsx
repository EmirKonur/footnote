"use client";

import { ChevronRight, MessageSquarePlus, Trash2 } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import type { NoteFocus } from "@/components/answer-view";
import { useShell } from "@/components/app-shell";
import { Composer } from "@/components/composer";
import { useHotkeys } from "@/components/shortcuts";
import { FootnoteRail, Turn } from "@/components/turn";
import { UploadPanel } from "@/components/upload-panel";
import { applyEvent, emptyDraft, type AnswerDraft } from "@/lib/answer";
import { ApiError } from "@/lib/api";
import { plural } from "@/lib/format";
import type { Answer, ChatMessage, Collection, ConversationSummary, DocumentSummary } from "@/lib/types";

interface Pending {
  question: string;
  draft: AnswerDraft;
}

/** Pair each user message with the assistant message that answered it. */
function turnsOf(messages: ChatMessage[]): { key: string; question: string; answer: Answer | null }[] {
  const turns: { key: string; question: string; answer: Answer | null }[] = [];
  for (const message of messages) {
    if (message.role === "user") turns.push({ key: message.id, question: message.content, answer: null });
    else if (turns.length && turns[turns.length - 1]!.answer === null) turns[turns.length - 1]!.answer = message.answer;
  }
  return turns;
}

function draftToAnswer(
  draft: AnswerDraft,
  status: Answer["status"],
  error: string | null,
  mode: Answer["generated_by"],
): Answer {
  return {
    status,
    segments: draft.segments,
    footnotes: draft.footnotes,
    passages: draft.passages,
    model: null,
    stop_reason: null,
    refusal_category: null,
    notices: draft.notices,
    error,
    generated_by: mode,
  };
}

export function Workspace({ collectionId, conversationId }: { collectionId: string; conversationId: string | null }) {
  const { api, health, setCrumbs } = useShell();
  const router = useRouter();
  const [collection, setCollection] = useState<Collection | null>(null);
  const [missing, setMissing] = useState(false);
  const [documents, setDocuments] = useState<DocumentSummary[] | null>(null);
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [activeId, setActiveId] = useState<string | null>(conversationId);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [pending, setPending] = useState<Pending | null>(null);
  const [focus, setFocus] = useState<NoteFocus | null>(null);
  const [railTurn, setRailTurn] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  const refreshLibrary = useCallback(() => {
    api.collection(collectionId).then(setCollection).catch((error: unknown) => {
      if (error instanceof ApiError && error.status === 404) setMissing(true);
    });
    api.documents(collectionId).then(setDocuments).catch(() => setDocuments([]));
    api.conversations(collectionId).then(setConversations).catch(() => setConversations([]));
  }, [api, collectionId]);

  useEffect(refreshLibrary, [refreshLibrary]);

  useEffect(() => {
    if (collection) document.title = `${collection.name} · Footnote`;
  }, [collection]);

  useEffect(() => {
    setCrumbs(
      collection ? (
        <>
          <ChevronRight className="icon" aria-hidden />
          <span>{collection.name}</span>
        </>
      ) : null,
    );
  }, [collection, setCrumbs]);

  // Loaded on selection rather than in an effect on activeId: ask() sets activeId when it
  // creates a conversation, and re-fetching then would race the stream it just opened.
  const loadConversation = useCallback(
    (id: string) => {
      api
        .conversation(id)
        .then((conversation) => setMessages(conversation.messages))
        .catch(() => setActiveId(null));
    },
    [api],
  );

  useEffect(() => {
    if (conversationId) loadConversation(conversationId);
  }, [conversationId, loadConversation]);

  // Bring the newest question to the top of the thread, so a long answer or passage
  // list reads from its start rather than its end.
  useEffect(() => {
    const turns = scrollRef.current?.querySelectorAll("article.turn");
    turns?.[turns.length - 1]?.scrollIntoView({ block: "start" });
  }, [messages.length, pending?.question]);

  const selectConversation = useCallback(
    (id: string | null) => {
      if (pending) return;
      setActiveId(id);
      setMessages([]);
      setRailTurn(null);
      if (id) loadConversation(id);
      // Keep the URL shareable without a server round trip for every switch.
      const url = id ? `/c/${collectionId}?conversation=${id}` : `/c/${collectionId}`;
      window.history.replaceState(null, "", url);
    },
    [collectionId, loadConversation, pending],
  );

  const mode = health?.generation ?? "disabled";
  const focusNote = useCallback((next: NoteFocus | null) => {
    setFocus(next);
    if (next) setRailTurn(next.turn);
  }, []);

  const ask = useCallback(
    async (question: string) => {
      if (pending) return;
      let id = activeId;
      if (!id) {
        const created = await api.createConversation(collectionId);
        id = created.id;
        setActiveId(id);
        window.history.replaceState(null, "", `/c/${collectionId}?conversation=${id}`);
      }
      const controller = new AbortController();
      abortRef.current = controller;
      let draft: AnswerDraft = emptyDraft;
      setPending({ question, draft });
      let final: Answer;
      let messageId: string | null = null;
      try {
        for await (const event of api.ask(id, question, controller.signal)) {
          draft = applyEvent(draft, event);
          setPending({ question, draft });
        }
        final = draft.final ?? draftToAnswer(draft, "error", "The stream ended without a final answer.", mode);
        messageId = draft.messageId;
      } catch (error) {
        const stopped = error instanceof DOMException && error.name === "AbortError";
        const message = error instanceof Error ? error.message : String(error);
        final = draftToAnswer(draft, stopped ? "stopped" : "error", stopped ? null : message, mode);
      } finally {
        abortRef.current = null;
      }
      const now = new Date().toISOString();
      setMessages((current) => [
        ...current,
        { id: `local-q-${now}`, role: "user", content: question, answer: null, created_at: now },
        { id: messageId ?? `local-a-${now}`, role: "assistant", content: "", answer: final, created_at: now },
      ]);
      setPending(null);
      setRailTurn(null);
      api.conversations(collectionId).then(setConversations).catch(() => undefined);
    },
    [activeId, api, collectionId, pending, mode],
  );

  const stop = useCallback(() => abortRef.current?.abort(), []);

  useHotkeys({
    "/": () => inputRef.current?.focus(),
    n: () => {
      selectConversation(null);
      inputRef.current?.focus();
    },
    u: () => fileRef.current?.click(),
    Escape: stop,
    b: () => router.push("/"),
  });

  const turns = useMemo(() => turnsOf(messages), [messages]);
  const railKey = railTurn ?? (pending ? "pending" : turns.at(-1)?.key ?? null);
  const railFootnotes =
    railKey === "pending"
      ? (pending?.draft.footnotes ?? [])
      : (turns.find((t) => t.key === railKey)?.answer?.footnotes ?? []);

  async function deleteConversation(id: string) {
    if (!window.confirm("Delete this conversation? Its questions and answers will be gone.")) return;
    await api.deleteConversation(id);
    if (id === activeId) selectConversation(null);
    setConversations((current) => current.filter((c) => c.id !== id));
  }

  async function deleteCollection() {
    if (!collection) return;
    if (!window.confirm(`Delete “${collection.name}” with all its documents and conversations?`)) return;
    await api.deleteCollection(collection.id);
    router.push("/");
  }

  if (missing) {
    return (
      <div className="lost">
        <h1>
          No such collection<sup>†</sup>
        </h1>
        <p>It may have been deleted.</p>
        <p>
          <Link className="button" href="/">
            Back to the library
          </Link>
        </p>
      </div>
    );
  }

  const empty = documents !== null && documents.length === 0;
  return (
    <div className="workspace">
      <nav className="side" aria-label="Collection">
        <div className="side__section">
          <p className="label">Collection</p>
          <h1 className="side__title">{collection?.name ?? "…"}</h1>
          {collection && (
            <p className="meta">
              {plural(collection.document_count, "document")} · {plural(collection.chunk_count, "passage")}
            </p>
          )}
        </div>

        <UploadPanel collectionId={collectionId} onIngested={refreshLibrary} fileInputRef={fileRef} />

        <div className="side__section">
          <div className="side__head">
            <p className="label">Conversations</p>
            <button
              type="button"
              className="button button--quiet"
              onClick={() => selectConversation(null)}
              aria-label="New conversation"
              title="New conversation (n)"
            >
              <MessageSquarePlus className="icon" aria-hidden />
            </button>
          </div>
          {conversations.length === 0 ? (
            <p className="meta">None yet. Ask something to start one.</p>
          ) : (
            <ul className="conv-list">
              {conversations.map((conversation) => (
                <li key={conversation.id}>
                  <a
                    href={`/c/${collectionId}?conversation=${conversation.id}`}
                    aria-current={conversation.id === activeId ? "page" : undefined}
                    onClick={(event) => {
                      event.preventDefault();
                      selectConversation(conversation.id);
                    }}
                  >
                    <span className="conv-list__title">{conversation.title || "Untitled"}</span>
                    <span className="meta">{plural(Math.ceil(conversation.message_count / 2), "question")}</span>
                  </a>
                  <button
                    type="button"
                    className="button button--quiet button--danger conv-list__delete"
                    aria-label={`Delete conversation ${conversation.title}`}
                    title="Delete conversation"
                    onClick={() => void deleteConversation(conversation.id)}
                  >
                    <Trash2 className="icon" aria-hidden />
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>

        <div className="side__section">
          <p className="label">Documents</p>
          {documents === null && <p className="meta">Loading…</p>}
          {empty && <p className="meta">Nothing shelved yet. Upload files above.</p>}
          {documents && documents.length > 0 && (
            <ul className="doc-list">
              {documents.map((doc) => (
                <li key={doc.id}>
                  <Link href={`/c/${collectionId}/doc/${doc.id}`}>
                    <span className="doc-list__title">{doc.title}</span>
                    <span className="meta">
                      {doc.format} · {plural(doc.chunk_count, "passage")}
                    </span>
                  </Link>
                </li>
              ))}
            </ul>
          )}
        </div>

        <button type="button" className="button button--quiet button--danger" onClick={() => void deleteCollection()}>
          <Trash2 className="icon" aria-hidden />
          Delete collection
        </button>
      </nav>

      <section className="thread" aria-label="Conversation">
        <div className="thread__scroll" ref={scrollRef}>
          <div className="thread__inner">
            {turns.length === 0 && !pending && (
              <div className="ask-intro">
                <p className="label">{activeId ? "Conversation" : "New conversation"}</p>
                <h1>What would you like to know?</h1>
                <p>
                  {empty
                    ? "Upload a few documents first; questions are answered only from what is on this shelf."
                    : "Questions are answered only from this collection. Every sentence that makes a claim carries a footnote; when the documents don't say, the answer says so."}
                </p>
              </div>
            )}
            {turns.map((turn) => (
              <Turn
                key={turn.key}
                turn={turn.key}
                question={turn.question}
                answer={turn.answer}
                draft={null}
                collectionId={collectionId}
                conversationId={activeId}
                active={focus}
                onFocusNote={focusNote}
                onRetry={(question) => void ask(question)}
              />
            ))}
            {pending && (
              <Turn
                turn="pending"
                question={pending.question}
                answer={null}
                draft={pending.draft}
                collectionId={collectionId}
                conversationId={activeId}
                active={focus}
                onFocusNote={focusNote}
                onRetry={(question) => void ask(question)}
              />
            )}
          </div>
        </div>
        <Composer
          onAsk={(question) => void ask(question)}
          onStop={stop}
          streaming={pending !== null}
          disabled={empty || health === null}
          mode={health?.generation ?? null}
          inputRef={inputRef}
        />
      </section>

      <FootnoteRail
        turn={railKey}
        footnotes={railFootnotes}
        collectionId={collectionId}
        conversationId={activeId}
        active={focus}
        onFocusNote={focusNote}
        retrievalOnly={health?.generation === "disabled"}
      />
    </div>
  );
}
