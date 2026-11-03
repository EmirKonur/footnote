"use client";

import { ArrowLeft, ChevronRight, Trash2 } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { Fragment, useEffect, useMemo, useRef, useState, type ReactNode } from "react";

import { useShell } from "@/components/app-shell";
import { useHotkeys } from "@/components/shortcuts";
import { ApiError } from "@/lib/api";
import { formatBytes, formatDate, plural } from "@/lib/format";
import type { DocumentDetail, ViewerBlock } from "@/lib/types";
import { pageStarts, parseRange, pieces, type Range } from "@/lib/viewer";

function BlockText({ text, block, range }: { text: string; block: ViewerBlock; range: Range | null }) {
  return pieces(text, block, range).map((piece, index) =>
    piece.marked ? (
      <mark key={index} data-highlight>
        {piece.text}
      </mark>
    ) : (
      <Fragment key={index}>{piece.text}</Fragment>
    ),
  );
}

function renderBlock(block: ViewerBlock, content: ReactNode, key: number): ReactNode {
  switch (block.kind) {
    case "heading": {
      const level = Math.min(Math.max(block.level, 1), 6);
      const Tag = `h${level}` as "h1";
      return <Tag key={key}>{content}</Tag>;
    }
    case "code":
      return (
        <pre key={key}>
          <code>{content}</code>
        </pre>
      );
    case "quote":
      return <blockquote key={key}>{content}</blockquote>;
    case "list_item":
      return (
        <p key={key} className="block-list">
          {content}
        </p>
      );
    case "row":
      return (
        <p key={key} className="block-row">
          {content}
        </p>
      );
    case "term":
      return (
        <p key={key} className="block-term">
          {content}
        </p>
      );
    default:
      return <p key={key}>{content}</p>;
  }
}

export function DocumentViewer({ collectionId, documentId, start, end, footnote, fromConversation }: {
  collectionId: string;
  documentId: string;
  start: string | null;
  end: string | null;
  footnote: string | null;
  fromConversation: string | null;
}) {
  const { api, setCrumbs } = useShell();
  const router = useRouter();
  const [doc, setDoc] = useState<DocumentDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [collectionName, setCollectionName] = useState<string>("");
  const pageRef = useRef<HTMLDivElement>(null);
  const backHref = fromConversation ? `/c/${collectionId}?conversation=${fromConversation}` : `/c/${collectionId}`;

  useEffect(() => {
    api.document(documentId).then(setDoc).catch((failure: unknown) => {
      setError(failure instanceof ApiError && failure.status === 404 ? "missing" : String(failure));
    });
    api.collection(collectionId).then((c) => setCollectionName(c.name)).catch(() => undefined);
  }, [api, collectionId, documentId]);

  useEffect(() => {
    setCrumbs(
      <>
        <ChevronRight className="icon" aria-hidden />
        <Link href={`/c/${collectionId}`}>{collectionName || "Collection"}</Link>
        <ChevronRight className="icon" aria-hidden />
        <span>{doc?.title ?? "…"}</span>
      </>,
    );
  }, [collectionId, collectionName, doc?.title, setCrumbs]);

  useEffect(() => {
    if (doc) document.title = `${doc.title} · Footnote`;
  }, [doc]);

  const range = useMemo(() => (doc ? parseRange(start, end, doc.text.length) : null), [doc, start, end]);
  const pages = useMemo(() => (doc ? pageStarts(doc.blocks) : new Map<number, number>()), [doc]);

  useEffect(() => {
    if (!range) return;
    const mark = pageRef.current?.querySelector("mark[data-highlight]");
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    mark?.scrollIntoView({ block: "center", behavior: reduce ? "auto" : "smooth" });
  }, [range]);

  function stepHighlight(direction: 1 | -1) {
    const marks = [...(pageRef.current?.querySelectorAll<HTMLElement>("mark[data-highlight]") ?? [])];
    if (marks.length === 0) return;
    const middle = window.innerHeight / 2;
    const positions = marks.map((m) => m.getBoundingClientRect().top - middle);
    const index = direction === 1 ? positions.findIndex((p) => p > 4) : positions.findLastIndex((p) => p < -4);
    marks[index === -1 ? (direction === 1 ? 0 : marks.length - 1) : index]?.scrollIntoView({ block: "center" });
  }

  useHotkeys({ b: () => router.push(backHref), "]": () => stepHighlight(1), "[": () => stepHighlight(-1) });

  async function remove() {
    if (!doc || !window.confirm(`Remove “${doc.title}” from this collection?`)) return;
    await api.deleteDocument(doc.id);
    router.push(`/c/${collectionId}`);
  }

  if (error === "missing") {
    return (
      <div className="lost">
        <h1>
          Document not found<sup>†</sup>
        </h1>
        <p>It may have been removed from the collection or re-ingested under another name.</p>
        <p>
          <Link className="button" href={backHref}>
            Back to the collection
          </Link>
        </p>
      </div>
    );
  }
  if (error) return <p className="callout callout--danger lost">{error}</p>;
  if (!doc) return <p className="meta lost">Fetching the document…</p>;

  const quote = range ? doc.text.slice(range.start, range.end) : null;
  return (
    <div className="viewer">
      <article className="viewer__page" ref={pageRef} aria-label={doc.title}>
        {doc.blocks.map((block, index) => {
          const page = pages.get(index);
          const content = <BlockText text={doc.text} block={block} range={range} />;
          return (
            <Fragment key={index}>
              {page !== undefined && doc.format === "pdf" && (
                <div className="page-rule">
                  <span className="label">Page {page}</span>
                </div>
              )}
              {renderBlock(block, content, index)}
            </Fragment>
          );
        })}
      </article>
      <aside className="viewer__aside" aria-label="About this document">
        <Link className="button" href={backHref}>
          <ArrowLeft className="icon" aria-hidden />
          {fromConversation ? "Back to the conversation" : "Back to the collection"}
        </Link>
        <div>
          <p className="label">Source</p>
          <p className="meta">{doc.path}</p>
          <p className="meta">
            {doc.format} · {formatBytes(doc.size_bytes)} · {plural(doc.chunk_count, "passage")} · ingested{" "}
            {formatDate(doc.ingested_at)}
          </p>
        </div>
        {quote ? (
          <div>
            <p className="label">{footnote ? `Footnote ${footnote}` : "Highlighted passage"}</p>
            <blockquote className="viewer__quote">{quote}</blockquote>
            <p className="meta">
              characters {range?.start.toLocaleString("en")}–{range?.end.toLocaleString("en")} · <kbd>[</kbd>{" "}
              <kbd>]</kbd> to jump
            </p>
          </div>
        ) : (
          start !== null && (
            <p className="callout">The requested passage is outside this document, so nothing is highlighted.</p>
          )
        )}
        <button type="button" className="button button--quiet button--danger" onClick={() => void remove()}>
          <Trash2 className="icon" aria-hidden />
          Remove from collection
        </button>
      </aside>
    </div>
  );
}
