"use client";

import { ArrowRight, Plus } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useState, type FormEvent } from "react";

import { useShell } from "@/components/app-shell";
import { ApiError } from "@/lib/api";
import { formatDate, plural } from "@/lib/format";
import type { Collection } from "@/lib/types";

export function Library() {
  const { api, offline, setCrumbs } = useShell();
  const router = useRouter();
  const [collections, setCollections] = useState<Collection[] | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [name, setName] = useState("");
  const [creating, setCreating] = useState(false);
  const [createError, setCreateError] = useState<string | null>(null);

  useEffect(() => setCrumbs(null), [setCrumbs]);
  useEffect(() => {
    api
      .collections()
      .then((value) => {
        setCollections(value);
        setLoadError(null);
      })
      .catch((error: unknown) => setLoadError(error instanceof Error ? error.message : String(error)));
  }, [api, offline]);

  async function create(event: FormEvent) {
    event.preventDefault();
    if (!name.trim()) return;
    setCreating(true);
    setCreateError(null);
    try {
      const collection = await api.createCollection(name.trim());
      router.push(`/c/${collection.id}`);
    } catch (error) {
      setCreateError(error instanceof ApiError ? error.message : "Could not create the collection.");
      setCreating(false);
    }
  }

  return (
    <div className="library">
      <section className="library__intro">
        <div>
          <p className="label">The library</p>
          <h1>
            Ask your documents, <em>check every answer.</em>
          </h1>
          <p className="library__lede">
            Each collection is a shelf of Markdown, HTML, PDF, or text files. Answers cite the exact passage
            they rest on; hover a footnote to read it, click to open the page.
          </p>
        </div>
        <form className="new-collection" onSubmit={create}>
          <label className="label" htmlFor="new-collection">
            New collection
          </label>
          <div className="new-collection__row">
            <input
              id="new-collection"
              className="field"
              value={name}
              onChange={(event) => setName(event.target.value)}
              placeholder="Design notes, contracts, papers"
              maxLength={80}
            />
            <button className="button button--primary" disabled={creating || !name.trim()}>
              <Plus className="icon" aria-hidden />
              Create
            </button>
          </div>
          {createError && <p className="callout callout--danger">{createError}</p>}
        </form>
      </section>

      {loadError && !offline && <p className="callout callout--danger">{loadError}</p>}
      {collections === null && !loadError && <p className="meta">Opening the catalogue…</p>}
      {collections?.length === 0 && (
        <div className="empty-state">
          <span className="empty-state__mark" aria-hidden>
            †
          </span>
          <h2>No collections yet</h2>
          <p>
            Create one above and drop in a folder of files, or load the bundled corpus of Python PEPs with{" "}
            <code>make seed</code>.
          </p>
        </div>
      )}
      {collections && collections.length > 0 && (
        <ol className="catalogue" aria-label="Collections">
          {collections.map((collection, index) => (
            <li key={collection.id} className="catalogue__entry">
              <span className="catalogue__number">{String(index + 1).padStart(3, "0")}</span>
              <div>
                <Link href={`/c/${collection.id}`} className="catalogue__name">
                  {collection.name}
                </Link>
                {collection.description && <p className="catalogue__description">{collection.description}</p>}
                <p className="meta">
                  {plural(collection.document_count, "document")} · {plural(collection.chunk_count, "passage")} ·
                  shelved {formatDate(collection.created_at)}
                </p>
              </div>
              <Link href={`/c/${collection.id}`} className="button button--quiet" aria-label={`Open ${collection.name}`}>
                Open <ArrowRight className="icon" aria-hidden />
              </Link>
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}
