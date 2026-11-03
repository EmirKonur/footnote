"use client";

import { FilePlus2, FolderPlus } from "lucide-react";
import { useEffect, useRef, useState, type DragEvent, type RefObject } from "react";

import { useShell } from "@/components/app-shell";
import { ApiError } from "@/lib/api";
import { plural } from "@/lib/format";
import type { FileProgress, IngestJob } from "@/lib/types";

const ACCEPT = ".md,.markdown,.html,.htm,.pdf,.txt,.text";
const SUPPORTED = /\.(md|markdown|html?|pdf|txt|text)$/i;
const POLL_MS = 400;

const STAGE_LABEL: Record<FileProgress["stage"], string> = {
  queued: "queued",
  parsing: "reading",
  chunking: "chunking",
  embedding: "embedding",
  storing: "storing",
  done: "done",
  unchanged: "unchanged",
  failed: "failed",
};

function progressOf(file: FileProgress): number {
  switch (file.stage) {
    case "queued":
      return 0.03;
    case "parsing":
      return 0.1;
    case "chunking":
      return 0.2;
    case "embedding":
      return 0.25 + 0.65 * (file.embed_total ? file.embedded / file.embed_total : 0);
    case "storing":
      return 0.95;
    default:
      return 1;
  }
}

function outcome(file: FileProgress): string {
  if (file.stage === "failed") return file.error ?? "failed";
  if (file.stage === "unchanged") return `unchanged, ${plural(file.chunks, "passage")}`;
  if (file.stage === "done") {
    return file.reused ? `${plural(file.chunks, "passage")}, ${file.reused} reused` : plural(file.chunks, "passage");
  }
  if (file.stage === "embedding" && file.embed_total) return `embedding ${file.embedded}/${file.embed_total}`;
  return STAGE_LABEL[file.stage];
}

export function UploadPanel({ collectionId, onIngested, fileInputRef }: {
  collectionId: string;
  onIngested: () => void;
  fileInputRef: RefObject<HTMLInputElement | null>;
}) {
  const { api } = useShell();
  const folderRef = useRef<HTMLInputElement>(null);
  const [job, setJob] = useState<IngestJob | null>(null);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // webkitdirectory isn't in React's attribute types; it is supported by every current browser.
    folderRef.current?.setAttribute("webkitdirectory", "");
  }, []);

  useEffect(() => {
    if (!job || job.status === "done" || job.status === "failed") return;
    const timer = window.setTimeout(() => {
      api
        .job(job.id)
        .then((next) => {
          setJob(next);
          if (next.status === "done" || next.status === "failed") onIngested();
        })
        .catch(() => setError("Lost track of the upload; refresh to see what was ingested."));
    }, POLL_MS);
    return () => window.clearTimeout(timer);
  }, [api, job, onIngested]);

  async function upload(files: File[]) {
    const supported = files.filter((file) => SUPPORTED.test(file.name));
    if (supported.length === 0) {
      setError("None of those files are Markdown, HTML, PDF, or text.");
      return;
    }
    setError(null);
    try {
      setJob(await api.upload(collectionId, supported));
    } catch (failure) {
      setError(failure instanceof ApiError ? failure.message : "The upload failed.");
    }
  }

  function onDrop(event: DragEvent) {
    event.preventDefault();
    setDragging(false);
    void upload([...event.dataTransfer.files]);
  }

  const running = job !== null && (job.status === "queued" || job.status === "running");
  return (
    <div className="side__section">
      <div
        className="dropzone"
        data-active={dragging}
        onDragOver={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
      >
        <span>Drop Markdown, HTML, PDF, or text files</span>
        <div className="dropzone__actions">
          <button type="button" className="button" onClick={() => fileInputRef.current?.click()} title="Upload files (u)">
            <FilePlus2 className="icon" aria-hidden />
            Files
          </button>
          <button type="button" className="button" onClick={() => folderRef.current?.click()}>
            <FolderPlus className="icon" aria-hidden />
            Folder
          </button>
        </div>
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept={ACCEPT}
          className="visually-hidden"
          aria-label="Choose files to upload"
          onChange={(event) => {
            void upload([...(event.target.files ?? [])]);
            event.target.value = "";
          }}
        />
        <input
          ref={folderRef}
          type="file"
          multiple
          className="visually-hidden"
          aria-label="Choose a folder to upload"
          onChange={(event) => {
            void upload([...(event.target.files ?? [])]);
            event.target.value = "";
          }}
        />
      </div>
      {error && <p className="callout callout--danger">{error}</p>}
      {job && (
        <div aria-live="polite">
          <p className="label">{running ? "Ingesting" : "Last upload"}</p>
          <ul className="progress-list">
            {job.files.map((file) => (
              <li key={file.filename} className="progress-item" data-stage={file.stage}>
                <div className="progress-item__row">
                  <span className="progress-item__name" title={file.filename}>
                    {file.filename}
                  </span>
                  <span>{STAGE_LABEL[file.stage]}</span>
                </div>
                <div
                  className="progress-bar"
                  role="progressbar"
                  aria-label={`${file.filename}: ${outcome(file)}`}
                  aria-valuemin={0}
                  aria-valuemax={100}
                  aria-valuenow={Math.round(progressOf(file) * 100)}
                >
                  <span style={{ width: `${progressOf(file) * 100}%` }} />
                </div>
                <span className="meta">{outcome(file)}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
