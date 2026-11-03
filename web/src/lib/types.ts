/**
 * Mirrors of the API's Pydantic models (api/src/footnote/schemas.py). Kept by hand:
 * the API is small, and a generated client would add a build step for six types.
 */

export type GenerationMode = "anthropic" | "fixtures" | "disabled";
export type SearchMode = "bm25" | "dense" | "hybrid";
export type BlockKind = "heading" | "paragraph" | "list_item" | "code" | "quote" | "row" | "term";

export interface Health {
  status: "ok";
  version: string;
  generation: GenerationMode;
  model: string | null;
  embedding_model: string;
}

export interface Collection {
  id: string;
  name: string;
  description: string;
  created_at: string;
  document_count: number;
  chunk_count: number;
}

export interface DocumentSummary {
  id: string;
  collection_id: string;
  path: string;
  title: string;
  format: "markdown" | "html" | "pdf" | "text";
  size_bytes: number;
  chunk_count: number;
  ingested_at: string;
}

export interface ViewerBlock {
  kind: BlockKind;
  start: number;
  end: number;
  level: number;
  page: number | null;
}

export interface DocumentDetail extends DocumentSummary {
  text: string;
  blocks: ViewerBlock[];
}

export interface Passage {
  chunk_id: number;
  document_id: string;
  document_title: string;
  document_path: string;
  heading_path: string[];
  page_start: number | null;
  page_end: number | null;
  start: number;
  end: number;
  text: string;
  score: number;
  ranks: Partial<Record<"bm25" | "dense", number>>;
}

export interface Footnote {
  n: number;
  document_id: string;
  document_title: string;
  document_path: string;
  chunk_id: number;
  heading_path: string[];
  page: number | null;
  start: number | null;
  end: number | null;
  cited_text: string;
  verified: boolean;
}

export interface Segment {
  text: string;
  footnotes: number[];
}

export type AnswerStatus = "complete" | "truncated" | "refused" | "stopped" | "error" | "retrieval_only";

export interface Answer {
  status: AnswerStatus;
  segments: Segment[];
  footnotes: Footnote[];
  passages: Passage[];
  model: string | null;
  stop_reason: string | null;
  refusal_category: string | null;
  notices: string[];
  error: string | null;
  generated_by: GenerationMode;
}

export interface ConversationSummary {
  id: string;
  collection_id: string;
  title: string;
  created_at: string;
  updated_at: string;
  message_count: number;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  answer: Answer | null;
  created_at: string;
}

export interface Conversation extends ConversationSummary {
  messages: ChatMessage[];
}

export type FileStage =
  | "queued"
  | "parsing"
  | "chunking"
  | "embedding"
  | "storing"
  | "done"
  | "unchanged"
  | "failed";

export interface FileProgress {
  filename: string;
  stage: FileStage;
  chunks: number;
  embed_total: number;
  embedded: number;
  reused: number;
  document_id: string | null;
  error: string | null;
}

export interface IngestJob {
  id: string;
  collection_id: string;
  status: "queued" | "running" | "done" | "failed";
  files: FileProgress[];
  created_at: string;
  finished_at: string | null;
}

export interface SearchResponse {
  query: string;
  mode: SearchMode;
  passages: Passage[];
  elapsed_ms: number;
}

/** Server-sent events from POST /api/conversations/{id}/messages. */
export type AnswerEvent =
  | { event: "passages"; passages: Passage[] }
  | { event: "text"; segment: number; text: string }
  | { event: "footnote"; segment: number; footnote: Footnote }
  | { event: "notice"; message: string }
  | { event: "error"; code: string; message: string; retryable: boolean }
  | { event: "done"; answer: Answer; message_id: string | null };
