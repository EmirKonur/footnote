import { readEventStream } from "@/lib/sse";
import type {
  AnswerEvent,
  Collection,
  Conversation,
  ConversationSummary,
  DocumentDetail,
  DocumentSummary,
  Health,
  IngestJob,
  SearchMode,
  SearchResponse,
} from "@/lib/types";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

/** Thrown when the API cannot be reached at all (as opposed to answering with an error). */
export class OfflineError extends Error {
  constructor(readonly baseUrl: string) {
    super(`Can't reach the Footnote API at ${baseUrl}.`);
    this.name = "OfflineError";
  }
}

export interface FootnoteApi {
  baseUrl: string;
  health(): Promise<Health>;
  collections(): Promise<Collection[]>;
  collection(id: string): Promise<Collection>;
  createCollection(name: string, description?: string): Promise<Collection>;
  deleteCollection(id: string): Promise<void>;
  documents(collectionId: string): Promise<DocumentSummary[]>;
  document(id: string): Promise<DocumentDetail>;
  deleteDocument(id: string): Promise<void>;
  upload(collectionId: string, files: File[]): Promise<IngestJob>;
  job(id: string): Promise<IngestJob>;
  search(collectionId: string, query: string, mode: SearchMode, k?: number): Promise<SearchResponse>;
  conversations(collectionId: string): Promise<ConversationSummary[]>;
  createConversation(collectionId: string, title?: string): Promise<ConversationSummary>;
  conversation(id: string): Promise<Conversation>;
  deleteConversation(id: string): Promise<void>;
  ask(conversationId: string, question: string, signal: AbortSignal): AsyncGenerator<AnswerEvent>;
}

export function createApi(baseUrl: string): FootnoteApi {
  const root = baseUrl.replace(/\/$/, "");

  async function request(path: string, init?: RequestInit): Promise<Response> {
    let response: Response;
    try {
      response = await fetch(`${root}${path}`, init);
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") throw error;
      throw new OfflineError(root);
    }
    if (!response.ok) {
      const detail = await response
        .json()
        .then((body: { detail?: unknown }) => (typeof body.detail === "string" ? body.detail : null))
        .catch(() => null);
      throw new ApiError(detail ?? `${response.status} ${response.statusText}`, response.status);
    }
    return response;
  }

  async function json<T>(path: string, init?: RequestInit): Promise<T> {
    return (await (await request(path, init)).json()) as T;
  }

  const send = (method: string, body: unknown): RequestInit => ({
    method,
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });

  return {
    baseUrl: root,
    health: () => json("/api/health"),
    collections: () => json("/api/collections"),
    collection: (id) => json(`/api/collections/${id}`),
    createCollection: (name, description = "") => json("/api/collections", send("POST", { name, description })),
    deleteCollection: async (id) => {
      await request(`/api/collections/${id}`, { method: "DELETE" });
    },
    documents: (collectionId) => json(`/api/collections/${collectionId}/documents`),
    document: (id) => json(`/api/documents/${id}`),
    deleteDocument: async (id) => {
      await request(`/api/documents/${id}`, { method: "DELETE" });
    },
    upload: (collectionId, files) => {
      const form = new FormData();
      for (const file of files) {
        // Folder uploads carry the relative path; keep it so re-uploads match by path.
        form.append("files", file, file.webkitRelativePath || file.name);
      }
      return json(`/api/collections/${collectionId}/documents`, { method: "POST", body: form });
    },
    job: (id) => json(`/api/jobs/${id}`),
    search: (collectionId, query, mode, k = 8) =>
      json(`/api/collections/${collectionId}/search`, send("POST", { query, mode, k })),
    conversations: (collectionId) => json(`/api/collections/${collectionId}/conversations`),
    createConversation: (collectionId, title = "") =>
      json(`/api/collections/${collectionId}/conversations`, send("POST", { title })),
    conversation: (id) => json(`/api/conversations/${id}`),
    deleteConversation: async (id) => {
      await request(`/api/conversations/${id}`, { method: "DELETE" });
    },
    async *ask(conversationId, question, signal) {
      const response = await request(`/api/conversations/${conversationId}/messages`, {
        ...send("POST", { question }),
        signal,
      });
      if (!response.body) throw new ApiError("The answer stream had no body.", response.status);
      for await (const message of readEventStream(response.body)) {
        yield { event: message.event, ...JSON.parse(message.data) } as AnswerEvent;
      }
    },
  };
}
