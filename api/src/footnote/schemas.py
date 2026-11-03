"""Pydantic models shared by the HTTP API, the answer pipeline, and the eval suite."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from footnote.ingest.document import BlockKind, DocFormat

SearchMode = Literal["bm25", "dense", "hybrid"]
GenerationMode = Literal["anthropic", "fixtures", "disabled"]


class Model(BaseModel):
    model_config = ConfigDict(frozen=True)


# Library ----------------------------------------------------------------------------


class CollectionCreate(Model):
    name: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=80)]
    description: Annotated[str, StringConstraints(strip_whitespace=True, max_length=400)] = ""


class CollectionUpdate(Model):
    name: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=80)]
    description: Annotated[str, StringConstraints(strip_whitespace=True, max_length=400)] | None = (
        None
    )


class Collection(Model):
    id: str
    name: str
    description: str
    created_at: str
    document_count: int
    chunk_count: int


class DocumentSummary(Model):
    id: str
    collection_id: str
    path: str
    title: str
    format: DocFormat
    size_bytes: int
    chunk_count: int
    ingested_at: str


class ViewerBlock(Model):
    kind: BlockKind
    start: int
    end: int
    level: int
    page: int | None


class DocumentDetail(DocumentSummary):
    text: str
    blocks: list[ViewerBlock]


# Retrieval --------------------------------------------------------------------------


class Passage(Model):
    """One retrieved chunk, with enough provenance to cite and to open it."""

    chunk_id: int
    document_id: str
    document_title: str
    document_path: str
    heading_path: list[str]
    page_start: int | None
    page_end: int | None
    start: int
    end: int
    text: str
    score: float
    ranks: dict[str, int] = Field(default_factory=dict[str, int])


class SearchRequest(Model):
    query: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]
    mode: SearchMode = "hybrid"
    k: Annotated[int, Field(ge=1, le=50)] = 8


class SearchResponse(Model):
    query: str
    mode: SearchMode
    passages: list[Passage]
    elapsed_ms: float


# Answers ----------------------------------------------------------------------------


class Footnote(Model):
    """A citation resolved against Footnote's own records.

    ``start``/``end`` are offsets into the document's canonical text. ``verified`` is
    true when the model's ``cited_text`` matched the stored text at those offsets; an
    unverified footnote still shows the quote but never links to a location.
    """

    n: int
    document_id: str
    document_title: str
    document_path: str
    chunk_id: int
    heading_path: list[str]
    page: int | None
    start: int | None
    end: int | None
    cited_text: str
    verified: bool


class Segment(Model):
    text: str
    footnotes: list[int] = Field(default_factory=list[int])


AnswerStatus = Literal["complete", "truncated", "refused", "stopped", "error", "retrieval_only"]


class Answer(Model):
    status: AnswerStatus
    segments: list[Segment]
    footnotes: list[Footnote]
    passages: list[Passage]
    model: str | None = None
    stop_reason: str | None = None
    refusal_category: str | None = None
    notices: list[str] = Field(default_factory=list[str])
    error: str | None = None
    generated_by: GenerationMode

    @property
    def text(self) -> str:
        return "".join(segment.text for segment in self.segments)


# Chat -------------------------------------------------------------------------------


class ConversationCreate(Model):
    title: Annotated[str, StringConstraints(strip_whitespace=True, max_length=120)] = ""


class ConversationSummary(Model):
    id: str
    collection_id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int


class ChatMessage(Model):
    id: str
    role: Literal["user", "assistant"]
    content: str
    answer: Answer | None
    created_at: str


class Conversation(ConversationSummary):
    messages: list[ChatMessage]


class AskRequest(Model):
    question: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)
    ]
    k: Annotated[int, Field(ge=1, le=20)] | None = None


# Ingest jobs ------------------------------------------------------------------------

FileStage = Literal[
    "queued", "parsing", "chunking", "embedding", "storing", "done", "unchanged", "failed"
]


class FileProgress(Model):
    filename: str
    stage: FileStage
    chunks: int = 0
    embed_total: int = 0
    embedded: int = 0
    reused: int = 0
    document_id: str | None = None
    error: str | None = None


class IngestJob(Model):
    id: str
    collection_id: str
    status: Literal["queued", "running", "done", "failed"]
    files: list[FileProgress]
    created_at: str
    finished_at: str | None = None


class Health(Model):
    status: Literal["ok"]
    version: str
    generation: GenerationMode
    model: str | None
    embedding_model: str
