"""Process-wide services, built once at startup and shared by every request."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Annotated

from anthropic import AsyncAnthropic
from fastapi import Depends, Request

from footnote.answer.anthropic_stream import AnthropicStreamer, has_credentials
from footnote.answer.fixtures import FixtureStreamer
from footnote.answer.generate import Streamer
from footnote.config import Settings
from footnote.db import connect
from footnote.retrieval.embedder import Embedder, FastEmbedder
from footnote.retrieval.search import Retriever
from footnote.retrieval.vectors import VectorIndex
from footnote.schemas import GenerationMode
from footnote.server.jobs import JobRegistry


@dataclass(slots=True)
class AppState:
    settings: Settings
    embedder: Embedder
    retriever: Retriever
    jobs: JobRegistry
    streamer: Streamer | None
    """None means retrieval-only mode: no credentials and no fixture mode."""

    @property
    def generation(self) -> GenerationMode:
        return self.streamer.mode if self.streamer else "disabled"


def build_state(
    settings: Settings, *, streamer: Streamer | None, embedder: Embedder | None = None
) -> AppState:
    """Wire the services. Production passes ``default_streamer(settings)``; tests pass
    their own streamer and a cheap embedder."""
    embedder = embedder or FastEmbedder(
        settings.embedding_model,
        settings.model_cache_dir,
        query_instruction=settings.query_instruction,
    )
    retriever = Retriever(embedder, VectorIndex(), settings.candidate_pool, settings.rrf_k)
    return AppState(
        settings, embedder, retriever, JobRegistry(settings.db_path, embedder), streamer
    )


def default_streamer(settings: Settings) -> Streamer | None:
    if settings.fake_llm == "fixtures":
        return FixtureStreamer()
    if settings.retrieval_only:
        return None
    client = AsyncAnthropic()
    if not has_credentials(client):
        return None
    return AnthropicStreamer(
        client, model=settings.model, effort=settings.effort, max_tokens=settings.max_tokens
    )


def get_state(request: Request) -> AppState:
    state: AppState = request.app.state.footnote
    return state


def get_conn(state: Annotated[AppState, Depends(get_state)]) -> Iterator[sqlite3.Connection]:
    conn = connect(state.settings.db_path)
    try:
        yield conn
    finally:
        conn.close()


StateDep = Annotated[AppState, Depends(get_state)]
ConnDep = Annotated[sqlite3.Connection, Depends(get_conn)]
