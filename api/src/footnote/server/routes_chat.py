"""Conversations and the streaming answer endpoint."""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Response, status
from fastapi.responses import StreamingResponse

from footnote import conversations, library
from footnote.answer.events import DoneEvent, PassagesEvent, to_sse
from footnote.answer.generate import AnswerRun, GenerationRequest
from footnote.answer.prompt import HistoryTurn
from footnote.db import connect
from footnote.log import get_logger
from footnote.schemas import (
    Answer,
    AskRequest,
    Conversation,
    ConversationCreate,
    ConversationSummary,
)
from footnote.server.state import AppState, ConnDep, StateDep

router = APIRouter(prefix="/api")
log = get_logger(__name__)


@router.get("/collections/{collection_id}/conversations")
def list_conversations(collection_id: str, conn: ConnDep) -> list[ConversationSummary]:
    return conversations.list_conversations(conn, collection_id)


@router.post("/collections/{collection_id}/conversations", status_code=status.HTTP_201_CREATED)
def create_conversation(
    collection_id: str, body: ConversationCreate, conn: ConnDep
) -> ConversationSummary:
    return conversations.create_conversation(conn, collection_id, body.title)


@router.get("/conversations/{conversation_id}")
def get_conversation(conversation_id: str, conn: ConnDep) -> Conversation:
    return conversations.get_conversation(conn, conversation_id)


@router.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_conversation(conversation_id: str, conn: ConnDep) -> Response:
    conversations.delete_conversation(conn, conversation_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/conversations/{conversation_id}/messages")
def ask(conversation_id: str, body: AskRequest, state: StateDep) -> StreamingResponse:
    """Answer a question as a server-sent event stream.

    Event order: ``passages`` once, then ``text`` / ``footnote`` / ``notice`` as the
    answer streams, then exactly one ``done`` carrying the final answer (preceded by
    ``error`` if generation failed). Without credentials the stream is ``passages``
    then ``done`` with status ``retrieval_only``.
    """
    conn = connect(state.settings.db_path)
    try:
        summary = conversations.get_conversation_summary(conn, conversation_id)
        turns = conversations.history(conn, conversation_id)
        conversations.add_user_message(conn, conversation_id, body.question)
    except BaseException:
        conn.close()
        raise
    stream = _answer_stream(state, conn, summary.collection_id, conversation_id, body, turns)
    return StreamingResponse(
        stream,
        media_type="text/event-stream",
        # X-Accel-Buffering stops nginx-style proxies from holding the stream back.
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _answer_stream(
    state: AppState,
    conn: sqlite3.Connection,
    collection_id: str,
    conversation_id: str,
    body: AskRequest,
    turns: list[HistoryTurn],
) -> AsyncGenerator[str]:
    message_id = library.new_id("msg")
    run: AnswerRun | None = None
    saved = False

    def save(answer: Answer) -> None:
        nonlocal saved
        if not saved:
            conversations.add_assistant_message(conn, conversation_id, message_id, answer)
            saved = True

    try:
        k = body.k or state.settings.top_k
        passages = await asyncio.to_thread(
            state.retriever.search, conn, collection_id, body.question, k, "hybrid"
        )
        if state.streamer is None:
            answer = Answer(
                status="retrieval_only",
                segments=[],
                footnotes=[],
                passages=passages,
                generated_by="disabled",
            )
            save(answer)
            yield to_sse(PassagesEvent(passages=passages))
            yield to_sse(DoneEvent(answer=answer, message_id=message_id))
            return
        run = AnswerRun(state.streamer, GenerationRequest(body.question, passages, turns))
        async for event in run.events():
            if isinstance(event, DoneEvent):
                save(event.answer)
                event = event.model_copy(update={"message_id": message_id})
            yield to_sse(event)
    except (asyncio.CancelledError, GeneratorExit):
        # The browser pressed stop or went away. Exiting the SDK's stream context (which
        # unwinding does) closes the HTTP connection, so the model stops generating.
        if run is not None:
            save(run.partial())
            log.info("answer stopped by client", conversation=conversation_id)
        raise
    finally:
        conn.close()
