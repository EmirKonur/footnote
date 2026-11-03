"""Chat history: conversations within a collection and their messages."""

from __future__ import annotations

import sqlite3

from footnote.answer.prompt import HistoryTurn
from footnote.db import transaction, utc_now
from footnote.library import NotFoundError, get_collection, new_id
from footnote.schemas import Answer, ChatMessage, Conversation, ConversationSummary

_SUMMARY_SELECT = """
    SELECT v.id, v.collection_id, v.title, v.created_at, v.updated_at,
           (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = v.id) AS message_count
    FROM conversations v
"""


def _summary(row: sqlite3.Row) -> ConversationSummary:
    return ConversationSummary(
        id=row["id"],
        collection_id=row["collection_id"],
        title=row["title"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        message_count=row["message_count"],
    )


def list_conversations(conn: sqlite3.Connection, collection_id: str) -> list[ConversationSummary]:
    get_collection(conn, collection_id)
    rows = conn.execute(
        _SUMMARY_SELECT + " WHERE v.collection_id = ? ORDER BY v.updated_at DESC", (collection_id,)
    ).fetchall()
    return [_summary(r) for r in rows]


def create_conversation(
    conn: sqlite3.Connection, collection_id: str, title: str = ""
) -> ConversationSummary:
    get_collection(conn, collection_id)
    conversation_id = new_id("conv")
    now = utc_now()
    with transaction(conn):
        conn.execute(
            "INSERT INTO conversations (id, collection_id, title, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (conversation_id, collection_id, title, now, now),
        )
    return get_conversation_summary(conn, conversation_id)


def get_conversation_summary(conn: sqlite3.Connection, conversation_id: str) -> ConversationSummary:
    row = conn.execute(_SUMMARY_SELECT + " WHERE v.id = ?", (conversation_id,)).fetchone()
    if row is None:
        raise NotFoundError(f"conversation {conversation_id} not found")
    return _summary(row)


def get_conversation(conn: sqlite3.Connection, conversation_id: str) -> Conversation:
    summary = get_conversation_summary(conn, conversation_id)
    rows = conn.execute(
        "SELECT id, role, content, payload_json, created_at FROM messages "
        "WHERE conversation_id = ? ORDER BY created_at, rowid",
        (conversation_id,),
    ).fetchall()
    messages = [
        ChatMessage(
            id=r["id"],
            role=r["role"],
            content=r["content"],
            answer=Answer.model_validate_json(r["payload_json"]) if r["payload_json"] else None,
            created_at=r["created_at"],
        )
        for r in rows
    ]
    return Conversation(**summary.model_dump(), messages=messages)


def delete_conversation(conn: sqlite3.Connection, conversation_id: str) -> None:
    with transaction(conn):
        deleted = conn.execute(
            "DELETE FROM conversations WHERE id = ?", (conversation_id,)
        ).rowcount
    if not deleted:
        raise NotFoundError(f"conversation {conversation_id} not found")


def add_user_message(conn: sqlite3.Connection, conversation_id: str, question: str) -> str:
    message_id = new_id("msg")
    now = utc_now()
    with transaction(conn):
        conn.execute(
            "INSERT INTO messages (id, conversation_id, role, content, created_at) "
            "VALUES (?, ?, 'user', ?, ?)",
            (message_id, conversation_id, question, now),
        )
        # The first question names the conversation unless the user already did.
        conn.execute(
            "UPDATE conversations SET updated_at = ?, "
            "title = CASE WHEN title = '' THEN ? ELSE title END WHERE id = ?",
            (now, _title_from(question), conversation_id),
        )
    return message_id


def add_assistant_message(
    conn: sqlite3.Connection, conversation_id: str, message_id: str, answer: Answer
) -> None:
    now = utc_now()
    with transaction(conn):
        conn.execute(
            "INSERT INTO messages (id, conversation_id, role, content, payload_json, created_at) "
            "VALUES (?, ?, 'assistant', ?, ?, ?)",
            (message_id, conversation_id, answer.text, answer.model_dump_json(), now),
        )
        conn.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conversation_id))


def history(conn: sqlite3.Connection, conversation_id: str) -> list[HistoryTurn]:
    """Earlier turns as plain text, for the model's context. Answers that produced no
    text (refusals, errors, retrieval-only results) are left out."""
    rows = conn.execute(
        "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY created_at, rowid",
        (conversation_id,),
    ).fetchall()
    turns: list[HistoryTurn] = []
    for row in rows:
        if row["role"] == "user":
            turns.append(HistoryTurn("user", row["content"]))
        elif row["content"]:
            turns.append(HistoryTurn("assistant", row["content"]))
    return turns


def _title_from(question: str) -> str:
    title = " ".join(question.split())
    return title if len(title) <= 80 else title[:77].rstrip() + "..."
