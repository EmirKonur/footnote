from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from footnote.answer.fixtures import (
    FixtureStreamer,
    citation_delta,
    message_delta,
    message_start,
    text_delta,
    text_start,
)
from footnote.answer.generate import GenerationRequest, Streamer, StreamEvent
from footnote.config import Settings
from footnote.db import connect
from footnote.schemas import AskRequest
from footnote.server.app import create_app
from footnote.server.routes_chat import _answer_stream  # pyright: ignore[reportPrivateUsage]
from footnote.server.state import AppState, build_state, default_streamer
from tests.conftest import fixture_bytes
from tests.fakes import HashingEmbedder, ScriptedStreamer


def make_client(settings: Settings, streamer: Streamer | None) -> tuple[TestClient, AppState]:
    state = build_state(settings, streamer=streamer, embedder=HashingEmbedder())
    return TestClient(create_app(state)), state


@pytest.fixture
def retrieval_only(settings: Settings) -> Iterator[tuple[TestClient, AppState]]:
    client, state = make_client(settings, None)
    with client:
        yield client, state


def sse_events(body: str) -> list[tuple[str, dict[str, Any]]]:
    events: list[tuple[str, dict[str, Any]]] = []
    for message in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in message.split("\n"))
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def seed(client: TestClient, state: AppState, name: str = "Reading room") -> str:
    collection = client.post("/api/collections", json={"name": name}).json()
    files = [
        ("files", (n, fixture_bytes(n))) for n in ("field-guide.md", "catalogue.html", "loans.txt")
    ]
    response = client.post(f"/api/collections/{collection['id']}/documents", files=files)
    assert response.status_code == 202
    state.jobs.wait_idle()
    job = client.get(f"/api/jobs/{response.json()['id']}").json()
    assert job["status"] == "done", job
    return collection["id"]


def test_health_reports_the_generation_mode(settings: Settings) -> None:
    for streamer, expected in ((None, "disabled"), (FixtureStreamer(), "fixtures")):
        client, _ = make_client(settings, streamer)
        with client:
            body = client.get("/api/health").json()
        assert body["generation"] == expected
        assert body["embedding_model"] == "test-hashing-256"


def test_collection_lifecycle(retrieval_only: tuple[TestClient, AppState]) -> None:
    client, _ = retrieval_only
    created = client.post(
        "/api/collections", json={"name": "  Archive  ", "description": "old letters"}
    )
    assert created.status_code == 201
    collection = created.json()
    assert collection["name"] == "Archive"
    assert client.post("/api/collections", json={"name": "Archive"}).status_code == 409
    assert client.post("/api/collections", json={"name": ""}).status_code == 422

    renamed = client.patch(f"/api/collections/{collection['id']}", json={"name": "Letters"}).json()
    assert renamed["name"] == "Letters"
    assert renamed["description"] == "old letters"
    assert [c["name"] for c in client.get("/api/collections").json()] == ["Letters"]

    assert client.delete(f"/api/collections/{collection['id']}").status_code == 204
    assert client.get(f"/api/collections/{collection['id']}").status_code == 404


def test_upload_reports_per_file_outcomes(retrieval_only: tuple[TestClient, AppState]) -> None:
    client, state = retrieval_only
    collection_id = seed(client, state)
    files = [
        ("files", ("loans.txt", fixture_bytes("loans.txt"))),  # same bytes: unchanged
        ("files", ("../../etc/notes.txt", b"Escaping paths are flattened.")),
        ("files", ("photo.jpg", b"\xff\xd8")),
        ("files", ("broken.pdf", b"%PDF-1.4 nope")),
    ]
    job = client.post(f"/api/collections/{collection_id}/documents", files=files).json()
    state.jobs.wait_idle()
    job = client.get(f"/api/jobs/{job['id']}").json()

    stages = {f["filename"]: (f["stage"], f["error"]) for f in job["files"]}
    assert stages["loans.txt"] == ("unchanged", None)
    assert stages["etc/notes.txt"] == ("done", None)
    assert stages["photo.jpg"][0] == "failed"
    assert "unsupported" in stages["photo.jpg"][1]
    assert stages["broken.pdf"][0] == "failed"
    assert "PDF" in stages["broken.pdf"][1]
    paths = [d["path"] for d in client.get(f"/api/collections/{collection_id}/documents").json()]
    assert sorted(paths) == ["catalogue.html", "etc/notes.txt", "field-guide.md", "loans.txt"]


def test_upload_limit_is_enforced_per_file(settings: Settings) -> None:
    client, state = make_client(settings.model_copy(update={"max_upload_bytes": 10}), None)
    with client:
        collection = client.post("/api/collections", json={"name": "Tiny"}).json()
        job = client.post(
            f"/api/collections/{collection['id']}/documents",
            files=[("files", ("big.txt", b"x" * 11)), ("files", ("ok.txt", b"tiny"))],
        ).json()
        state.jobs.wait_idle()
        files = client.get(f"/api/jobs/{job['id']}").json()["files"]
    assert [(f["filename"], f["stage"]) for f in files] == [
        ("big.txt", "failed"),
        ("ok.txt", "done"),
    ]


def test_search_and_document_offsets_agree(retrieval_only: tuple[TestClient, AppState]) -> None:
    client, state = retrieval_only
    collection_id = seed(client, state)
    response = client.post(
        f"/api/collections/{collection_id}/search", json={"query": "renew a loan twice", "k": 3}
    ).json()

    top = response["passages"][0]
    assert top["document_path"] == "loans.txt"
    assert set(top["ranks"]) <= {"bm25", "dense"}
    document = client.get(f"/api/documents/{top['document_id']}").json()
    assert document["text"][top["start"] : top["end"]] == top["text"]
    assert document["blocks"][0] == {
        "kind": "heading",
        "start": 0,
        "end": 11,
        "level": 1,
        "page": None,
    }


def test_retrieval_only_answers_are_ranked_passages(
    retrieval_only: tuple[TestClient, AppState],
) -> None:
    client, state = retrieval_only
    collection_id = seed(client, state)
    conversation = client.post(f"/api/collections/{collection_id}/conversations", json={}).json()

    response = client.post(
        f"/api/conversations/{conversation['id']}/messages",
        json={"question": "Can I renew a loan?"},
    )
    assert response.headers["content-type"].startswith("text/event-stream")
    events = sse_events(response.text)
    assert [name for name, _ in events] == ["passages", "done"]
    done = events[-1][1]
    assert done["answer"]["status"] == "retrieval_only"
    assert done["answer"]["passages"][0]["document_path"] == "loans.txt"

    stored = client.get(f"/api/conversations/{conversation['id']}").json()
    assert stored["title"] == "Can I renew a loan?"
    assert [m["role"] for m in stored["messages"]] == ["user", "assistant"]
    assert stored["messages"][1]["answer"]["status"] == "retrieval_only"
    assert stored["messages"][1]["id"] == done["message_id"]


def test_streamed_answer_events_arrive_in_order_and_are_persisted(settings: Settings) -> None:
    def script(request: GenerationRequest) -> list[StreamEvent]:
        passage = request.passages[0]
        return [
            message_start("claude-opus-5-5"),
            text_start(0),
            text_delta(0, "Twice."),
            citation_delta(0, 0, passage, 0, min(20, len(passage.text))),
            message_delta("end_turn"),
        ]

    streamer = ScriptedStreamer(script)
    client, state = make_client(settings, streamer)
    with client:
        collection_id = seed(client, state)
        conversation = client.post(
            f"/api/collections/{collection_id}/conversations", json={"title": "Loans"}
        ).json()
        body = client.post(
            f"/api/conversations/{conversation['id']}/messages",
            json={"question": "How often can loans renew?"},
        ).text
        events = sse_events(body)
        names = [name for name, _ in events]
        assert names[0] == "passages"
        assert names[-1] == "done"
        assert names[1:-1] == ["text", "footnote"]
        footnote = events[2][1]["footnote"]
        passage = streamer.requests[0].passages[0]
        assert (footnote["start"], footnote["end"]) == (
            passage.start,
            passage.start + min(20, len(passage.text)),
        )

        stored = client.get(f"/api/conversations/{conversation['id']}").json()
        assert stored["title"] == "Loans"
        assistant = stored["messages"][1]
        assert assistant["content"] == "Twice."
        assert assistant["answer"]["footnotes"][0]["n"] == 1
        listing = client.get(f"/api/collections/{collection_id}/conversations").json()
        assert listing[0]["message_count"] == 2


def test_missing_resources_are_404(retrieval_only: tuple[TestClient, AppState]) -> None:
    client, _ = retrieval_only
    assert client.get("/api/documents/doc_nope").status_code == 404
    assert client.get("/api/jobs/job_nope").status_code == 404
    assert (
        client.post("/api/conversations/conv_nope/messages", json={"question": "hi"}).status_code
        == 404
    )
    assert client.post("/api/collections/col_nope/search", json={"query": "x"}).status_code == 404


async def test_a_stopped_stream_saves_the_partial_answer(settings: Settings) -> None:
    streamer = FixtureStreamer(word_delay=0)
    client, state = make_client(settings, streamer)
    with client:
        collection_id = seed(client, state)
        conversation = client.post(
            f"/api/collections/{collection_id}/conversations", json={}
        ).json()
    conn = connect(settings.db_path)
    stream = _answer_stream(
        state,
        conn,
        collection_id,
        conversation["id"],
        AskRequest(question="How many books can members borrow at once?"),
        [],
    )
    received = [await anext(stream) for _ in range(4)]
    await stream.aclose()  # what Starlette does when the browser disconnects

    assert received[0].startswith("event: passages")
    check = connect(settings.db_path)
    row = check.execute("SELECT payload_json FROM messages WHERE role = 'assistant'").fetchone()
    check.close()
    answer = json.loads(row["payload_json"])
    assert answer["status"] == "stopped"
    assert answer["generated_by"] == "fixtures"


def test_streamer_selection_follows_settings(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-used")
    live = default_streamer(settings)
    assert live is not None
    assert live.mode == "anthropic"
    assert live.model == "claude-opus-5-5"
    assert default_streamer(settings.model_copy(update={"retrieval_only": True})) is None
    fixtures = default_streamer(
        settings.model_copy(update={"fake_llm": "fixtures", "retrieval_only": True})
    )
    assert fixtures is not None
    assert fixtures.mode == "fixtures"
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("HOME", str(settings.data_dir))  # no `ant` profile on disk either
    assert default_streamer(settings) is None
