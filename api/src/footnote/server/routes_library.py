"""Collections, documents, uploads, and search."""

from __future__ import annotations

import asyncio
import time
from pathlib import PurePosixPath
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Response, UploadFile, status

from footnote import library
from footnote.ingest.parsers import SUPPORTED_SUFFIXES
from footnote.schemas import (
    Collection,
    CollectionCreate,
    CollectionUpdate,
    DocumentDetail,
    DocumentSummary,
    IngestJob,
    SearchRequest,
    SearchResponse,
)
from footnote.server.state import ConnDep, StateDep

router = APIRouter(prefix="/api")


@router.get("/collections")
def list_collections(conn: ConnDep) -> list[Collection]:
    return library.list_collections(conn)


@router.post("/collections", status_code=status.HTTP_201_CREATED)
def create_collection(body: CollectionCreate, conn: ConnDep) -> Collection:
    return library.create_collection(conn, body.name, body.description)


@router.get("/collections/{collection_id}")
def get_collection(collection_id: str, conn: ConnDep) -> Collection:
    return library.get_collection(conn, collection_id)


@router.patch("/collections/{collection_id}")
def update_collection(collection_id: str, body: CollectionUpdate, conn: ConnDep) -> Collection:
    return library.update_collection(conn, collection_id, body.name, body.description)


@router.delete("/collections/{collection_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_collection(collection_id: str, conn: ConnDep, state: StateDep) -> Response:
    library.delete_collection(conn, collection_id)
    state.retriever.index.forget(collection_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/collections/{collection_id}/documents")
def list_documents(collection_id: str, conn: ConnDep) -> list[DocumentSummary]:
    return library.list_documents(conn, collection_id)


@router.post("/collections/{collection_id}/documents", status_code=status.HTTP_202_ACCEPTED)
async def upload_documents(
    collection_id: str,
    files: Annotated[list[UploadFile], File(description="Markdown, HTML, PDF, or text files")],
    conn: ConnDep,
    state: StateDep,
) -> IngestJob:
    """Accept files and ingest them in the background; poll the returned job."""
    library.get_collection(conn, collection_id)
    if not files:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "no files in the upload")
    limit = state.settings.max_upload_bytes
    accepted: list[tuple[str, bytes | None, str | None]] = []
    for upload in files:
        name = safe_filename(upload.filename or "untitled.txt")
        if PurePosixPath(name).suffix.lower() not in SUPPORTED_SUFFIXES:
            accepted.append(
                (name, None, f"unsupported file type; use {', '.join(SUPPORTED_SUFFIXES)}")
            )
            continue
        data = await upload.read(limit + 1)
        if len(data) > limit:
            accepted.append(
                (name, None, f"larger than the {limit // (1024 * 1024)} MB upload limit")
            )
            continue
        accepted.append((name, data, None))
    return state.jobs.submit(collection_id, accepted)


def safe_filename(raw: str) -> str:
    """Keep a folder upload's relative path, but never a path that climbs out of it."""
    parts = [
        p for p in PurePosixPath(raw.replace("\\", "/")).parts if p not in ("", ".", "..", "/")
    ]
    return "/".join(parts) or "untitled.txt"


@router.get("/jobs/{job_id}")
def get_job(job_id: str, state: StateDep) -> IngestJob:
    job = state.jobs.get(job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"job {job_id} not found")
    return job


@router.get("/documents/{document_id}")
def get_document(document_id: str, conn: ConnDep) -> DocumentDetail:
    return library.get_document(conn, document_id)


@router.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(document_id: str, conn: ConnDep) -> Response:
    library.delete_document(conn, document_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/collections/{collection_id}/search")
async def search(
    collection_id: str, body: SearchRequest, conn: ConnDep, state: StateDep
) -> SearchResponse:
    library.get_collection(conn, collection_id)
    started = time.perf_counter()
    # Embedding the query and the matrix product are CPU work; keep them off the loop.
    passages = await asyncio.to_thread(
        state.retriever.search, conn, collection_id, body.query, body.k, body.mode
    )
    elapsed = (time.perf_counter() - started) * 1000
    return SearchResponse(
        query=body.query, mode=body.mode, passages=passages, elapsed_ms=round(elapsed, 1)
    )
