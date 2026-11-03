"""Background ingest jobs with per-file progress the UI can poll.

Jobs run on a single worker thread. That serialises writes to SQLite (one writer at a
time is all it allows anyway) and keeps the embedding model from being loaded twice.
Job state is in memory: a restart loses progress for in-flight uploads, but whatever a
job finished is already committed.
"""

from __future__ import annotations

import sqlite3
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from footnote.db import connect, utc_now
from footnote.ingest.parsers import DocumentParseError, UnsupportedFormatError
from footnote.ingest.pipeline import Stage, ingest_file
from footnote.library import new_id
from footnote.log import get_logger
from footnote.retrieval.embedder import Embedder
from footnote.schemas import FileProgress, FileStage, IngestJob

log = get_logger(__name__)

MAX_JOBS_KEPT = 100


@dataclass(slots=True)
class _FileState:
    filename: str
    data: bytes | None
    stage: FileStage = "queued"
    chunks: int = 0
    embed_total: int = 0
    embedded: int = 0
    reused: int = 0
    document_id: str | None = None
    error: str | None = None


@dataclass(slots=True)
class _JobState:
    id: str
    collection_id: str
    files: list[_FileState]
    created_at: str = field(default_factory=utc_now)
    status: str = "queued"
    finished_at: str | None = None


class JobRegistry:
    def __init__(self, db_path: Path, embedder: Embedder) -> None:
        self._db_path = db_path
        self._embedder = embedder
        self._jobs: OrderedDict[str, _JobState] = OrderedDict()
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ingest")

    def submit(
        self, collection_id: str, files: list[tuple[str, bytes | None, str | None]]
    ) -> IngestJob:
        """Queue files for ingest. A file whose data is ``None`` was rejected before the
        job started (for example, too large) and carries its error message."""
        states = [
            _FileState(name, data, stage="failed" if data is None else "queued", error=error)
            for name, data, error in files
        ]
        job = _JobState(new_id("job"), collection_id, states)
        with self._lock:
            self._jobs[job.id] = job
            while len(self._jobs) > MAX_JOBS_KEPT:
                self._jobs.popitem(last=False)
        self._executor.submit(self._run, job)
        return self._snapshot(job)

    def get(self, job_id: str) -> IngestJob | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return self._snapshot(job) if job else None

    def wait_idle(self) -> None:
        """Block until every queued job has finished (used by tests and the CLI)."""
        self._executor.submit(lambda: None).result()

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)

    def _run(self, job: _JobState) -> None:
        with self._lock:
            job.status = "running"
        conn = connect(self._db_path)
        try:
            for file in job.files:
                if file.data is not None:
                    self._ingest_one(conn, job, file)
        finally:
            conn.close()
        with self._lock:
            job.status = "failed" if all(f.stage == "failed" for f in job.files) else "done"
            job.finished_at = utc_now()

    def _ingest_one(self, conn: sqlite3.Connection, job: _JobState, file: _FileState) -> None:
        data = file.data or b""
        file.data = None

        def progress(stage: Stage, counts: dict[str, int]) -> None:
            with self._lock:
                file.stage = stage
                if stage == "embedding":
                    file.embed_total = counts.get("total", 0)
                    file.embedded = counts.get("done", 0)

        try:
            outcome = ingest_file(
                conn, self._embedder, job.collection_id, file.filename, data, on_progress=progress
            )
        except (UnsupportedFormatError, DocumentParseError) as exc:
            with self._lock:
                file.stage, file.error = "failed", str(exc)
            return
        except Exception:
            log.exception("ingest failed", filename=file.filename, job=job.id)
            with self._lock:
                file.stage, file.error = "failed", "Ingest failed unexpectedly; see the server log."
            return
        with self._lock:
            file.stage = "unchanged" if outcome.status == "unchanged" else "done"
            file.chunks = outcome.chunks
            file.embedded = outcome.embedded
            file.embed_total = outcome.embedded
            file.reused = outcome.reused
            file.document_id = outcome.document_id

    def _snapshot(self, job: _JobState) -> IngestJob:
        return IngestJob.model_validate(
            {
                "id": job.id,
                "collection_id": job.collection_id,
                "status": job.status,
                "created_at": job.created_at,
                "finished_at": job.finished_at,
                "files": [
                    FileProgress(
                        filename=f.filename,
                        stage=f.stage,
                        chunks=f.chunks,
                        embed_total=f.embed_total,
                        embedded=f.embedded,
                        reused=f.reused,
                        document_id=f.document_id,
                        error=f.error,
                    )
                    for f in job.files
                ],
            }
        )
