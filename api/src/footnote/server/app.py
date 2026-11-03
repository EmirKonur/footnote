"""FastAPI application factory."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from importlib.metadata import version

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from footnote.config import Settings
from footnote.db import connect, migrate
from footnote.library import ConflictError, NotFoundError
from footnote.log import configure_logging, get_logger
from footnote.schemas import Health
from footnote.server import routes_chat, routes_library
from footnote.server.state import AppState, StateDep, build_state, default_streamer

log = get_logger(__name__)


def create_app(state: AppState | None = None) -> FastAPI:
    if state is None:
        settings = Settings()
        configure_logging(settings.log_level, settings.log_format)
        state = build_state(settings, streamer=default_streamer(settings))
    app_state = state

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncGenerator[None]:
        conn = connect(app_state.settings.db_path)
        try:
            migrate(conn)
        finally:
            conn.close()
        log.info(
            "footnote api ready",
            db=str(app_state.settings.db_path),
            generation=app_state.generation,
            model=app_state.streamer.model if app_state.streamer else None,
        )
        yield
        app_state.jobs.shutdown()

    app = FastAPI(title="Footnote API", version=version("footnote"), lifespan=lifespan)
    app.state.footnote = app_state
    app.add_middleware(
        CORSMiddleware,
        allow_origins=app_state.settings.cors_origins,
        allow_methods=["GET", "POST", "PATCH", "DELETE"],
        allow_headers=["content-type"],
    )
    app.include_router(routes_library.router)
    app.include_router(routes_chat.router)

    @app.exception_handler(NotFoundError)
    async def not_found(_request: Request, exc: NotFoundError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=status.HTTP_404_NOT_FOUND)

    @app.exception_handler(ConflictError)
    async def conflict(_request: Request, exc: ConflictError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=status.HTTP_409_CONFLICT)

    @app.get("/api/health")
    def health(state: StateDep) -> Health:
        return Health(
            status="ok",
            version=version("footnote"),
            generation=state.generation,
            model=state.streamer.model if state.streamer else None,
            embedding_model=state.embedder.model_name,
        )

    return app
