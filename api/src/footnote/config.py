"""Runtime configuration, read from ``FOOTNOTE_*`` environment variables."""

from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

Effort = Literal["low", "medium", "high", "xhigh", "max"]
FakeLLM = Literal["off", "fixtures"]


class Settings(BaseSettings):
    """All knobs in one place. Defaults are what a fresh clone should run with."""

    # The repo-root .env is found from api/ (where the server runs) and from the root.
    model_config = SettingsConfigDict(
        env_prefix="FOOTNOTE_", env_file=("../.env", ".env"), extra="ignore"
    )

    data_dir: Path = Path("data")
    host: str = "127.0.0.1"
    port: int = 4410
    cors_origins: list[str] = ["http://localhost:4411", "http://127.0.0.1:4411"]

    model: str = "claude-opus-5-5"
    judge_model: str = "claude-opus-5-5"
    # Opus 5.5 defaults to medium when effort is omitted; stating it keeps the choice visible.
    effort: Effort = "medium"
    max_tokens: int = 16000
    fake_llm: FakeLLM = "off"
    # Force retrieval-only answers even when the SDK can find credentials (for example
    # an `ant auth login` profile): demos that must not spend, and the e2e suite.
    retrieval_only: bool = False

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    models_dir: Path | None = None
    """Where the embedding model is cached; defaults to <data_dir>/models."""
    query_instruction: bool = True
    top_k: int = 8
    candidate_pool: int = 50
    rrf_k: int = 60

    max_upload_bytes: int = 25 * 1024 * 1024
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["console", "json"] = "console"

    @cached_property
    def db_path(self) -> Path:
        return self.data_dir / "footnote.sqlite3"

    @cached_property
    def model_cache_dir(self) -> Path:
        return self.models_dir or self.data_dir / "models"
