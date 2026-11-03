"""Structured logging setup shared by the API server and the CLI."""

from __future__ import annotations

import logging
import sys
from typing import Literal

import structlog


def configure_logging(level: str = "INFO", fmt: Literal["console", "json"] = "console") -> None:
    numeric = logging.getLevelNamesMapping()[level]
    renderer: structlog.typing.Processor = (
        structlog.processors.JSONRenderer()
        if fmt == "json"
        else structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )
    logging.basicConfig(
        level=numeric, stream=sys.stderr, format="%(levelname)s %(name)s %(message)s"
    )


def get_logger(name: str) -> structlog.typing.FilteringBoundLogger:
    logger: structlog.typing.FilteringBoundLogger = structlog.get_logger(name)
    return logger
