# Build context: the repository root (the image also carries the bundled eval corpus).
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.9.27 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app/api
COPY api/pyproject.toml api/uv.lock ./
RUN uv sync --locked --no-dev --no-install-project
COPY api/src ./src
COPY evals ../evals
RUN uv sync --locked --no-dev

RUN useradd --create-home footnote && mkdir -p /data && chown footnote /data
USER footnote
ENV FOOTNOTE_DATA_DIR=/data \
    FOOTNOTE_HOST=0.0.0.0 \
    FOOTNOTE_PORT=4410 \
    FOOTNOTE_LOG_FORMAT=json
VOLUME ["/data"]
EXPOSE 4410
HEALTHCHECK --interval=10s --timeout=3s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:4410/api/health')"
CMD ["footnote", "serve"]
