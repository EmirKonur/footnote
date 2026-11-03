"""Local dense embeddings. No API key and no network after the first model download."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from fastembed import TextEmbedding

Vector = NDArray[np.float32]
Matrix = NDArray[np.float32]


class Embedder(Protocol):
    """Anything that maps text to unit-length float32 vectors."""

    @property
    def model_name(self) -> str: ...

    def embed_passages(self, texts: Sequence[str]) -> Matrix: ...

    def embed_query(self, text: str) -> Vector: ...


def normalize_rows(matrix: Matrix) -> Matrix:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized: Matrix = (matrix / norms).astype(np.float32)
    return normalized


class FastEmbedder:
    """BGE through fastembed's ONNX runtime.

    BGE v1.5 was trained with an instruction prefix for short retrieval queries and no
    prefix for passages. fastembed applies neither, so the query side adds it here; the
    eval suite reports the effect of switching it off.
    """

    QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

    def __init__(self, model_name: str, cache_dir: Path, *, query_instruction: bool = True) -> None:
        self._model_name = model_name
        self._cache_dir = cache_dir
        self._query_instruction = query_instruction
        self._model: TextEmbedding | None = None
        self._lock = threading.Lock()

    @property
    def model_name(self) -> str:
        return self._model_name

    def _load(self) -> TextEmbedding:
        with self._lock:
            if self._model is None:
                # Imported lazily: loading onnxruntime takes about a second, and the CLI
                # commands that never embed should not pay for it.
                from fastembed import TextEmbedding

                self._cache_dir.mkdir(parents=True, exist_ok=True)
                self._model = TextEmbedding(self._model_name, cache_dir=str(self._cache_dir))
            return self._model

    def embed_passages(self, texts: Sequence[str]) -> Matrix:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        model = self._load()
        vectors = list(model.embed(list(texts), batch_size=32))
        return normalize_rows(np.asarray(vectors, dtype=np.float32))

    def embed_query(self, text: str) -> Vector:
        model = self._load()
        prompt = self.QUERY_INSTRUCTION + text if self._query_instruction else text
        (vector,) = list(model.embed([prompt]))
        return normalize_rows(np.asarray([vector], dtype=np.float32))[0]
