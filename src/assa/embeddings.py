"""Embeddings.

The default provider is ``sentence-transformers`` with ``all-MiniLM-L6-v2`` (or
``BAAI/bge-small-en-v1.5``), which is a choice the participant guide allows. The
model is loaded once per process and reused.

``ASSA_EMBEDDING_PROVIDER=hash`` selects a deterministic bag-of-words hashing
embedder. It exists so unit tests and the ``MOCK_LLM`` smoke path can run without
downloading a model; it is **not** a retrieval-quality substitute and is never
the default in any deployed configuration.
"""

from __future__ import annotations

import hashlib
import os
import re
from functools import lru_cache
from typing import Protocol, Sequence

import numpy as np

from .config import get_settings

TOKEN_RE = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    dimension: int
    name: str

    def encode(self, texts: Sequence[str]) -> list[list[float]]: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self.name = model_name
        self.dimension = int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._model.encode(
            list(texts), normalize_embeddings=True, show_progress_bar=False, batch_size=16
        )
        return [vector.tolist() for vector in vectors]


class HashingEmbedder:
    """Deterministic test-only embedder (see the module docstring)."""

    dimension = 384
    name = "hash-test-only"

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vector = np.zeros(self.dimension, dtype=np.float32)
            for token in TOKEN_RE.findall(text.casefold()):
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                bucket = int.from_bytes(digest[:4], "big") % self.dimension
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vector[bucket] += sign
            norm = float(np.linalg.norm(vector))
            if norm:
                vector /= norm
            vectors.append(vector.tolist())
        return vectors


@lru_cache(maxsize=2)
def get_embedder() -> Embedder:
    provider = os.environ.get("ASSA_EMBEDDING_PROVIDER", "sentence-transformers").strip().lower()
    settings = get_settings()
    if provider in {"hash", "hashing", "test"}:
        return HashingEmbedder()
    if provider in {"sentence-transformers", "sentence_transformers", "minilm", "st"}:
        return SentenceTransformerEmbedder(settings.embedding_model)
    raise ValueError(
        f"unknown ASSA_EMBEDDING_PROVIDER {provider!r} (use sentence-transformers or hash)"
    )


def embedder_health() -> tuple[bool, str]:
    try:
        embedder = get_embedder()
    except Exception as exc:  # pragma: no cover - depends on the local environment
        return False, str(exc)
    return True, f"{embedder.name} ({embedder.dimension} dims)"
