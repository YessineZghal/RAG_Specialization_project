"""Embedding backends.

Two interchangeable implementations behind one tiny protocol, so
`retrieve.py` and `pipeline.py` never need to know which one is active:

- `OllamaEmbedder` (default) — calls a local Ollama server. No extra
  Python dependency beyond the lightweight `ollama` client; the actual
  model weights are pulled by Ollama itself (`ollama pull nomic-embed-text`),
  not by this repo.
- `SentenceTransformerEmbedder` (optional) — runs a small model in-process
  via `sentence-transformers`. Useful if you don't want to run Ollama for
  embeddings, at the cost of a much heavier dependency (torch). Install
  with `uv sync --extra sentence-transformers`.

Both raise a clear `RuntimeError` (not an import stack trace) if the
backend isn't available, telling you exactly what to install/run.
"""

from __future__ import annotations

import logging
from typing import Protocol

from .config import settings

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    """Anything that turns text into a fixed-size vector."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Order-preserving."""
        ...

    def embed_one(self, text: str) -> list[float]:
        """Embed a single piece of text (e.g. a query)."""
        ...


class OllamaEmbedder:
    """Embeddings via a local Ollama server (default backend)."""

    def __init__(self, model: str | None = None, host: str | None = None) -> None:
        try:
            import ollama
        except ImportError as exc:  # pragma: no cover - defensive
            raise RuntimeError(
                "The 'ollama' package is required for OllamaEmbedder. "
                "Run `uv sync` from the repo root."
            ) from exc

        self._client = ollama.Client(host=host or settings.ollama_host)
        self.model = model or settings.ollama_embed_model

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_one(text) for text in texts]

    def embed_one(self, text: str) -> list[float]:
        try:
            response = self._client.embeddings(model=self.model, prompt=text)
        except Exception as exc:  # noqa: BLE001 - surface a friendlier message
            raise RuntimeError(
                f"Could not reach Ollama at {settings.ollama_host} to embed text with "
                f"model '{self.model}'. Is `ollama serve` running, and have you run "
                f"`ollama pull {self.model}`? Original error: {exc}"
            ) from exc
        return list(response["embedding"])


class SentenceTransformerEmbedder:
    """In-process embeddings via `sentence-transformers` (optional backend)."""

    def __init__(self, model_name: str | None = None) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "The 'sentence-transformers' extra is required for "
                "SentenceTransformerEmbedder. Run `uv sync --extra sentence-transformers`."
            ) from exc

        self.model_name = model_name or settings.sentence_transformer_model
        logger.info("Loading sentence-transformers model '%s'...", self.model_name)
        self._model = SentenceTransformer(self.model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        return [vector.tolist() for vector in vectors]

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


def get_embedder(backend: str | None = None, **kwargs) -> Embedder:
    """Factory: `get_embedder("ollama")` or `get_embedder("sentence-transformers")`."""
    backend = (backend or settings.embedding_backend).lower()
    if backend == "ollama":
        return OllamaEmbedder(**kwargs)
    if backend in {"sentence-transformers", "sentence_transformers", "st"}:
        return SentenceTransformerEmbedder(**kwargs)
    raise ValueError(
        f"Unknown embedding backend '{backend}'. Expected 'ollama' or 'sentence-transformers'."
    )
