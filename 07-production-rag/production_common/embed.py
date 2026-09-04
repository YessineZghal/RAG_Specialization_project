"""Embedding client (Ollama) -- same interface as every prior level.
Unlike Levels 1-6, persistence isn't a local .npy cache here: Qdrant
itself is the durable vector store (that's the point of "retrieval
infrastructure" in a production system).
"""

from __future__ import annotations

from .config import settings


class OllamaEmbedder:
    def __init__(self, model: str | None = None, host: str | None = None) -> None:
        import ollama

        self.model = model or settings.ollama_embed_model
        self._client = ollama.Client(host=host or settings.ollama_host)

    def embed_one(self, text: str) -> list[float]:
        try:
            response = self._client.embeddings(model=self.model, prompt=text)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Could not reach Ollama at {settings.ollama_host} with model "
                f"'{self.model}'. Is `ollama serve` running and the model pulled? {exc}"
            ) from exc
        return list(response["embedding"])

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_one(t) for t in texts]
