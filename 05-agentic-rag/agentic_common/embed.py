"""Embedding client with on-disk caching — same pattern as every prior level."""

from __future__ import annotations

import hashlib
import logging

import numpy as np
from tqdm import tqdm

from .config import settings

logger = logging.getLogger(__name__)


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

    def embed_many(self, texts: list[str], desc: str = "Embedding") -> np.ndarray:
        vectors = [self.embed_one(t) for t in tqdm(texts, desc=desc)]
        return np.array(vectors, dtype=np.float32)


def _cache_key(ids: list[str], model: str) -> str:
    digest = hashlib.sha1(f"{model}:{len(ids)}:{ids[0]}:{ids[-1]}".encode()).hexdigest()
    return digest[:12]


def embed_texts(
    texts: dict[str, str],
    embedder: OllamaEmbedder | None = None,
    cache_name: str = "texts",
    force: bool = False,
) -> tuple[list[str], np.ndarray]:
    embedder = embedder or OllamaEmbedder()
    ids = list(texts.keys())
    if not ids:
        return [], np.zeros((0, 0), dtype=np.float32)

    key = _cache_key(ids, embedder.model)
    vectors_path = settings.cache_dir / f"{cache_name}-{key}.npy"
    ids_path = settings.cache_dir / f"{cache_name}-{key}.ids.txt"

    if not force and vectors_path.exists() and ids_path.exists():
        cached_ids = ids_path.read_text().splitlines()
        if cached_ids == ids:
            logger.info("Loaded cached embeddings from %s", vectors_path)
            return ids, np.load(vectors_path)

    matrix = embedder.embed_many([texts[i] for i in ids], desc=f"Embedding {cache_name}")

    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(vectors_path, matrix)
    ids_path.write_text("\n".join(ids))
    logger.info("Cached %d embeddings to %s", len(ids), vectors_path)
    return ids, matrix
