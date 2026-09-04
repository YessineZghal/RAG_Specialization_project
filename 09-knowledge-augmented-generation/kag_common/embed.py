"""Embedding client for Level 9 -- same on-disk caching pattern as every
prior level's `embed.py`, reimplemented here to keep this level
self-contained.
"""

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
    """Embed every value in `texts` (id -> text), with disk caching.
    Returns `(ids, matrix)` where `matrix[i]` is the embedding of `texts[ids[i]]`.
    """
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

    # np.asarray, not a bare assignment: a real OllamaEmbedder.embed_many
    # already returns an ndarray, but a test double reasonably returns a
    # plain list -- see 03-modular-rag's own real bug (and fix) caught by
    # exactly this gap.
    matrix = np.asarray(embedder.embed_many([texts[i] for i in ids], desc=f"Embedding {cache_name}"))

    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(vectors_path, matrix)
    ids_path.write_text("\n".join(ids))
    logger.info("Cached %d embeddings to %s", len(ids), vectors_path)
    return ids, matrix


def cosine_search(query_vector: np.ndarray, ids: list[str], matrix: np.ndarray, top_k: int = 5) -> list[tuple[str, float]]:
    """Shared brute-force cosine search, same math as every prior level."""
    if matrix.shape[0] == 0:
        return []
    norms = np.linalg.norm(matrix, axis=1) + 1e-12
    query_norm = np.linalg.norm(query_vector) + 1e-12
    scores = (matrix @ query_vector) / (norms * query_norm)
    top_k = min(top_k, len(ids))
    top_indices = np.argpartition(-scores, top_k - 1)[:top_k]
    top_indices = top_indices[np.argsort(-scores[top_indices])]
    return [(ids[i], float(scores[i])) for i in top_indices]
