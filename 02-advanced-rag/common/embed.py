"""Embedding client for Level 2, with on-disk caching of the corpus matrix.

Embedding the same ~1,000-document scifact subset from scratch every time a
notebook runs would cost several minutes each time. `embed_corpus` caches
the resulting matrix (+ the doc-id order) to `data/cache/`, keyed by model
name and corpus size, so only the *first* run pays the real cost — every
notebook, test, and example after that loads it in milliseconds.
"""

from __future__ import annotations

import hashlib
import logging

import numpy as np
from tqdm import tqdm

from .config import settings

logger = logging.getLogger(__name__)


class OllamaEmbedder:
    """Thin wrapper around the local Ollama embeddings API (same backend as Level 1)."""

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


def _cache_key(doc_ids: list[str], model: str) -> str:
    digest = hashlib.sha1(f"{model}:{len(doc_ids)}:{doc_ids[0]}:{doc_ids[-1]}".encode()).hexdigest()
    return digest[:12]


def embed_corpus(
    corpus_texts: dict[str, str],
    embedder: OllamaEmbedder | None = None,
    cache_name: str = "corpus",
    force: bool = False,
) -> tuple[list[str], np.ndarray]:
    """Embed every text in `corpus_texts` (doc_id -> text), with disk caching.

    Returns `(doc_ids, matrix)` where `matrix[i]` is the embedding of
    `corpus_texts[doc_ids[i]]`.
    """
    embedder = embedder or OllamaEmbedder()
    doc_ids = list(corpus_texts.keys())
    key = _cache_key(doc_ids, embedder.model)
    vectors_path = settings.cache_dir / f"{cache_name}-{key}.npy"
    ids_path = settings.cache_dir / f"{cache_name}-{key}.ids.txt"

    if not force and vectors_path.exists() and ids_path.exists():
        cached_ids = ids_path.read_text().splitlines()
        if cached_ids == doc_ids:
            logger.info("Loaded cached embeddings from %s", vectors_path)
            return doc_ids, np.load(vectors_path)
        logger.warning("Cache id mismatch for %s — recomputing.", cache_name)

    matrix = embedder.embed_many([corpus_texts[d] for d in doc_ids], desc=f"Embedding {cache_name}")

    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(vectors_path, matrix)
    ids_path.write_text("\n".join(doc_ids))
    logger.info("Cached %d embeddings to %s", len(doc_ids), vectors_path)
    return doc_ids, matrix
