"""Offline tests for modular_common/embed.py's `embed_texts` -- in
particular a regression test for a real bug caught while building
multimodal-rag/vision_embedding.py's tests: a fresh (uncached) call with
a fake embedder returned a plain list instead of a numpy array, which
`VectorRetriever.search()` then failed on the moment it touched `.shape`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modular_common.embed import embed_texts


class ListReturningEmbedder:
    """Mirrors the exact shape of `tests/conftest.py`'s `FakeEmbedder`:
    `embed_many` returns a plain `list[list[float]]`, not an ndarray --
    this is what a real `OllamaEmbedder` never does, but a test double
    reasonably does, and `embed_texts` must not assume otherwise.
    """

    model = "list-returning-fake"

    def embed_one(self, text: str) -> list[float]:
        return [float(len(text)), 0.0]

    def embed_many(self, texts: list[str], desc: str = "") -> list[list[float]]:
        return [self.embed_one(t) for t in texts]


def test_embed_texts_returns_a_real_numpy_array_on_a_fresh_uncached_call():
    embedder = ListReturningEmbedder()
    ids, matrix = embed_texts({"a": "hello", "b": "hi"}, embedder=embedder, cache_name="test-embed-fresh")

    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (2, 2)


def test_embed_texts_returns_a_real_numpy_array_on_a_cached_call_too():
    embedder = ListReturningEmbedder()
    texts = {"a": "hello", "b": "hi"}
    embed_texts(texts, embedder=embedder, cache_name="test-embed-cache-roundtrip")  # populate the cache
    ids, matrix = embed_texts(texts, embedder=embedder, cache_name="test-embed-cache-roundtrip")  # cache hit

    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (2, 2)


def test_embed_texts_with_no_texts_returns_an_empty_array():
    ids, matrix = embed_texts({}, embedder=ListReturningEmbedder(), cache_name="test-embed-empty")
    assert ids == []
    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (0, 0)
