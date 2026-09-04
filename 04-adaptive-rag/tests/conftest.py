"""Shared fixtures for Level 4's offline test suite — same fake-embedder/
fake-LLM pattern as every prior level's conftest.py.
"""

from __future__ import annotations

import hashlib

import pytest

VECTOR_DIM = 64


def _hash_embed(text: str) -> list[float]:
    vector = [0.0] * VECTOR_DIM
    for word in text.lower().split():
        index = int(hashlib.md5(word.encode()).hexdigest(), 16) % VECTOR_DIM
        vector[index] += 1.0
    return vector


class FakeEmbedder:
    model = "fake-hash-embedder"

    def embed_one(self, text: str) -> list[float]:
        return _hash_embed(text)

    def embed_many(self, texts, desc: str = ""):
        return [self.embed_one(t) for t in texts]


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()


class FakeLLM:
    def __init__(self, response: str = "", responses: list[str] | None = None) -> None:
        self.response = response
        self.responses = list(responses) if responses else None
        self.calls: list[dict] = []

    def complete(self, prompt: str, system: str | None = None, temperature: float = 0.2) -> str:
        self.calls.append({"prompt": prompt, "system": system, "temperature": temperature})
        if self.responses:
            return self.responses.pop(0)
        return self.response


@pytest.fixture
def fake_llm():
    return FakeLLM


@pytest.fixture
def tiny_corpus() -> dict[str, str]:
    return {
        "Russell Hobbs": "Russell Hobbs is a manufacturer of household appliances based in Failsworth, Greater Manchester.",
        "Peter Hobbs (engineer)": "Peter Wallace Hobbs was an English engineer who with Bill Russell formed Russell Hobbs.",
        "Austrolebias bellottii": "Austrolebias bellottii is a species of fish that lives in Argentina and Uruguay.",
    }


@pytest.fixture
def fake_retriever(fake_embedder, tiny_corpus):
    import numpy as np
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from adaptive_common.retrieval import DenseRetriever

    doc_ids = list(tiny_corpus.keys())
    matrix = np.array([fake_embedder.embed_one(tiny_corpus[d]) for d in doc_ids])
    return DenseRetriever(doc_ids, matrix, fake_embedder)
