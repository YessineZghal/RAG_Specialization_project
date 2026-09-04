"""Shared fixtures for Level 3's offline test suite — same fake-embedder/
fake-LLM pattern as Levels 1-2's `conftest.py`. No network, no Ollama, no
PDF/SQLite download required to run these tests.
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
    def __init__(self, response: str = "") -> None:
        self.response = response
        self.calls: list[dict] = []

    def complete(self, prompt: str, system: str | None = None, temperature: float = 0.2) -> str:
        self.calls.append({"prompt": prompt, "system": system, "temperature": temperature})
        return self.response


@pytest.fixture
def fake_llm():
    return FakeLLM
