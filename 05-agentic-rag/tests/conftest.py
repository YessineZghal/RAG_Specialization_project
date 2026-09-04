"""Shared fixtures for Level 5's offline test suite — same fake-embedder/
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


class FakeVectorTool:
    """Duck-types tools/vector_tool.py's VectorTool.__call__ interface."""

    def __init__(self, results: list[dict] | None = None) -> None:
        self.results = results if results is not None else [
            {"chunk_id": "doc-1", "text": "Sophomores are second-year students.", "article_title": "Student", "score": 0.9}
        ]
        self.calls: list[str] = []

    def __call__(self, query: str, top_k: int = 5) -> list[dict]:
        self.calls.append(query)
        return self.results


@pytest.fixture
def fake_vector_tool():
    return FakeVectorTool
