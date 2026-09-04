"""Shared fixtures for Level 9's offline test suite -- same fake-LLM/
fake-embedder pattern as every prior level's conftest.py. Fixtures return
the *class* itself (not an instance), so each test can instantiate it
with whatever scripted behavior that test needs.

No Ollama required -- verifies schema validation, extraction parsing,
graph construction, and operator routing against scripted fakes; "does
this work against a real model" is left to the notebooks and the real
evaluation run, both of which do hit the real, running Ollama.
"""

from __future__ import annotations

import pytest


class FakeLLM:
    """Returns scripted responses in order (one per call); falls back to
    `response` (default `""`) once `responses` is exhausted -- same
    contract as every prior level's conftest.py FakeLLM.
    """

    def __init__(self, response: str = "", responses: list[str] | None = None) -> None:
        self.response = response
        self.responses = list(responses) if responses else None
        self.calls: list[str] = []

    def complete(self, prompt: str, system: str | None = None, temperature: float = 0.2) -> str:
        self.calls.append(prompt)
        if self.responses:
            return self.responses.pop(0)
        return self.response


@pytest.fixture
def fake_llm():
    return FakeLLM


class FakeEmbedder:
    """Deterministic, hand-assigned vectors keyed by exact input string, so
    cosine-similarity tests are reproducible without a real embedding model.
    """

    def __init__(self, vectors: dict[str, list[float]] | None = None) -> None:
        self.vectors = vectors or {}
        self.model = "fake-embed-model"

    def embed_one(self, text: str) -> list[float]:
        return self.vectors.get(text, [0.0, 0.0, 1.0])

    def embed_many(self, texts, desc: str = "Embedding"):
        return [self.embed_one(t) for t in texts]


@pytest.fixture
def fake_embedder():
    return FakeEmbedder
