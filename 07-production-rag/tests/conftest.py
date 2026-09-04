"""Shared fixtures for Level 7's offline test suite -- same fake-LLM
pattern as every prior level's conftest.py. Fixtures return the *class*
itself (not an instance), matching Levels 3-6's convention, so each test
can instantiate it with whatever scripted behavior that test needs:

    def test_something(fake_llm):
        llm = fake_llm(responses=["yes"])
        ...

No Ollama, Redis, Qdrant, or Postgres required -- verifies orchestration
logic against scripted fakes; "does the real backend behave" is left to
the notebooks, which do hit the real, running stack.
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

    def embed_one(self, text: str) -> list[float]:
        return self.vectors.get(text, [0.0, 0.0, 1.0])


@pytest.fixture
def fake_embedder():
    return FakeEmbedder


class FakeRedisStore:
    """In-memory stand-in for retrieval-infrastructure/redis_store.py's
    RedisStore -- same get/set/delete/scan_keys surface, backed by a dict.
    """

    def __init__(self) -> None:
        self._data: dict[str, str] = {}

    def ping(self) -> bool:
        return True

    def get(self, key: str) -> str | None:
        return self._data.get(key)

    def set(self, key: str, value: str, ttl_seconds: int | None = None) -> None:
        self._data[key] = value

    def delete(self, key: str) -> None:
        self._data.pop(key, None)

    def scan_keys(self, pattern: str) -> list[str]:
        prefix = pattern.rstrip("*")
        return [k for k in self._data if k.startswith(prefix)]


@pytest.fixture
def fake_redis():
    return FakeRedisStore
