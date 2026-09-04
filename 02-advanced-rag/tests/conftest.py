"""Shared fixtures for Level 2's offline test suite — no network, no
Ollama, no sentence-transformers model downloads. `FakeEmbedder` and
`fake_embed_fn` are hashing-trick stand-ins (same technique as Level 1's
`FakeHashEmbedder`): no real semantic understanding, but texts sharing
more words *do* get higher cosine similarity, which is enough to exercise
ranking logic meaningfully.
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
    """Duck-types `common.embed.OllamaEmbedder`'s `.embed_one()` interface."""

    model = "fake-hash-embedder"

    def embed_one(self, text: str) -> list[float]:
        return _hash_embed(text)

    def embed_many(self, texts: list[str], desc: str = "") -> "list[list[float]]":
        return [self.embed_one(t) for t in texts]


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def fake_embed_fn():
    """A plain `Callable[[str], list[float]]`, for functions expecting that
    signature directly (`chunking.semantic`, `context_compression.compressor`).
    """
    return _hash_embed


class FakeLLM:
    """Duck-types `common.llm.OllamaLLM.complete()` with a scripted response.

    Pass `response` for the common case (every call gets the same answer).
    Pass `responses` (a list) when a test needs to script a *sequence* of
    different answers across several calls, such as RAPTOR's recursive
    summarization, which calls the LLM once per cluster and once per tree
    level. `responses` is consumed one item per call and repeats its last
    item once exhausted, so a test does not have to predict the exact call
    count in advance.
    """

    def __init__(self, response: str = "", responses: list[str] | None = None) -> None:
        self.response = response
        self.responses = list(responses) if responses else None
        self.calls: list[dict] = []

    def complete(self, prompt: str, system: str | None = None, temperature: float = 0.3) -> str:
        self.calls.append({"prompt": prompt, "system": system, "temperature": temperature})
        if self.responses:
            return self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        return self.response


@pytest.fixture
def fake_llm():
    return FakeLLM


@pytest.fixture
def tiny_corpus() -> dict[str, str]:
    return {
        "doc-refunds": "Refunds are processed within thirty days of purchase for unused items.",
        "doc-onboarding": "New employees complete a ninety day probation period after hiring.",
        "doc-shipping": "Standard shipping takes five to seven business days within the country.",
    }
