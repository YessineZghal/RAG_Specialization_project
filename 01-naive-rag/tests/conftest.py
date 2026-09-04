"""Shared fixtures for the offline test suite.

Nothing here touches the network, Ollama, or the real Hugging Face
dataset — `FakeHashEmbedder` stands in for a real embedding model so the
full pipeline (chunk -> embed -> store -> retrieve -> generate) can be
tested deterministically and fast.
"""

from __future__ import annotations

import hashlib

import pytest

from src.schema import Document

VECTOR_DIM = 64


class FakeHashEmbedder:
    """A deterministic, dependency-free stand-in for a real embedder.

    Uses the classic "hashing trick": each word votes for one dimension of
    a fixed-size vector. It has no real semantic understanding, but texts
    that share more words *do* end up with higher cosine similarity, which
    is enough to exercise retrieval logic meaningfully in tests.
    """

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_one(t) for t in texts]

    def embed_one(self, text: str) -> list[float]:
        vector = [0.0] * VECTOR_DIM
        for word in text.lower().split():
            index = int(hashlib.md5(word.encode()).hexdigest(), 16) % VECTOR_DIM
            vector[index] += 1.0
        return vector


@pytest.fixture
def fake_embedder() -> FakeHashEmbedder:
    return FakeHashEmbedder()


@pytest.fixture
def sample_documents() -> list[Document]:
    return [
        Document(
            id="doc-refunds",
            text=(
                "Customers may request a full refund within thirty days of purchase. "
                "Refund requests must include the original order number. "
                "Enterprise refunds are prorated and approved by an account manager."
            ),
        ),
        Document(
            id="doc-onboarding",
            text=(
                "New employees have a ninety day probation period. "
                "Full time employees accrue fifteen days of paid time off per year. "
                "Remote work is allowed up to three days per week after probation."
            ),
        ),
    ]
