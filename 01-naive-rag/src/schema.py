"""Shared data structures used across every stage of the pipeline.

Keeping these in one place means `ingest.py`, `chunk.py`, `embed.py`,
`retrieve.py`, and `generate.py` all speak the same language, and every
stage can be unit-tested against plain dataclasses without needing a real
embedder, LLM, or dataset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    """One raw, unchunked source document."""

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """A piece of a `Document` produced by `chunk.py`, ready to be embedded."""

    id: str
    text: str
    document_id: str
    position: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddedChunk:
    """A `Chunk` plus its embedding vector, ready to be indexed."""

    chunk: Chunk
    vector: list[float]


@dataclass
class RetrievedChunk:
    """A chunk returned by the vector store, with its similarity score."""

    chunk: Chunk
    score: float


@dataclass
class RagAnswer:
    """The final output of the pipeline for one question."""

    question: str
    answer: str
    sources: list[RetrievedChunk]

    def source_document_ids(self) -> list[str]:
        # Order preserved, duplicates removed (a document can contribute
        # more than one retrieved chunk).
        seen: set[str] = set()
        ids: list[str] = []
        for retrieved in self.sources:
            doc_id = retrieved.chunk.document_id
            if doc_id not in seen:
                seen.add(doc_id)
                ids.append(doc_id)
        return ids
