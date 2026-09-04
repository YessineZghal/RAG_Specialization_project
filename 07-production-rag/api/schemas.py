"""Request/response contracts — validated by Pydantic before any route
handler logic runs, so a malformed request never reaches the RAG pipeline
at all.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)


class Source(BaseModel):
    doc_id: str
    title: str
    score: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[Source]
    cache_hit: str | None = None  # None | "response" | "semantic"
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    qdrant_docs: int
    version: str = "0.1.0"


class IngestRequest(BaseModel):
    doc_id: str = Field(..., min_length=1)
    title: str = Field(default="")
    text: str = Field(..., min_length=1, max_length=20000)


class IngestResponse(BaseModel):
    doc_id: str
    status: str  # "created" | "updated"
    corpus_size: int
