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


class BatchIngestRequest(BaseModel):
    """A list of already-extracted documents, indexed directly with no
    parsing step of any kind -- the same "direct content list insertion"
    flexibility RAG-Anything's ingestion API offers (see
    `../../missing_to_complite.md`), and the natural batched counterpart
    to `IngestRequest`'s one-document-per-call shape."""

    documents: list[IngestRequest] = Field(..., min_length=1, max_length=200)


class BatchIngestResponse(BaseModel):
    results: list[IngestResponse]
    n_created: int
    n_updated: int
    corpus_size: int


class ReindexRequest(BaseModel):
    """Rebuild the vector index from the current in-memory corpus under a
    brand-new collection name, optionally with a different embedding
    model -- the "deploy a new embedding model without corrupting the
    index" gap this level's own Success Criteria disclosed as
    unexercised. The currently-serving collection is never touched until
    `activate=True` explicitly swaps to the new one."""

    new_collection_name: str = Field(..., min_length=1)
    embed_model: str | None = Field(default=None, description="Defaults to the currently configured embedding model if omitted.")
    activate: bool = Field(default=False, description="If true, atomically point the live app at the new collection/embedder once reindexing succeeds.")


class ReindexResponse(BaseModel):
    new_collection_name: str
    embed_model: str
    documents_written: int
    activated: bool
