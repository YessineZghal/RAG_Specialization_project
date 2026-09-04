"""Vector storage and Top-K retrieval.

`InMemoryVectorStore` is the default: a brute-force cosine-similarity
search over a numpy matrix. For the corpus sizes used in Level 1 (a few
thousand chunks at most) this is fast, has zero external dependencies, and
— crucially for a *first* RAG system — makes the entire "search" step
readable in about ten lines. See ../theory/vector_search.md.

`QdrantVectorStore` is an optional, persistent alternative with the exact
same interface, used by `examples/rag_with_qdrant.py`. It requires
`uv sync --extra qdrant` and a running Qdrant instance
(`docker compose up -d qdrant`).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Protocol

import numpy as np

from .config import settings
from .schema import Chunk, EmbeddedChunk, RetrievedChunk

logger = logging.getLogger(__name__)


class VectorStore(Protocol):
    def add(self, embedded_chunks: list[EmbeddedChunk]) -> None: ...

    def search(self, query_vector: list[float], top_k: int) -> list[RetrievedChunk]: ...

    def __len__(self) -> int: ...


def _cosine_similarity(matrix: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Cosine similarity of `query` against every row of `matrix`.

    Assumes neither is the zero vector to keep the naive implementation
    readable; embeddings from real models never are in practice.
    """
    matrix_norms = np.linalg.norm(matrix, axis=1)
    query_norm = np.linalg.norm(query)
    return (matrix @ query) / (matrix_norms * query_norm + 1e-12)


class InMemoryVectorStore:
    """A brute-force, numpy-backed vector store. No server, no persistence."""

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._vectors: np.ndarray | None = None  # shape: (n_chunks, dim)

    def add(self, embedded_chunks: list[EmbeddedChunk]) -> None:
        if not embedded_chunks:
            return
        new_vectors = np.array([ec.vector for ec in embedded_chunks], dtype=np.float32)
        self._chunks.extend(ec.chunk for ec in embedded_chunks)
        self._vectors = (
            new_vectors if self._vectors is None else np.vstack([self._vectors, new_vectors])
        )

    def search(self, query_vector: list[float], top_k: int = settings.top_k) -> list[RetrievedChunk]:
        if self._vectors is None or len(self._chunks) == 0:
            return []

        query = np.array(query_vector, dtype=np.float32)
        scores = _cosine_similarity(self._vectors, query)
        top_k = min(top_k, len(self._chunks))
        # argpartition for O(n) top-k selection, then sort just those k.
        top_indices = np.argpartition(-scores, top_k - 1)[:top_k]
        top_indices = top_indices[np.argsort(-scores[top_indices])]

        return [
            RetrievedChunk(chunk=self._chunks[i], score=float(scores[i])) for i in top_indices
        ]

    def __len__(self) -> int:
        return len(self._chunks)

    # -- Persistence (so `ask` doesn't re-embed the corpus every run) -------

    def save(self, directory: Path | str) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        if self._vectors is not None:
            np.save(directory / "vectors.npy", self._vectors)
        with (directory / "chunks.jsonl").open("w", encoding="utf-8") as f:
            for chunk in self._chunks:
                f.write(
                    json.dumps(
                        {
                            "id": chunk.id,
                            "text": chunk.text,
                            "document_id": chunk.document_id,
                            "position": chunk.position,
                            "metadata": chunk.metadata,
                        }
                    )
                    + "\n"
                )
        logger.info("Saved index (%d chunks) to %s", len(self._chunks), directory)

    @classmethod
    def load(cls, directory: Path | str) -> "InMemoryVectorStore":
        directory = Path(directory)
        store = cls()
        vectors_path = directory / "vectors.npy"
        chunks_path = directory / "chunks.jsonl"
        if not vectors_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(f"No saved index found at {directory}")

        store._vectors = np.load(vectors_path)
        with chunks_path.open(encoding="utf-8") as f:
            store._chunks = [
                Chunk(
                    id=row["id"],
                    text=row["text"],
                    document_id=row["document_id"],
                    position=row["position"],
                    metadata=row["metadata"],
                )
                for row in (json.loads(line) for line in f if line.strip())
            ]
        logger.info("Loaded index (%d chunks) from %s", len(store._chunks), directory)
        return store


class QdrantVectorStore:
    """Persistent vector store backed by Qdrant (optional dependency).

    Requires `uv sync --extra qdrant` and a reachable Qdrant instance
    (see ../../docker-compose.yml). Kept behind a lazy import so the core
    pipeline never requires `qdrant-client` unless this class is used.
    """

    def __init__(
        self,
        collection_name: str | None = None,
        url: str | None = None,
        vector_size: int | None = None,
    ) -> None:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
        except ImportError as exc:
            raise RuntimeError(
                "The 'qdrant-client' package is required for QdrantVectorStore. "
                "Run `uv sync --extra qdrant` and `docker compose up -d qdrant`."
            ) from exc

        self._Distance = Distance
        self._VectorParams = VectorParams
        self.collection_name = collection_name or settings.qdrant_collection
        self.client = QdrantClient(url=url or settings.qdrant_url)
        self._vector_size = vector_size
        self._count = 0

    def _ensure_collection(self, vector_size: int) -> None:
        if self.client.collection_exists(self.collection_name):
            return
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=self._VectorParams(size=vector_size, distance=self._Distance.COSINE),
        )

    def add(self, embedded_chunks: list[EmbeddedChunk]) -> None:
        if not embedded_chunks:
            return
        from qdrant_client.models import PointStruct

        self._ensure_collection(vector_size=len(embedded_chunks[0].vector))
        points = [
            PointStruct(
                id=self._count + i,
                vector=ec.vector,
                payload={
                    "id": ec.chunk.id,
                    "text": ec.chunk.text,
                    "document_id": ec.chunk.document_id,
                    "position": ec.chunk.position,
                    "metadata": ec.chunk.metadata,
                },
            )
            for i, ec in enumerate(embedded_chunks)
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)
        self._count += len(points)

    def search(self, query_vector: list[float], top_k: int = settings.top_k) -> list[RetrievedChunk]:
        results = self.client.query_points(
            collection_name=self.collection_name, query=query_vector, limit=top_k
        ).points
        return [
            RetrievedChunk(
                chunk=Chunk(
                    id=r.payload["id"],
                    text=r.payload["text"],
                    document_id=r.payload["document_id"],
                    position=r.payload["position"],
                    metadata=r.payload["metadata"],
                ),
                score=float(r.score),
            )
            for r in results
        ]

    def __len__(self) -> int:
        if not self.client.collection_exists(self.collection_name):
            return 0
        return self.client.count(self.collection_name).count


def get_vector_store(backend: str | None = None, **kwargs) -> VectorStore:
    """Factory: `get_vector_store("memory")` or `get_vector_store("qdrant")`."""
    backend = (backend or settings.vector_store_backend).lower()
    if backend == "memory":
        return InMemoryVectorStore(**kwargs)
    if backend == "qdrant":
        return QdrantVectorStore(**kwargs)
    raise ValueError(f"Unknown vector store backend '{backend}'. Expected 'memory' or 'qdrant'.")
