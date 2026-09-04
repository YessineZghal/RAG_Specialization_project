"""Qdrant client — the durable vector store a production API points at,
instead of the in-memory numpy arrays every earlier level used (fine for
learning; gone the moment the process restarts).
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from production_common.config import settings


def _point_id(doc_id: str) -> str:
    """A stable, deterministic Qdrant point id derived from `doc_id`.

    Real bug this fixes: an earlier version assigned point ids from each
    call's own `enumerate()` position (0, 1, 2, ...). That was invisible
    the whole time `upsert()` was only ever called once, with the entire
    corpus at startup (see `api/main.py`) -- id 0 always meant "the first
    document in that one batch." The moment `/admin/ingest` (see
    `api/routes.py`) calls `upsert()` again with a single new document,
    that document would *also* get id 0, silently overwriting whatever
    real document already held that id instead of being added or
    correctly updating itself. Deriving the id from `doc_id` instead
    (UUID5, deterministic for a given string) makes every call safe: the
    same `doc_id` always maps to the same point (a real update in place),
    and different `doc_id`s never collide.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, doc_id))


class QdrantStore:
    def __init__(self, url: str | None = None, collection: str | None = None) -> None:
        from qdrant_client import QdrantClient

        self.collection = collection or settings.qdrant_collection
        self.client = QdrantClient(url=url or settings.qdrant_url)

    def ensure_collection(self, vector_size: int) -> None:
        from qdrant_client.models import Distance, VectorParams

        if self.client.collection_exists(self.collection):
            return
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

    def upsert(self, doc_ids: list[str], vectors: list[list[float]], payloads: list[dict]) -> None:
        from qdrant_client.models import PointStruct

        self.ensure_collection(vector_size=len(vectors[0]))
        points = [
            PointStruct(id=_point_id(doc_id), vector=vector, payload={**payload, "doc_id": doc_id})
            for doc_id, vector, payload in zip(doc_ids, vectors, payloads, strict=True)
        ]
        self.client.upsert(collection_name=self.collection, points=points)

    def search(self, query_vector: list[float], top_k: int = 5, allowed_doc_ids: set[str] | None = None) -> list[dict]:
        query_filter = None
        if allowed_doc_ids is not None:
            from qdrant_client.models import FieldCondition, Filter, MatchAny

            query_filter = Filter(
                must=[FieldCondition(key="doc_id", match=MatchAny(any=list(allowed_doc_ids)))]
            )

        results = self.client.query_points(
            collection_name=self.collection, query=query_vector, limit=top_k, query_filter=query_filter
        ).points
        return [{"doc_id": r.payload["doc_id"], "score": r.score, "payload": r.payload} for r in results]

    def count(self) -> int:
        if not self.client.collection_exists(self.collection):
            return 0
        return self.client.count(self.collection).count
