"""Cross-encoder reranking.

Dense/sparse/hybrid retrieval all score a query against a document
*independently* — a "bi-encoder" approach, fast enough to search
thousands of documents. A cross-encoder instead reads the (query,
document) pair *together* through one transformer, which is far more
accurate at judging true relevance — but too slow to run over an entire
corpus. The standard pattern (used here) is: retrieve wide with a cheap
retriever, rerank a small candidate set with the expensive-but-accurate
cross-encoder.

Requires `uv sync --extra sentence-transformers` (adds `torch`).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.config import settings


class CrossEncoderReranker:
    def __init__(self, model_name: str | None = None) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "The 'sentence-transformers' extra is required for reranking. "
                "Run `uv sync --extra sentence-transformers`."
            ) from exc

        self.model_name = model_name or settings.cross_encoder_model
        self._model = CrossEncoder(self.model_name)

    def rerank(
        self,
        query: str,
        candidates: list[tuple[str, str]],  # (doc_id, text)
        top_k: int | None = None,
    ) -> list[tuple[str, float]]:
        if not candidates:
            return []
        pairs = [(query, text) for _, text in candidates]
        scores = self._model.predict(pairs)
        ranked = sorted(
            zip((doc_id for doc_id, _ in candidates), scores), key=lambda item: item[1], reverse=True
        )
        ranked = [(doc_id, float(score)) for doc_id, score in ranked]
        return ranked[:top_k] if top_k else ranked
