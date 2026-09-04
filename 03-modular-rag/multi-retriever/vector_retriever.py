"""A generic, reusable vector retriever over any named text collection.

Level 2 built one dense retriever for one corpus. Level 3's "multi-
retriever" idea is broader: a modular system may hold *several* vector
collections at once (this level's PDF chunks, table descriptions, graph
node summaries, ...) — `VectorRetriever` is the one building block all of
them share, and `retriever_fusion.py` combines their results when a
question could plausibly be answered by more than one collection.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modular_common.embed import OllamaEmbedder, cosine_search, embed_texts  # noqa: E402


class VectorRetriever:
    def __init__(self, ids: list[str], matrix, embedder: OllamaEmbedder) -> None:
        self.ids = ids
        self.matrix = matrix
        self.embedder = embedder

    @classmethod
    def from_texts(
        cls, texts: dict[str, str], embedder: OllamaEmbedder | None = None, cache_name: str = "texts"
    ) -> "VectorRetriever":
        embedder = embedder or OllamaEmbedder()
        ids, matrix = embed_texts(texts, embedder=embedder, cache_name=cache_name)
        return cls(ids, matrix, embedder)

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        if not self.ids:
            return []
        import numpy as np

        query_vector = np.array(self.embedder.embed_one(query), dtype=np.float32)
        return cosine_search(query_vector, self.ids, self.matrix, top_k=top_k)

    def __len__(self) -> int:
        return len(self.ids)
