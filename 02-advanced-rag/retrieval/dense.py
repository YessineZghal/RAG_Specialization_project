"""Dense (embedding-based) retrieval.

Same cosine-similarity brute-force search as Level 1's
`InMemoryVectorStore`, but reworked around a precomputed, cached corpus
matrix (see `common/embed.py`) so it can be reused, unmodified, by
`hybrid-search/` and `reranking/` without re-embedding anything.
"""

from __future__ import annotations

import numpy as np


class DenseRetriever:
    def __init__(self, doc_ids: list[str], matrix: np.ndarray, embedder) -> None:
        self.doc_ids = doc_ids
        self.matrix = matrix.astype(np.float32)
        self._norms = np.linalg.norm(self.matrix, axis=1) + 1e-12
        self.embedder = embedder

    @classmethod
    def from_corpus(cls, corpus_texts: dict[str, str], embedder=None, cache_name: str = "corpus"):
        from common.embed import OllamaEmbedder, embed_corpus

        embedder = embedder or OllamaEmbedder()
        doc_ids, matrix = embed_corpus(corpus_texts, embedder=embedder, cache_name=cache_name)
        return cls(doc_ids, matrix, embedder)

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        query_vector = np.array(self.embedder.embed_one(query), dtype=np.float32)
        query_norm = np.linalg.norm(query_vector) + 1e-12
        scores = (self.matrix @ query_vector) / (self._norms * query_norm)

        top_k = min(top_k, len(self.doc_ids))
        top_indices = np.argpartition(-scores, top_k - 1)[:top_k]
        top_indices = top_indices[np.argsort(-scores[top_indices])]
        return [(self.doc_ids[i], float(scores[i])) for i in top_indices]
