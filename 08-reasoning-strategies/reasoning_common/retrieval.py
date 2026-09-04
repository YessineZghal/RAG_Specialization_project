"""Dense retrieval over the pooled fact corpus -- the same brute-force
cosine-similarity approach every prior level's simplest retriever uses.
Reasoning strategy is this level's actual subject; retrieval quality
itself is a solved, deliberately unremarkable dependency, not something
this level tries to improve on.
"""

from __future__ import annotations

import numpy as np

from .embed import OllamaEmbedder, cosine_search, embed_texts


class DenseRetriever:
    def __init__(self, ids: list[str], matrix: np.ndarray, embedder: OllamaEmbedder) -> None:
        self.ids = ids
        self.matrix = matrix
        self.embedder = embedder

    @classmethod
    def from_corpus(
        cls, corpus: dict[str, str], embedder: OllamaEmbedder | None = None, cache_name: str = "corpus"
    ) -> "DenseRetriever":
        embedder = embedder or OllamaEmbedder()
        ids, matrix = embed_texts(corpus, embedder=embedder, cache_name=cache_name)
        return cls(ids, matrix, embedder)

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        if not self.ids:
            return []
        query_vector = np.array(self.embedder.embed_one(query), dtype=np.float32)
        return cosine_search(query_vector, self.ids, self.matrix, top_k=top_k)
