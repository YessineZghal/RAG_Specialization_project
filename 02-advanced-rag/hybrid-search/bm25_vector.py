"""Hybrid retrieval: dense + BM25, combined with Reciprocal Rank Fusion.

Same `.search(query, top_k)` interface as `DenseRetriever`/`BM25Retriever`
(see `retrieval/`), so it's a drop-in replacement anywhere either one is
used — including inside `reranking/` and `examples/advanced_pipeline.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

# `hybrid-search` has a hyphen and can't be dotted-imported as a package —
# every file here adds its own directory to sys.path so sibling modules
# import as plain top-level names, regardless of how this file is invoked.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rrf import reciprocal_rank_fusion  # noqa: E402


class HybridRetriever:
    def __init__(self, dense, sparse, candidate_k: int = 50, rrf_k: int = 60) -> None:
        self.dense = dense
        self.sparse = sparse
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        dense_results = self.dense.search(query, top_k=self.candidate_k)
        sparse_results = self.sparse.search(query, top_k=self.candidate_k)
        fused = reciprocal_rank_fusion([dense_results, sparse_results], k=self.rrf_k)
        return fused[:top_k]
