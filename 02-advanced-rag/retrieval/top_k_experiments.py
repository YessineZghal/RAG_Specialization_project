"""Sweep Top-K and report Recall@K for any retriever exposing
`.search(query, top_k) -> list[(doc_id, score)]` — dense, sparse, or hybrid.

This is what actually justifies a Top-K choice instead of guessing: see
`notebooks/02_dense_vs_sparse_retrieval.ipynb` and
`../evaluation/recall_at_k.py` for the metric itself.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Protocol

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 02-advanced-rag/
from evaluation.recall_at_k import recall_at_k  # noqa: E402


class Retriever(Protocol):
    def search(self, query: str, top_k: int) -> list[tuple[str, float]]: ...


def sweep_top_k(
    retriever: Retriever,
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
    k_values: tuple[int, ...] = (1, 3, 5, 10, 20),
) -> dict[int, float]:
    """Run `retriever` once per k in `k_values`, returning {k: recall@k}."""
    results: dict[int, float] = {}
    max_k = max(k_values)

    # Retrieve once at the largest K, then slice — avoids re-querying the
    # retriever (and re-embedding the query, for dense retrievers) per k.
    all_results = {qid: retriever.search(text, top_k=max_k) for qid, text in queries.items()}

    for k in k_values:
        sliced = {qid: [doc_id for doc_id, _ in ranked[:k]] for qid, ranked in all_results.items()}
        results[k] = recall_at_k(sliced, qrels, k)
    return results
