"""Normalized Discounted Cumulative Gain (NDCG@K).

Unlike Recall@K (binary hit/miss) and MRR (only the *first* hit), NDCG
credits every relevant document in the Top-K, discounted by how far down
the ranking it appears, and normalizes against the best possible ordering
— so it rewards ranking *all* relevant documents near the top, not just
finding one.
"""

from __future__ import annotations

import math


def _dcg(gains: list[float]) -> float:
    return sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, start=1))


def ndcg_at_k(
    results: dict[str, list[str]],
    qrels: dict[str, dict[str, int]],
    k: int,
) -> float:
    evaluable = [qid for qid in results if qrels.get(qid)]
    if not evaluable:
        return 0.0

    scores = []
    for qid in evaluable:
        relevance = qrels[qid]
        ranked = results[qid][:k]

        gains = [relevance.get(doc_id, 0) for doc_id in ranked]
        dcg = _dcg(gains)

        ideal_gains = sorted(relevance.values(), reverse=True)[:k]
        idcg = _dcg(ideal_gains)

        scores.append(dcg / idcg if idcg > 0 else 0.0)
    return sum(scores) / len(scores)
