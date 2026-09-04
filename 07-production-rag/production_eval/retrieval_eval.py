"""Recall@K / MRR / NDCG@K — same pure-Python IR metrics as Level 2's
`evaluation/`, reimplemented here so this level's regression suite has no
cross-level dependency.
"""

from __future__ import annotations

import math


def recall_at_k(results: dict[str, list[str]], qrels: dict[str, set[str]], k: int) -> float:
    evaluable = [qid for qid in results if qrels.get(qid)]
    if not evaluable:
        return 0.0
    hits = sum(1 for qid in evaluable if set(results[qid][:k]) & qrels[qid])
    return hits / len(evaluable)


def mrr(results: dict[str, list[str]], qrels: dict[str, set[str]]) -> float:
    evaluable = [qid for qid in results if qrels.get(qid)]
    if not evaluable:
        return 0.0
    total = 0.0
    for qid in evaluable:
        for rank, doc_id in enumerate(results[qid], start=1):
            if doc_id in qrels[qid]:
                total += 1.0 / rank
                break
    return total / len(evaluable)


def ndcg_at_k(results: dict[str, list[str]], qrels: dict[str, set[str]], k: int) -> float:
    evaluable = [qid for qid in results if qrels.get(qid)]
    if not evaluable:
        return 0.0

    def dcg(gains: list[float]) -> float:
        return sum(g / math.log2(i + 1) for i, g in enumerate(gains, start=1))

    scores = []
    for qid in evaluable:
        relevant = qrels[qid]
        ranked = results[qid][:k]
        gains = [1.0 if doc_id in relevant else 0.0 for doc_id in ranked]
        ideal = sorted(gains, reverse=True)
        idcg = dcg(ideal)
        scores.append(dcg(gains) / idcg if idcg > 0 else 0.0)
    return sum(scores) / len(scores)
