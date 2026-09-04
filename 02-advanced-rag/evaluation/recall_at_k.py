"""Recall@K — of the queries with at least one relevant document, what
fraction had *any* relevant document in their Top-K results?

This is query-level (binary hit/miss per query), matching the metric
reported by Level 1's `src/cli.py evaluate` — Level 2's version is
measured against real qrels instead of a heuristic.
"""

from __future__ import annotations


def recall_at_k(
    results: dict[str, list[str]],
    qrels: dict[str, dict[str, int]],
    k: int,
) -> float:
    """`results`: query_id -> ranked list of doc_ids (already truncated to K,
    or longer — only the first `k` are considered). `qrels`: query_id ->
    {doc_id: relevance}. Returns the fraction of *evaluable* queries (those
    with a qrels entry) whose Top-K contains at least one relevant doc.
    """
    evaluable = [qid for qid in results if qrels.get(qid)]
    if not evaluable:
        return 0.0

    hits = 0
    for qid in evaluable:
        relevant = {doc_id for doc_id, rel in qrels[qid].items() if rel > 0}
        retrieved = set(results[qid][:k])
        if relevant & retrieved:
            hits += 1
    return hits / len(evaluable)
