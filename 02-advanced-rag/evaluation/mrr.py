"""Mean Reciprocal Rank (MRR) — rewards ranking the *first* relevant
document as early as possible: 1/rank of the first hit, averaged over
queries, 0 if no relevant doc appears at all.
"""

from __future__ import annotations


def reciprocal_rank(ranked_doc_ids: list[str], relevant_doc_ids: set[str]) -> float:
    for rank, doc_id in enumerate(ranked_doc_ids, start=1):
        if doc_id in relevant_doc_ids:
            return 1.0 / rank
    return 0.0


def mrr(
    results: dict[str, list[str]],
    qrels: dict[str, dict[str, int]],
    k: int | None = None,
) -> float:
    evaluable = [qid for qid in results if qrels.get(qid)]
    if not evaluable:
        return 0.0

    scores = []
    for qid in evaluable:
        relevant = {doc_id for doc_id, rel in qrels[qid].items() if rel > 0}
        ranked = results[qid][:k] if k else results[qid]
        scores.append(reciprocal_rank(ranked, relevant))
    return sum(scores) / len(scores)
