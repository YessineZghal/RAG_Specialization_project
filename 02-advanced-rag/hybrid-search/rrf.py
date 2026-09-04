"""Reciprocal Rank Fusion (RRF).

Combines multiple *ranked lists* into one, using only each item's **rank**
in each list — not its raw score. This sidesteps the real problem with
combining dense (cosine, ~0-1) and sparse (BM25, unbounded) results: their
scores are not on comparable scales, but their *rankings* are.

    RRF(d) = sum over rankings r containing d of  1 / (k + rank_r(d))

`k` (default 60, the value used in the original RRF paper) dampens the
influence of very high ranks so the fused order isn't dominated by
whichever single list ranked one document first.
"""

from __future__ import annotations


def reciprocal_rank_fusion(
    rankings: list[list[tuple[str, float]]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Fuse multiple `[(doc_id, score), ...]` rankings into one.

    Returns `[(doc_id, rrf_score), ...]` sorted by descending `rrf_score`.
    """
    fused_scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, (doc_id, _score) in enumerate(ranking, start=1):
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
