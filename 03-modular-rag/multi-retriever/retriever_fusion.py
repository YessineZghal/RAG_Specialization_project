"""Fuse results from multiple named vector collections with Reciprocal
Rank Fusion (same algorithm as `02-advanced-rag/hybrid-search/rrf.py`,
reimplemented here so this level stays self-contained).

Useful when a question could be answered from more than one collection at
once — e.g. "what does the paper say, and who wrote it?" draws on both
document chunks and graph node text.
"""

from __future__ import annotations


def reciprocal_rank_fusion(
    rankings: dict[str, list[tuple[str, float]]],
    k: int = 60,
) -> list[tuple[str, str, float]]:
    """`rankings`: {source_name: [(item_id, score), ...]}.
    Returns `[(source_name, item_id, rrf_score), ...]` sorted by descending
    score — source is kept because the same `item_id` string could
    coincidentally collide across unrelated collections.
    """
    fused: dict[tuple[str, str], float] = {}
    for source, ranking in rankings.items():
        for rank, (item_id, _score) in enumerate(ranking, start=1):
            key = (source, item_id)
            fused[key] = fused.get(key, 0.0) + 1.0 / (k + rank)
    return sorted(
        ((source, item_id, score) for (source, item_id), score in fused.items()),
        key=lambda row: row[2],
        reverse=True,
    )
