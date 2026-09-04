from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "multi-retriever"))
from retriever_fusion import reciprocal_rank_fusion


def test_fusion_boosts_items_ranked_highly_across_sources():
    rankings = {
        "docs": [("a", 0.9), ("b", 0.8)],
        "graph": [("b", 0.95), ("a", 0.85)],
    }
    fused = reciprocal_rank_fusion(rankings)
    fused_ids = [item_id for _, item_id, _ in fused]
    assert set(fused_ids[:2]) == {"a", "b"}


def test_fusion_keeps_source_name_to_avoid_id_collisions():
    rankings = {
        "docs": [("1", 0.9)],
        "sql": [("1", 0.5)],  # same id string, different collection
    }
    fused = reciprocal_rank_fusion(rankings)
    sources = {source for source, _, _ in fused}
    assert sources == {"docs", "sql"}
    assert len(fused) == 2  # NOT collapsed into one


def test_fusion_single_source_preserves_order():
    rankings = {"docs": [("a", 3.0), ("b", 2.0), ("c", 1.0)]}
    fused = reciprocal_rank_fusion(rankings)
    assert [item_id for _, item_id, _ in fused] == ["a", "b", "c"]
