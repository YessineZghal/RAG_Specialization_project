from __future__ import annotations

from rrf import reciprocal_rank_fusion


def test_rrf_boosts_docs_ranked_highly_in_multiple_lists():
    ranking_a = [("doc1", 0.9), ("doc2", 0.8), ("doc3", 0.7)]
    ranking_b = [("doc2", 50.0), ("doc1", 40.0), ("doc4", 30.0)]

    fused = reciprocal_rank_fusion([ranking_a, ranking_b])
    fused_ids = [doc_id for doc_id, _ in fused]

    # doc1 and doc2 both appear near the top of both lists -> should lead.
    assert set(fused_ids[:2]) == {"doc1", "doc2"}


def test_rrf_ignores_raw_score_magnitude():
    # ranking_b's scores are on a wildly different scale, but rank-only
    # fusion should not let that dominate.
    ranking_a = [("doc1", 0.99)]
    ranking_b = [("doc2", 1_000_000.0), ("doc1", 999_999.0)]

    fused = dict(reciprocal_rank_fusion([ranking_a, ranking_b]))
    # doc1 is rank 1 in A and rank 2 in B; doc2 is rank 1 in B only.
    assert fused["doc1"] > fused["doc2"]


def test_rrf_single_ranking_preserves_order():
    ranking = [("a", 3.0), ("b", 2.0), ("c", 1.0)]
    fused = reciprocal_rank_fusion([ranking])
    assert [doc_id for doc_id, _ in fused] == ["a", "b", "c"]
