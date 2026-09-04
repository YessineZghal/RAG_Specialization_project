from __future__ import annotations

import pytest

from evaluation.mrr import mrr, reciprocal_rank
from evaluation.ndcg import ndcg_at_k
from evaluation.recall_at_k import recall_at_k

QRELS = {
    "q1": {"docA": 1, "docB": 1},
    "q2": {"docC": 1},
    "q3": {"docD": 1},  # no results returned for q3 at all
}


def test_recall_at_k_counts_any_relevant_hit():
    results = {
        "q1": ["docX", "docA", "docY"],  # hit within top 3
        "q2": ["docZ", "docW"],  # miss
    }
    assert recall_at_k(results, QRELS, k=3) == pytest.approx(0.5)


def test_recall_at_k_respects_k_cutoff():
    results = {"q1": ["docX", "docY", "docA"]}  # relevant doc at rank 3
    assert recall_at_k(results, QRELS, k=2) == 0.0
    assert recall_at_k(results, QRELS, k=3) == 1.0


def test_recall_at_k_ignores_queries_with_no_qrels():
    results = {"q1": ["docA"], "unknown-query": ["docA"]}
    assert recall_at_k(results, QRELS, k=5) == 1.0  # only q1 is evaluable


def test_recall_at_k_empty_results_is_zero():
    assert recall_at_k({}, QRELS, k=5) == 0.0


def test_reciprocal_rank_basic():
    assert reciprocal_rank(["a", "b", "c"], {"b"}) == pytest.approx(0.5)
    assert reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0
    assert reciprocal_rank(["a", "b", "c"], {"z"}) == 0.0


def test_mrr_averages_over_evaluable_queries():
    results = {
        "q1": ["docX", "docA"],  # rank 2 -> 0.5
        "q2": ["docC"],  # rank 1 -> 1.0
    }
    assert mrr(results, QRELS) == pytest.approx(0.75)


def test_ndcg_perfect_ranking_is_one():
    results = {"q1": ["docA", "docB"]}
    assert ndcg_at_k(results, QRELS, k=2) == pytest.approx(1.0)


def test_ndcg_worse_ranking_scores_lower_than_perfect():
    perfect = ndcg_at_k({"q1": ["docA", "docB"]}, QRELS, k=2)
    worse = ndcg_at_k({"q1": ["docX", "docA"]}, QRELS, k=2)
    assert worse < perfect


def test_ndcg_no_relevant_found_is_zero():
    assert ndcg_at_k({"q2": ["docX", "docY"]}, QRELS, k=2) == 0.0
