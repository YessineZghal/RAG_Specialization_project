"""production_eval/retrieval_eval.py -- recall@k / mrr / ndcg@k, pure Python."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from production_eval.retrieval_eval import mrr, ndcg_at_k, recall_at_k


def test_recall_at_k_perfect_case():
    results = {"q1": ["a", "b", "c"]}
    qrels = {"q1": {"a"}}
    assert recall_at_k(results, qrels, k=3) == 1.0


def test_recall_at_k_miss_case():
    results = {"q1": ["b", "c", "d"]}
    qrels = {"q1": {"a"}}
    assert recall_at_k(results, qrels, k=3) == 0.0


def test_recall_at_k_respects_k_cutoff():
    results = {"q1": ["b", "c", "a"]}
    qrels = {"q1": {"a"}}
    assert recall_at_k(results, qrels, k=2) == 0.0
    assert recall_at_k(results, qrels, k=3) == 1.0


def test_mrr_rewards_earlier_rank():
    results = {"q1": ["a", "b"], "q2": ["b", "a"]}
    qrels = {"q1": {"a"}, "q2": {"a"}}
    assert mrr(results, qrels) == (1.0 + 0.5) / 2


def test_ndcg_at_k_is_one_for_perfect_ranking():
    results = {"q1": ["a", "b"]}
    qrels = {"q1": {"a", "b"}}
    assert ndcg_at_k(results, qrels, k=2) == 1.0


def test_queries_with_no_qrels_are_excluded_not_zero_scored():
    results = {"q1": ["a"], "q2": ["x"]}
    qrels = {"q1": {"a"}}  # q2 has no gold labels at all
    assert recall_at_k(results, qrels, k=1) == 1.0
    assert mrr(results, qrels) == 1.0
