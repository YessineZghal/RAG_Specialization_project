from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reasoning_eval.metrics import accuracy, compare_strategies

QUESTIONS = {
    "q1": {"answer": True},
    "q2": {"answer": False},
    "q3": {"answer": True},
}


def test_accuracy_scores_correct_and_incorrect_answers():
    results = {
        "q1": {"answer": True, "llm_calls": 1},  # correct
        "q2": {"answer": True, "llm_calls": 1},  # wrong -- ground truth is False
        "q3": {"answer": True, "llm_calls": 1},  # correct
    }
    summary = accuracy(results, QUESTIONS)
    assert summary["accuracy"] == 2 / 3
    assert summary["n_correct"] == 2
    assert summary["n_total"] == 3


def test_accuracy_counts_unparseable_answers_as_wrong_but_tracks_them_separately():
    results = {
        "q1": {"answer": None, "llm_calls": 1},  # unparseable -- wrong, but distinct from a confident miss
        "q2": {"answer": False, "llm_calls": 1},  # correct
        "q3": {"answer": True, "llm_calls": 1},  # correct
    }
    summary = accuracy(results, QUESTIONS)
    assert summary["n_correct"] == 2
    assert summary["n_unparseable"] == 1


def test_accuracy_sums_and_averages_llm_calls():
    results = {
        "q1": {"answer": True, "llm_calls": 1},
        "q2": {"answer": False, "llm_calls": 7},
        "q3": {"answer": True, "llm_calls": 4},
    }
    summary = accuracy(results, QUESTIONS)
    assert summary["total_llm_calls"] == 12
    assert summary["avg_llm_calls_per_question"] == 4.0


def test_accuracy_excludes_results_with_no_matching_ground_truth():
    results = {
        "q1": {"answer": True, "llm_calls": 1},
        "unknown-qid": {"answer": True, "llm_calls": 1},
    }
    summary = accuracy(results, QUESTIONS)
    assert summary["n_total"] == 1  # "unknown-qid" is silently excluded, not counted as wrong


def test_accuracy_with_no_evaluable_questions_returns_zero_not_a_crash():
    summary = accuracy({}, QUESTIONS)
    assert summary["accuracy"] == 0.0
    assert summary["n_total"] == 0


def test_compare_strategies_scores_each_strategy_independently():
    cot_results = {"q1": {"answer": True, "llm_calls": 1}, "q2": {"answer": False, "llm_calls": 1}}
    tot_results = {"q1": {"answer": True, "llm_calls": 6}, "q2": {"answer": True, "llm_calls": 6}}

    comparison = compare_strategies({"cot": cot_results, "tot": tot_results}, QUESTIONS)

    assert comparison["cot"]["accuracy"] == 1.0
    assert comparison["tot"]["accuracy"] == 0.5
    assert comparison["cot"]["avg_llm_calls_per_question"] == 1.0
    assert comparison["tot"]["avg_llm_calls_per_question"] == 6.0
