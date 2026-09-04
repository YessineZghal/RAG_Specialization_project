"""Accuracy and cost scoring against real StrategyQA ground truth --
pure-Python, no LLM/Ollama call of its own, matching every prior level's
`evaluation/` pattern.

This level's whole evaluation question is two-dimensional, not one: not
just "which strategy is more accurate" but "was the extra accuracy (if
any) worth the extra LLM calls it cost" -- see the README's Evaluation
plan. `compare_strategies` is built to answer both at once.
"""

from __future__ import annotations


def accuracy(results: dict[str, dict], questions: dict[str, dict]) -> dict:
    """`results`: qid -> {"answer": bool | None, "llm_calls": int, ...}.
    `questions`: qid -> {"answer": bool, ...} (real ground truth).

    A question missing from `questions`, or a result with no matching
    question, is silently excluded rather than counted as wrong --
    consistent with every prior level's evaluation metrics (Level 2's
    `recall_at_k`, for instance) only ever scoring what is genuinely
    evaluable.
    """
    evaluable = [qid for qid in results if qid in questions]
    if not evaluable:
        return {
            "accuracy": 0.0,
            "n_correct": 0,
            "n_total": 0,
            "n_unparseable": 0,
            "total_llm_calls": 0,
            "avg_llm_calls_per_question": 0.0,
        }

    correct = sum(1 for qid in evaluable if results[qid]["answer"] == questions[qid]["answer"])
    unparseable = sum(1 for qid in evaluable if results[qid]["answer"] is None)
    total_calls = sum(results[qid].get("llm_calls", 0) for qid in evaluable)

    return {
        "accuracy": correct / len(evaluable),
        "n_correct": correct,
        "n_total": len(evaluable),
        "n_unparseable": unparseable,
        "total_llm_calls": total_calls,
        "avg_llm_calls_per_question": total_calls / len(evaluable),
    }


def compare_strategies(strategy_results: dict[str, dict[str, dict]], questions: dict[str, dict]) -> dict[str, dict]:
    """`strategy_results`: strategy name -> (qid -> result), one entry per
    strategy run on the same real question sample. Returns strategy name
    -> its `accuracy()` summary, for a direct, side-by-side comparison of
    accuracy *and* cost across CoT/ToT/GoT/HGoT on equal terms.
    """
    return {name: accuracy(results, questions) for name, results in strategy_results.items()}
