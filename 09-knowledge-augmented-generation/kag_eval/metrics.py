"""Evaluation metrics for the KAG-vs-unconstrained-graph-rag comparison
-- plain accuracy plus a per-label breakdown, since PubMedQA's real
answer distribution is imbalanced (552 yes / 338 no / 110 maybe in the
full `pqa_labeled` split) and a single accuracy number can hide a system
that is only ever guessing "yes".
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field


@dataclass(frozen=True)
class EvalResult:
    accuracy: float
    n_total: int
    n_correct: int
    n_unparseable: int
    per_label: dict[str, dict[str, int]] = field(default_factory=dict)
    predicted_distribution: dict[str, int] = field(default_factory=dict)


def evaluate(predictions: dict[str, str | None], gold: dict[str, str]) -> EvalResult:
    """`predictions`/`gold` are qid -> "yes"/"no"/"maybe" (predictions
    may also be `None` for an unparseable model response, counted
    separately rather than silently marked wrong or dropped)."""
    per_label: dict[str, Counter] = {label: Counter() for label in ("yes", "no", "maybe")}
    n_correct = 0
    n_unparseable = 0
    predicted_distribution: Counter = Counter()

    for qid, gold_label in gold.items():
        predicted = predictions.get(qid)
        if predicted is None:
            n_unparseable += 1
            per_label.setdefault(gold_label, Counter())["total"] += 1
            continue
        predicted_distribution[predicted] += 1
        bucket = per_label.setdefault(gold_label, Counter())
        bucket["total"] += 1
        if predicted == gold_label:
            n_correct += 1
            bucket["correct"] += 1

    n_total = len(gold)
    return EvalResult(
        accuracy=n_correct / n_total if n_total else 0.0,
        n_total=n_total,
        n_correct=n_correct,
        n_unparseable=n_unparseable,
        per_label={label: dict(counts) for label, counts in per_label.items()},
        predicted_distribution=dict(predicted_distribution),
    )
