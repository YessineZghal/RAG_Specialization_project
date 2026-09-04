from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kag_eval.metrics import evaluate  # noqa: E402


def test_evaluate_computes_plain_accuracy():
    gold = {"q1": "yes", "q2": "no", "q3": "maybe"}
    predictions = {"q1": "yes", "q2": "no", "q3": "yes"}

    result = evaluate(predictions, gold)

    assert result.n_total == 3
    assert result.n_correct == 2
    assert result.accuracy == 2 / 3


def test_evaluate_counts_unparseable_predictions_separately_not_as_wrong():
    gold = {"q1": "yes"}
    predictions = {"q1": None}

    result = evaluate(predictions, gold)

    assert result.n_unparseable == 1
    assert result.n_correct == 0
    assert result.accuracy == 0.0


def test_evaluate_handles_a_missing_prediction_as_unparseable():
    gold = {"q1": "yes"}
    predictions: dict = {}

    result = evaluate(predictions, gold)

    assert result.n_unparseable == 1


def test_evaluate_per_label_breakdown_reflects_class_imbalance():
    gold = {"q1": "yes", "q2": "yes", "q3": "no"}
    predictions = {"q1": "yes", "q2": "no", "q3": "no"}

    result = evaluate(predictions, gold)

    assert result.per_label["yes"]["total"] == 2
    assert result.per_label["yes"]["correct"] == 1
    assert result.per_label["no"]["total"] == 1
    assert result.per_label["no"]["correct"] == 1


def test_evaluate_on_empty_gold_does_not_divide_by_zero():
    result = evaluate({}, {})
    assert result.accuracy == 0.0
    assert result.n_total == 0


def test_evaluate_tracks_predicted_distribution():
    gold = {"q1": "yes", "q2": "no"}
    predictions = {"q1": "yes", "q2": "yes"}

    result = evaluate(predictions, gold)

    assert result.predicted_distribution == {"yes": 2}
