"""production_eval/regression_suite.py -- baseline save/load + drop detection."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import production_eval.regression_suite as regression_suite
from production_eval.regression_suite import check_regression


def test_no_baseline_reports_no_baseline_status(monkeypatch):
    # `baseline=None` alone isn't enough to exercise this path: the real
    # signature treats an explicit None as "go load the default baseline
    # from disk" (see check_regression's first line), and this repo's own
    # baseline_metrics.json genuinely exists on disk -- so it would load
    # that real file instead. Monkeypatch load_baseline itself to model
    # the actual "no baseline saved yet" case.
    monkeypatch.setattr(regression_suite, "load_baseline", lambda: None)
    result = check_regression({"recall_at_5": 0.9})
    assert result["status"] == "no_baseline"
    assert result["regressions"] == []


def test_small_drop_within_tolerance_is_ok():
    baseline = {"recall_at_5": 0.90}
    current = {"recall_at_5": 0.87}  # 0.03 drop, tolerance is 0.05
    result = check_regression(current, baseline=baseline)
    assert result["status"] == "ok"


def test_large_drop_beyond_tolerance_is_flagged():
    baseline = {"recall_at_5": 0.90}
    current = {"recall_at_5": 0.70}  # 0.20 drop
    result = check_regression(current, baseline=baseline)
    assert result["status"] == "regression"
    assert result["regressions"][0]["metric"] == "recall_at_5"


def test_improvement_is_never_flagged_as_regression():
    baseline = {"recall_at_5": 0.70}
    current = {"recall_at_5": 0.95}
    result = check_regression(current, baseline=baseline)
    assert result["status"] == "ok"


def test_metric_missing_from_baseline_is_ignored_not_crashed():
    baseline = {"recall_at_5": 0.90}
    current = {"recall_at_5": 0.90, "a_brand_new_metric": 0.10}
    result = check_regression(current, baseline=baseline)
    assert result["status"] == "ok"
