from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kag_common.dataset import _join_contexts


def test_join_contexts_prefixes_each_segment_with_its_real_label():
    context = {"contexts": ["Some background.", "Some results."], "labels": ["BACKGROUND", "RESULTS"]}
    assert _join_contexts(context) == "BACKGROUND: Some background. RESULTS: Some results."


def test_join_contexts_normalizes_extra_whitespace_within_a_segment():
    context = {"contexts": ["  extra   spaces   here  "], "labels": ["BACKGROUND"]}
    assert _join_contexts(context) == "BACKGROUND: extra spaces here"


def test_join_contexts_without_labels_omits_the_prefix():
    context = {"contexts": ["No label here."], "labels": []}
    assert _join_contexts(context) == "No label here."
