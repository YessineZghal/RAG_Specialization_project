from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "dynamic-retrieval"))
from dynamic_top_k import dynamic_top_k  # noqa: E402


def test_dynamic_top_k_none_is_zero():
    assert dynamic_top_k("none") == 0


def test_dynamic_top_k_increases_with_complexity():
    assert dynamic_top_k("simple") < dynamic_top_k("complex")


def test_dynamic_top_k_unknown_label_falls_back_to_simple():
    assert dynamic_top_k("unknown_label") == dynamic_top_k("simple")


def test_run_policy_none_skips_retrieval(fake_retriever, fake_llm):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "dynamic-retrieval"))
    from retrieval_policy import run_policy

    result = run_policy("hi there!", fake_retriever, llm=fake_llm())
    assert result["strategy"] == "no_retrieval"
    assert result["results"] == []


def test_run_policy_simple_does_one_retrieval(fake_retriever, fake_llm, tiny_corpus):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "dynamic-retrieval"))
    from retrieval_policy import run_policy

    result = run_policy("Where is Russell Hobbs based?", fake_retriever, llm=fake_llm())
    assert result["strategy"] == "single_retrieval"
    assert result["results"]
