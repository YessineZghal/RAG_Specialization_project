"""Tests for the mini project's dispatch logic (`run_auto`) -- verifies
it routes to the *correct* strategy function for each classification and
assembles the result correctly. The strategy functions themselves
(tree/graph search, HGoT decomposition) are already covered by their own
test files; here they're monkeypatched so this file tests only the new
wiring, not re-verifies already-tested internals.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))
import strategy_selector


class FakeRetriever:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def search(self, query, top_k=5):
        self.calls.append(query)
        return self.results[:top_k]


def test_run_auto_dispatches_to_cot_when_classified_simple(fake_llm, monkeypatch):
    llm = fake_llm(response="simple")
    monkeypatch.setattr(strategy_selector, "cot_answer", lambda q, c, llm=None: {"answer": True, "reasoning": "r", "llm_calls": 1})

    retriever = FakeRetriever([("d1", 0.9)])
    corpus = {"d1": "some fact"}
    result = strategy_selector.run_auto("some question", corpus, retriever, llm)

    assert result["strategy"] == "cot"
    assert result["answer"] is True
    assert result["llm_calls"] == 2  # 1 classify + 1 cot
    assert retriever.calls == ["some question"]


def test_run_auto_dispatches_to_tot_when_classified_comparative(fake_llm, monkeypatch):
    llm = fake_llm(response="comparative")
    monkeypatch.setattr(
        strategy_selector, "tree_of_thought_search",
        lambda q, c, llm=None: {"answer": False, "best_path": ["a", "b"], "reasoning": "r", "llm_calls": 5},
    )

    retriever = FakeRetriever([("d1", 0.9)])
    result = strategy_selector.run_auto("some question", {"d1": "fact"}, retriever, llm)

    assert result["strategy"] == "tot"
    assert result["best_path"] == ["a", "b"]
    assert result["llm_calls"] == 6  # 1 classify + 5 from the search


def test_run_auto_dispatches_to_got_when_classified_combinatorial(fake_llm, monkeypatch):
    llm = fake_llm(response="combinatorial")
    monkeypatch.setattr(
        strategy_selector, "graph_of_thought_search",
        lambda q, c, llm=None: {"answer": True, "best_path": ["a"], "graph_size": 3, "reasoning": "r", "llm_calls": 7},
    )

    retriever = FakeRetriever([("d1", 0.9)])
    result = strategy_selector.run_auto("some question", {"d1": "fact"}, retriever, llm)

    assert result["strategy"] == "got"
    assert result["graph_size"] == 3
    assert result["llm_calls"] == 8


def test_run_auto_dispatches_to_hgot_when_classified_multi_hop(fake_llm, monkeypatch):
    llm = fake_llm(response="multi_hop")
    monkeypatch.setattr(
        strategy_selector, "hgot_answer",
        lambda q, retriever, corpus, llm=None: {
            "answer": True, "sub_questions": ["q1", "q2"], "reasoning": "r", "llm_calls": 4,
        },
    )

    retriever = FakeRetriever([("d1", 0.9)])
    result = strategy_selector.run_auto("some question", {"d1": "fact"}, retriever, llm)

    assert result["strategy"] == "hgot"
    assert result["sub_questions"] == ["q1", "q2"]
    assert result["llm_calls"] == 5
    assert retriever.calls == []  # hgot does its own retrieval -- run_auto must not pre-retrieve for it


def test_run_auto_falls_back_to_cot_on_an_unparseable_classification(fake_llm, monkeypatch):
    llm = fake_llm(response="not a category at all")
    monkeypatch.setattr(strategy_selector, "cot_answer", lambda q, c, llm=None: {"answer": None, "reasoning": "r", "llm_calls": 1})

    retriever = FakeRetriever([("d1", 0.9)])
    result = strategy_selector.run_auto("some question", {"d1": "fact"}, retriever, llm)

    assert result["strategy"] == "cot"
