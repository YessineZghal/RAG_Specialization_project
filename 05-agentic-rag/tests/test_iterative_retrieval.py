from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "iterative-retrieval"))
from loop import is_evidence_sufficient, iterative_retrieve  # noqa: E402


def test_is_evidence_sufficient_false_for_empty_evidence(fake_llm):
    assert is_evidence_sufficient("q", "", llm=fake_llm(response="yes")) is False


def test_is_evidence_sufficient_true_on_yes(fake_llm):
    assert is_evidence_sufficient("q", "some evidence", llm=fake_llm(response="yes")) is True


def test_is_evidence_sufficient_word_boundary_not_fooled_by_substring(fake_llm):
    # "no" is a literal substring of "not" -- a naive `"no" in response`
    # check would misfire here. Word-boundary matching must see neither a
    # standalone "no" nor a standalone "yes" in this response, and fall
    # through to the default (insufficient).
    llm = fake_llm(response="Not entirely certain this covers it.")
    assert is_evidence_sufficient("q", "some evidence", llm=llm) is False


def test_is_evidence_sufficient_true_despite_containing_not_as_substring(fake_llm):
    # Conversely: a real standalone "yes" elsewhere must still be found even
    # when the response also contains "not" (which must NOT be read as "no").
    llm = fake_llm(response="Not everything is covered, but yes, this is enough.")
    assert is_evidence_sufficient("q", "some evidence", llm=llm) is True


def test_iterative_retrieve_stops_when_sufficient(fake_vector_tool, fake_llm):
    tool = fake_vector_tool()
    llm = fake_llm(response="yes")
    result = iterative_retrieve("q", tool, llm=llm, max_iterations=3)
    assert len(result["rounds"]) == 1
    assert result["rounds"][0]["sufficient"] is True
    assert tool.calls == ["q"]


def test_iterative_retrieve_widens_query_when_insufficient(fake_vector_tool, fake_llm):
    tool = fake_vector_tool()
    llm = fake_llm(responses=["no", "a rephrased query", "yes"])
    result = iterative_retrieve("q", tool, llm=llm, max_iterations=3)
    assert len(result["rounds"]) == 2
    assert tool.calls == ["q", "a rephrased query"]


def test_iterative_retrieve_respects_max_iterations(fake_vector_tool, fake_llm):
    tool = fake_vector_tool()
    llm = fake_llm(response="no")  # never sufficient
    result = iterative_retrieve("q", tool, llm=llm, max_iterations=2)
    assert len(result["rounds"]) == 2
