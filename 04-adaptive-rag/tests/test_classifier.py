from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "query-classification"))
from classifier import classify_ensemble, classify_llm, classify_rule  # noqa: E402


@pytest.mark.parametrize(
    "question,expected",
    [
        ("hi there!", "none"),
        ("thanks a lot", "none"),
        ("good morning", "none"),
        ("What is the capital of France?", "simple"),
        ("Were both Gabriela Mistral and G. K. Chesterton authors?", "complex"),
        ("Which magazine was published weekly, Aeon or Life?", "complex"),
    ],
)
def test_classify_rule(question, expected):
    assert classify_rule(question) == expected


def test_classify_rule_does_not_use_missing_question_mark_as_none_signal():
    # Real HotpotQA data quirk: some genuine questions lack trailing "?" --
    # this must NOT be misread as conversational chit-chat.
    question = "Which court case occurred first, Phillips v. Martin Marietta Corp. or United States v. Miller"
    assert classify_rule(question) != "none"


def test_classify_llm_normalizes_hyphenated_multi_hop(fake_llm):
    # Models reliably answer "multi-hop" (hyphen) despite the prompt asking
    # for "multi_hop" (underscore) -- must still be recognized.
    llm = fake_llm(response="multi-hop")
    assert classify_llm("some bridge question", llm=llm) == "multi_hop"


def test_classify_llm_falls_back_to_simple_on_unparseable_output(fake_llm):
    llm = fake_llm(response="I'm not sure how to classify this.")
    assert classify_llm("some question", llm=llm) == "simple"


def test_classify_ensemble_trusts_rule_for_none_and_multi_hop(fake_llm):
    llm = fake_llm(response="simple")  # would be wrong if ever consulted
    assert classify_ensemble("hi there!", llm=llm) == "none"
    assert llm.calls == []  # rule handled it -- LLM never called


def test_classify_ensemble_defers_to_llm_for_simple_vs_complex(fake_llm):
    llm = fake_llm(response="complex")
    result = classify_ensemble("Were both X and Y authors?", llm=llm)
    assert result == "complex"
    assert len(llm.calls) == 1
