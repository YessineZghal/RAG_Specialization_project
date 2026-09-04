from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "chain-of-thought"))
from cot_prompt import cot_answer  # noqa: E402


def test_cot_answer_parses_a_clean_yes(fake_llm):
    llm = fake_llm(response="The UFC uses an enclosed octagon.\nAnswer: Yes")
    result = cot_answer("Is X true?", "some context", llm=llm)
    assert result["answer"] is True
    assert result["llm_calls"] == 1


def test_cot_answer_parses_a_clean_no(fake_llm):
    llm = fake_llm(response="There is no supporting evidence.\nAnswer: No")
    result = cot_answer("Is X true?", "some context", llm=llm)
    assert result["answer"] is False


def test_cot_answer_returns_none_for_an_unparseable_response(fake_llm):
    llm = fake_llm(response="I cannot determine this from the context.")
    result = cot_answer("Is X true?", "some context", llm=llm)
    assert result["answer"] is None


def test_cot_answer_makes_exactly_one_llm_call(fake_llm):
    llm = fake_llm(response="Answer: Yes")
    cot_answer("Is X true?", "some context", llm=llm)
    assert len(llm.calls) == 1


def test_cot_answer_sends_both_context_and_question(fake_llm):
    llm = fake_llm(response="Answer: Yes")
    cot_answer("What is the question?", "THE CONTEXT TEXT", llm=llm)
    prompt = llm.calls[0]
    assert "What is the question?" in prompt
    assert "THE CONTEXT TEXT" in prompt
