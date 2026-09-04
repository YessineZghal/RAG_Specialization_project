from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "reasoning-engine"))
from language_reasoning_op import reason_in_language  # noqa: E402


def test_reason_in_language_parses_a_clear_yes_verdict(fake_llm):
    llm = fake_llm(response="The evidence supports it.\nAnswer: Yes")
    result = reason_in_language("Is X true?", "some evidence", llm)
    assert result.verdict == "yes"


def test_reason_in_language_parses_a_clear_no_verdict(fake_llm):
    llm = fake_llm(response="The evidence contradicts it.\nAnswer: No")
    result = reason_in_language("Is X true?", "some evidence", llm)
    assert result.verdict == "no"


def test_reason_in_language_parses_a_maybe_verdict(fake_llm):
    llm = fake_llm(response="The evidence is mixed.\nAnswer: Maybe")
    result = reason_in_language("Is X true?", "some evidence", llm)
    assert result.verdict == "maybe"


def test_reason_in_language_handles_no_evidence_gracefully(fake_llm):
    llm = fake_llm(response="Answer: Maybe")
    result = reason_in_language("Is X true?", "", llm)
    assert "(no evidence retrieved)" in llm.calls[0]
    assert result.verdict == "maybe"
