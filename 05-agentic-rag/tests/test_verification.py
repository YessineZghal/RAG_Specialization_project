from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "verification"))
from answer_verifier import normalize, verify_answer


def test_verify_answer_matches_embedded_alias():
    assert verify_answer("The answer is Second.", ["second"]) is True


def test_verify_answer_case_and_punctuation_insensitive():
    assert verify_answer("It's the SECOND, definitely.", ["second"]) is True


def test_verify_answer_false_when_no_alias_present():
    assert verify_answer("The answer is First.", ["second", "secs"]) is False


def test_verify_answer_ignores_empty_aliases():
    assert verify_answer("anything", ["", "  "]) is False


def test_normalize_strips_punctuation_and_lowercases():
    assert normalize("Second!") == "second"
