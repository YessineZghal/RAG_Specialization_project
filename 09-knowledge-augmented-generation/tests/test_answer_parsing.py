from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kag_common.answer_parsing import parse_yes_no_maybe  # noqa: E402


def test_parses_a_clear_yes():
    assert parse_yes_no_maybe("Reasoning...\nAnswer: Yes") == "yes"


def test_parses_a_clear_no():
    assert parse_yes_no_maybe("Reasoning...\nAnswer: No") == "no"


def test_parses_a_clear_maybe():
    assert parse_yes_no_maybe("Reasoning...\nAnswer: Maybe") == "maybe"


def test_maybe_wins_over_a_hedged_yes_or_no_mention():
    text = "It's not clearly yes or no, so maybe."
    assert parse_yes_no_maybe(text) == "maybe"


def test_returns_none_when_both_yes_and_no_appear_without_maybe():
    assert parse_yes_no_maybe("Answer: Yes, definitely not no.") is None


def test_returns_none_on_completely_unrelated_text():
    assert parse_yes_no_maybe("I cannot determine this from the context.") is None


def test_checks_the_last_line_first():
    text = "Some early reasoning mentions no evidence either way.\nAnswer: Yes"
    assert parse_yes_no_maybe(text) == "yes"


def test_word_boundary_does_not_match_substrings():
    # "nostalgia" contains "no", "yesterday" contains "yes" -- neither should trigger
    assert parse_yes_no_maybe("This is about nostalgia and yesterday.") is None
