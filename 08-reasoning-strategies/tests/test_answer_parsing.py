from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reasoning_common.answer_parsing import parse_yes_no


def test_parse_yes_no_reads_a_clean_final_line():
    assert parse_yes_no("Some reasoning here.\nAnswer: Yes") is True
    assert parse_yes_no("Some reasoning here.\nAnswer: No") is False


def test_parse_yes_no_is_word_boundary_safe_not_a_substring_check():
    # "yes" is not a substring of anything problematic here, but this
    # guards the exact bug class this repo has hit more than once: a
    # naive `"no" in text` would misfire on words merely containing "no".
    assert parse_yes_no("Answer: Nonetheless, this is unclear") is None
    assert parse_yes_no("Answer: Notable but irrelevant") is None


def test_parse_yes_no_prefers_the_last_line_over_earlier_reasoning():
    # Earlier reasoning says "no"; the actual concluding line says "Yes" --
    # the final line must win.
    multi_line = "There is no evidence for this initially.\nAnswer: Yes"
    assert parse_yes_no(multi_line) is True


def test_parse_yes_no_falls_back_to_the_whole_text_when_the_last_line_is_ambiguous():
    text = "Reasoning: this seems true.\nSomething else entirely."
    assert parse_yes_no(text) is True


def test_parse_yes_no_returns_none_when_neither_appears():
    assert parse_yes_no("The reasoning is inconclusive.") is None


def test_parse_yes_no_returns_none_when_both_appear_ambiguously_on_the_last_line():
    assert parse_yes_no("Answer: yes and no, it depends") is None


def test_parse_yes_no_accepts_true_false_as_synonyms():
    assert parse_yes_no("Answer: True") is True
    assert parse_yes_no("Answer: False") is False
