from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reasoning_common.dataset import split_sentences


def test_split_sentences_splits_on_sentence_boundaries():
    text = "The sky is blue. Water is wet. Fire is hot."
    assert split_sentences(text) == ["The sky is blue.", "Water is wet.", "Fire is hot."]


def test_split_sentences_normalizes_extra_whitespace():
    text = "  The sky is blue.   Water is wet.  "
    assert split_sentences(text) == ["The sky is blue.", "Water is wet."]


def test_split_sentences_handles_question_and_exclamation_marks():
    text = "Is this true? Yes it is! Confirmed."
    assert split_sentences(text) == ["Is this true?", "Yes it is!", "Confirmed."]


def test_split_sentences_on_empty_text_returns_empty_list():
    assert split_sentences("") == []
    assert split_sentences("   ") == []


def test_split_sentences_on_a_single_sentence():
    assert split_sentences("Just one sentence.") == ["Just one sentence."]
