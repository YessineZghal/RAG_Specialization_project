"""Pure-logic tests for equation-aware indexing in
multimodal-rag/equation_retrieval.py — synthetic page text modeled
directly on this level's real PDF's own extracted patterns (verified by
hand against the actual downloaded text), no PDF download required.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "multimodal-rag"))
from equation_retrieval import extract_equations, gloss_symbols

# Modeled directly on the real PDF's own extracted text shape: a real
# equation ends its own line with a bare "(N)"; a Big-O complexity table
# row and a citation's volume/issue number both produce the exact same
# "(digit)" shape but are followed by more same-line text, not a newline,
# and have no "=" sign nearby.
SAMPLE_PAGES = [
    "Some intro text with no equations on this page.",
    "We compute the outputs as:\nAttention(Q,K,V) = softmax(QK^T/sqrt(dk))V (1)\nThe two most common",
    "Self-Attention O(n^2 d) O(1) O(1)\nRecurrent O(n d^2) O(n) O(n)",  # false positive shape
    "Long short-term memory. Neural computation, 9(8):1735-1780.",  # citation false positive shape
    "FFN(x) = max(0, xW1 + b1)W2 + b2 (2)\nWhile the linear transformations",
]


def test_extract_equations_finds_only_real_numbered_equations():
    equations = extract_equations(SAMPLE_PAGES)
    assert set(equations.keys()) == {"equation-1", "equation-2"}


def test_extract_equations_ignores_big_o_complexity_table_rows():
    equations = extract_equations(SAMPLE_PAGES)
    assert not any("Self-Attention" in eq["text"] for eq in equations.values())


def test_extract_equations_ignores_citation_volume_issue_numbers():
    equations = extract_equations(SAMPLE_PAGES)
    assert not any("Neural computation" in eq["text"] for eq in equations.values())


def test_extract_equations_records_the_correct_page_number():
    equations = extract_equations(SAMPLE_PAGES)
    assert equations["equation-1"]["page"] == 2
    assert equations["equation-2"]["page"] == 5


def test_extract_equations_captures_the_equation_text_itself():
    equations = extract_equations(SAMPLE_PAGES)
    assert "Attention(Q,K,V)" in equations["equation-1"]["text"]
    assert "softmax" in equations["equation-1"]["text"]


def test_extract_equations_attaches_recognized_symbol_glosses():
    equations = extract_equations(SAMPLE_PAGES)
    assert "softmax normalization" in equations["equation-1"]["concepts"]
    assert "maximum function" in equations["equation-2"]["concepts"]


def test_extract_equations_on_no_equations_returns_empty():
    assert extract_equations(["Just some plain text.", "And some more."]) == {}


def test_extract_equations_keeps_only_the_first_occurrence_of_a_number():
    pages = [
        "x = y (1)\nSome real equation",
        "Later the text says (see equation (1)) again, a different shape entirely (1)\nnot real",
    ]
    equations = extract_equations(pages)
    assert equations["equation-1"]["page"] == 1  # the first (real) occurrence, not the second


def test_gloss_symbols_returns_only_symbols_actually_present():
    assert gloss_symbols("a summation like sum: sigma notation, no real symbol here") == []
    assert gloss_symbols("compute the softmax over all sqrt(dk) terms, then take the sum: ∑") == [
        "summation",
        "softmax normalization",
    ]


def test_gloss_symbols_on_plain_text_with_no_math_returns_empty():
    assert gloss_symbols("This sentence has no math symbols in it at all.") == []
