from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "routing"))
from rule_router import rule_route  # noqa: E402


@pytest.mark.parametrize(
    "question,expected",
    [
        ("When was the Attention Is All You Need paper published?", "api"),
        ("What is the arXiv id of this paper?", "api"),
        ("Can you give me the DOI for this paper?", "api"),
        ("How many tracks are in the database?", "sql"),
        ("What is the total revenue from invoices?", "sql"),
        ("List the top 5 genres by number of tracks.", "sql"),
        ("Who is affiliated with Google Brain?", "graph"),
        ("Who wrote this paper?", "graph"),
        ("What is the latest news on transformers?", "web"),
        ("What happened today in AI research?", "web"),
        ("What is the Transformer architecture based on?", "documents"),
        ("Explain multi-head attention.", "documents"),
    ],
)
def test_rule_route_classifies_correctly(question, expected):
    assert rule_route(question) == expected


def test_rule_route_defaults_to_documents_for_unmatched_questions():
    assert rule_route("asdkjfh qiweuroiqwuer") == "documents"
