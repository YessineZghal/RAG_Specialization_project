from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "reflection"))
from reflection import reflect  # noqa: E402


def test_reflect_satisfactory(fake_llm):
    llm = fake_llm(response="satisfactory")
    result = reflect("q", ["vector_search"], "evidence", "answer", llm=llm)
    assert result["satisfactory"] is True


def test_reflect_unsatisfactory(fake_llm):
    # "satisfactory" is a literal substring of "unsatisfactory" -- must not
    # be misread as a positive judgment.
    llm = fake_llm(response="unsatisfactory")
    result = reflect("q", ["vector_search"], "evidence", "answer", llm=llm)
    assert result["satisfactory"] is False
