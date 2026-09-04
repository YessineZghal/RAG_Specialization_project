from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "planning"))
from agent_planner import make_plan


def test_make_plan_splits_lines(fake_llm):
    llm = fake_llm(response="Search the corpus\nCheck the result\nAnswer the question")
    steps = make_plan("some question", llm=llm)
    assert steps == ["Search the corpus", "Check the result", "Answer the question"]


def test_make_plan_filters_preamble_lines(fake_llm):
    llm = fake_llm(response="Here's a 3-step plan:\nSearch the corpus\nAnswer the question")
    steps = make_plan("some question", llm=llm)
    assert steps == ["Search the corpus", "Answer the question"]


def test_make_plan_falls_back_when_empty(fake_llm):
    llm = fake_llm(response="   ")
    steps = make_plan("some question", llm=llm)
    assert len(steps) == 1
    assert "some question" in steps[0]
