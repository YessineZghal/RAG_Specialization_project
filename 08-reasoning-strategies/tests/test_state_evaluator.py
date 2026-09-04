from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tree-of-thought"))
from state_evaluator import evaluate_state


def test_evaluate_state_normalizes_a_ten_point_score_to_zero_one(fake_llm):
    llm = fake_llm(response="8")
    score = evaluate_state("question", "context", ["a reasoning step"], llm=llm)
    assert score == 0.8


def test_evaluate_state_clamps_an_out_of_range_score(fake_llm):
    llm = fake_llm(response="15")  # a real model can ignore the 1-10 instruction
    score = evaluate_state("question", "context", ["step"], llm=llm)
    assert score == 1.0


def test_evaluate_state_extracts_a_number_from_a_wordy_response(fake_llm):
    llm = fake_llm(response="I would rate this a 6 out of 10.")
    score = evaluate_state("question", "context", ["step"], llm=llm)
    assert score == 0.6


def test_evaluate_state_degrades_to_zero_on_unparseable_output(fake_llm):
    llm = fake_llm(response="This reasoning path seems fine.")
    score = evaluate_state("question", "context", ["step"], llm=llm)
    assert score == 0.0


def test_evaluate_state_includes_the_path_in_the_prompt(fake_llm):
    llm = fake_llm(response="5")
    evaluate_state("q", "c", ["a distinctive reasoning step"], llm=llm)
    assert "a distinctive reasoning step" in llm.calls[0]
