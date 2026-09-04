from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tree-of-thought"))
from thought_generator import format_partial_path, generate_thoughts  # noqa: E402


def test_format_partial_path_with_no_steps_yet():
    assert format_partial_path([]) == "(nothing yet -- this is the first step)"


def test_format_partial_path_joins_steps_with_newlines():
    assert format_partial_path(["step one", "step two"]) == "step one\nstep two"


def test_generate_thoughts_splits_lines_and_strips_bullets(fake_llm):
    llm = fake_llm(response="1. first thought\n2. second thought\n- third thought")
    thoughts = generate_thoughts("question", "context", [], k=3, llm=llm)
    assert thoughts == ["first thought", "second thought", "third thought"]


def test_generate_thoughts_respects_k_even_if_the_model_returns_more(fake_llm):
    llm = fake_llm(response="one\ntwo\nthree\nfour")
    thoughts = generate_thoughts("question", "context", [], k=2, llm=llm)
    assert len(thoughts) == 2


def test_generate_thoughts_falls_back_to_the_raw_response_if_unparseable(fake_llm):
    llm = fake_llm(response="   ")
    thoughts = generate_thoughts("question", "context", [], k=3, llm=llm)
    assert thoughts == [""]  # empty response, stripped -- degrades, does not crash


def test_generate_thoughts_includes_the_partial_path_in_the_prompt(fake_llm):
    llm = fake_llm(response="next thought")
    generate_thoughts("q", "c", ["earlier step one"], k=1, llm=llm)
    assert "earlier step one" in llm.calls[0]
