"""Integration test for tree_of_thought_search -- scripts the exact
sequence of LLM calls (generate, evaluate, evaluate, generate, evaluate,
evaluate, final-answer) so the search's own orchestration (branching,
scoring, pruning to the top `beam_width`, stopping) is verified directly,
the same "script the fake and check the exact call sequence" approach
Level 5's agent-loop test already used.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tree-of-thought"))
from tree_search import tree_of_thought_search  # noqa: E402


def test_full_search_prunes_to_the_best_branch_at_each_depth(fake_llm):
    llm = fake_llm(
        responses=[
            "thought A\nthought B",  # depth 0: generate 2 candidates
            "3",  # depth 0: score thought A -> 0.3
            "9",  # depth 0: score thought B -> 0.9 (best -- survives beam_width=1)
            "thought C\nthought D",  # depth 1: generate from thought B's path
            "2",  # depth 1: score B+C -> 0.2
            "7",  # depth 1: score B+D -> 0.7 (best)
            "Reasoning complete.\nAnswer: Yes",  # final answer
        ]
    )

    result = tree_of_thought_search(
        "question", "context", llm=llm, branching_factor=2, max_depth=2, beam_width=1, score_threshold=0.99
    )

    assert result["best_path"] == ["thought B", "thought D"]
    assert result["best_score"] == 0.7
    assert result["answer"] is True
    assert result["llm_calls"] == 7


def test_search_stops_early_once_the_score_threshold_is_met(fake_llm):
    llm = fake_llm(
        responses=[
            "thought A\nthought B",  # depth 0: generate
            "3",  # score A -> 0.3
            "9",  # score B -> 0.9 -- clears threshold 0.5, stop here
            "Reasoning complete.\nAnswer: No",  # final answer -- no second-depth generate call
        ]
    )

    result = tree_of_thought_search(
        "question", "context", llm=llm, branching_factor=2, max_depth=5, beam_width=1, score_threshold=0.5
    )

    assert result["best_score"] == 0.9
    assert result["llm_calls"] == 4  # 1 generate + 2 evaluate + 1 final -- depth 1 never ran
    assert result["answer"] is False


def test_search_respects_max_depth_as_a_hard_cap(fake_llm):
    # A single fixed response ("5") answers every call regardless of which
    # function made it: generate_thoughts falls back to treating "5" as
    # its one thought, and evaluate_state parses it as a 0.5 score -- both
    # degrade gracefully rather than erroring, which is exactly what lets
    # this test isolate the one thing it actually checks: the loop stops
    # after precisely `max_depth` rounds even though 0.5 never clears the
    # near-impossible 0.99 threshold.
    llm = fake_llm(response="5")

    result = tree_of_thought_search(
        "question", "context", llm=llm, branching_factor=1, max_depth=3, beam_width=1, score_threshold=0.99
    )

    # 3 depths * (1 generate + 1 evaluate) + 1 final = 7
    assert result["llm_calls"] == 7


def test_search_keeps_multiple_branches_alive_when_beam_width_is_greater_than_one(fake_llm):
    llm = fake_llm(
        responses=[
            "thought A\nthought B\nthought C",  # depth 0: 3 candidates
            "9",  # A -> 0.9
            "8",  # B -> 0.8
            "1",  # C -> 0.1 -- pruned, beam_width=2 keeps only A and B
            "next from A",  # depth 1 generate, called once per surviving node -- for A
            "6",  # A+next -> 0.6
            "next from B",  # depth 1 generate -- for B
            "7",  # B+next -> 0.7 (best overall)
            "Answer: Yes",
        ]
    )

    # branching_factor=3 so all 3 scripted depth-0 thoughts actually survive
    # `generate_thoughts`'s own `thoughts[:k]` truncation -- with k=1 only
    # "thought A" would ever be kept, which was a real bug in this test,
    # not in tree_search.py, caught by actually running it.
    result = tree_of_thought_search(
        "question", "context", llm=llm, branching_factor=3, max_depth=2, beam_width=2, score_threshold=0.99
    )

    assert result["best_path"] == ["thought B", "next from B"]
    assert result["best_score"] == 0.7
