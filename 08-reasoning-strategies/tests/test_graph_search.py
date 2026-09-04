"""Integration tests for graph_of_thought_search -- same scripted-call-
sequence approach as tests/test_tree_search.py, plus a direct structural
check that aggregation produces a real multi-parent node in the graph,
not just a plausible-looking `llm_calls` count.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "graph-of-thought"))
from graph_search import aggregate_thoughts, graph_of_thought_search


def test_aggregate_thoughts_sends_both_branches_and_returns_the_llms_synthesis(fake_llm):
    llm = fake_llm(response="A synthesis of both branches.")
    result = aggregate_thoughts("branch A text", "branch B text", "question", "context", llm)
    assert result == "A synthesis of both branches."
    prompt = llm.calls[0]
    assert "branch A text" in prompt
    assert "branch B text" in prompt


def test_search_with_max_depth_one_never_aggregates(fake_llm):
    # depth < max_depth - 1 is 0 < 0, always False when max_depth=1 --
    # aggregation must never trigger, regardless of how many candidates
    # survive pruning. This should behave exactly like a one-round
    # Tree-of-Thought search.
    llm = fake_llm(
        responses=[
            "thought A\nthought B",  # generate
            "3",  # A -> 0.3
            "9",  # B -> 0.9
            "Answer: Yes",  # final -- no aggregate call, no depth-1 call
        ]
    )

    result = graph_of_thought_search(
        "question", "context", llm=llm, branching_factor=2, max_depth=1, beam_width=1, score_threshold=0.99
    )

    assert result["llm_calls"] == 4
    assert result["best_path"] == ["thought B"]
    assert result["best_score"] == 0.9
    assert result["answer"] is True
    assert result["graph_size"] == 3  # root + thought A + thought B, no merged node


def test_search_with_two_depths_aggregates_once_and_produces_a_real_multi_parent_node(fake_llm):
    llm = fake_llm(
        responses=[
            "thought A\nthought B",  # depth 0: generate
            "6",  # A -> 0.6
            "9",  # B -> 0.9
            "merged thought X",  # aggregate(B, A)
            "9.5",  # score the merged thought -> 0.95
            "thought E\nthought F",  # depth 1: generate from the merged node
            "4",  # E -> 0.4
            "5",  # F -> 0.5
            "thought G\nthought H",  # depth 1: generate from node B (still in the beam)
            "2",  # G -> 0.2
            "3",  # H -> 0.3
            "Answer: No",  # final
        ]
    )

    result = graph_of_thought_search(
        "question", "context", llm=llm, branching_factor=2, max_depth=2, beam_width=2, score_threshold=0.99
    )

    assert result["llm_calls"] == 12
    # root(1) + A,B(2) + merged(1) + E,F(2) + G,H(2) = 8
    assert result["graph_size"] == 8

    graph = result["graph"]
    merge_nodes = [n for n in graph.graph.nodes if len(graph.parents(n)) == 2]
    assert len(merge_nodes) == 1  # exactly the one real merge this run produced
    assert graph.text(merge_nodes[0]) == "merged thought X"


def test_search_still_stops_early_when_the_merged_node_clears_the_threshold(fake_llm):
    llm = fake_llm(
        responses=[
            "thought A\nthought B",  # depth 0: generate
            "6",  # A -> 0.6
            "7",  # B -> 0.7
            "merged thought X",  # aggregate
            "9",  # merged -> 0.9, clears threshold 0.8 -- stop before depth 1
            "Answer: Yes",  # final
        ]
    )

    result = graph_of_thought_search(
        "question", "context", llm=llm, branching_factor=2, max_depth=5, beam_width=2, score_threshold=0.8
    )

    assert result["llm_calls"] == 6
    assert result["best_score"] == 0.9
    # The merged node's path follows its first parent (B, the higher-
    # scored of the two branches that were merged) back to the root, then
    # appends the synthesized text itself.
    assert result["best_path"] == ["thought B", "merged thought X"]
