from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "graph-of-thought"))
from thought_graph import ThoughtGraph


def test_add_thought_with_no_parents_is_a_root():
    graph = ThoughtGraph()
    root_id = graph.add_thought("root thought", score=1.0)
    assert graph.text(root_id) == "root thought"
    assert graph.score(root_id) == 1.0
    assert graph.parents(root_id) == []


def test_add_thought_records_a_single_parent():
    graph = ThoughtGraph()
    root_id = graph.add_thought("root")
    child_id = graph.add_thought("child", parents=[root_id], score=0.5)
    assert graph.parents(child_id) == [root_id]


def test_add_thought_can_have_multiple_parents_a_real_merge():
    graph = ThoughtGraph()
    a = graph.add_thought("branch A")
    b = graph.add_thought("branch B")
    merged = graph.add_thought("synthesis of A and B", parents=[a, b], score=0.9)
    assert set(graph.parents(merged)) == {a, b}


def test_path_to_follows_a_simple_chain_root_first():
    graph = ThoughtGraph()
    root = graph.add_thought("root")
    mid = graph.add_thought("middle step", parents=[root])
    leaf = graph.add_thought("final step", parents=[mid])
    assert graph.path_to(leaf) == ["root", "middle step", "final step"]


def test_path_to_a_root_is_just_that_node():
    graph = ThoughtGraph()
    root = graph.add_thought("only node")
    assert graph.path_to(root) == ["only node"]


def test_path_to_a_merged_node_follows_its_first_parent():
    graph = ThoughtGraph()
    root = graph.add_thought("root")
    a = graph.add_thought("branch A", parents=[root])
    b = graph.add_thought("branch B", parents=[root])
    merged = graph.add_thought("merged", parents=[a, b])
    assert graph.path_to(merged) == ["root", "branch A", "merged"]


def test_len_counts_every_node():
    graph = ThoughtGraph()
    graph.add_thought("a")
    graph.add_thought("b")
    assert len(graph) == 2
