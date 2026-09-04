from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "reasoning-engine"))
from kg_reasoning_op import find_matching_nodes, reason_over_graph  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from indexing.mutual_index import MutualIndex  # noqa: E402


def _build_graph():
    graph = nx.MultiDiGraph()
    graph.add_node("study-1", name="Study-1", type="Study", attributes={})
    graph.add_node("diabetes", name="Diabetes", type="Condition", attributes={})
    graph.add_node("metformin", name="Metformin", type="Intervention", attributes={})
    graph.add_edge("study-1", "diabetes", relation="STUDIES", doc_id="d1")
    graph.add_edge("study-1", "metformin", relation="USES_INTERVENTION", doc_id="d1")

    index = MutualIndex()
    index.add_relation("Study-1", "STUDIES", "Diabetes", "d1")
    index.add_relation("Study-1", "USES_INTERVENTION", "Metformin", "d1")
    return graph, index


def test_find_matching_nodes_matches_by_substring_on_the_display_name():
    graph, _ = _build_graph()
    assert find_matching_nodes(graph, "diabetes") == ["diabetes"]


def test_find_matching_nodes_returns_empty_for_no_hint():
    graph, _ = _build_graph()
    assert find_matching_nodes(graph, None) == []
    assert find_matching_nodes(graph, "") == []


def test_find_matching_nodes_returns_empty_when_nothing_matches():
    graph, _ = _build_graph()
    assert find_matching_nodes(graph, "cancer") == []


def test_reason_over_graph_follows_outgoing_edges_and_cites_source_docs():
    graph, index = _build_graph()

    evidence = reason_over_graph(graph, index, "study-1")

    assert "Study-1 --STUDIES--> Diabetes" in evidence.facts
    assert "Study-1 --USES_INTERVENTION--> Metformin" in evidence.facts
    assert evidence.doc_ids == {"d1"}


def test_reason_over_graph_also_follows_incoming_edges():
    graph, index = _build_graph()

    evidence = reason_over_graph(graph, index, "diabetes")

    assert "Study-1 --STUDIES--> Diabetes" in evidence.facts


def test_reason_over_graph_on_no_match_returns_empty_evidence():
    graph, index = _build_graph()

    evidence = reason_over_graph(graph, index, "cancer")

    assert evidence.facts == []
    assert evidence.doc_ids == set()
