from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "graph-rag"))
from graph_builder import build_graph  # noqa: E402
from graph_retrieval import describe_node, find_matching_nodes, graph_search  # noqa: E402

TRIPLES = [
    {"subject": "Ashish Vaswani", "relation": "works at", "object": "Google Brain"},
    {"subject": "Aidan Gomez", "relation": "affiliated with", "object": "University of Toronto"},
    {"subject": "Transformer", "relation": "achieves", "object": "28.4 BLEU"},
]


def test_build_graph_creates_expected_nodes_and_edges():
    graph = build_graph(TRIPLES)
    assert set(graph.nodes) == {
        "Ashish Vaswani", "Google Brain", "Aidan Gomez", "University of Toronto",
        "Transformer", "28.4 BLEU",
    }
    assert graph.number_of_edges() == 3


def test_find_matching_nodes_matches_case_insensitively():
    graph = build_graph(TRIPLES)
    matches = find_matching_nodes(graph, "Tell me about google brain")
    assert "Google Brain" in matches


def test_describe_node_returns_both_directions():
    graph = build_graph(TRIPLES)
    facts = describe_node(graph, "Google Brain")
    assert "Ashish Vaswani works at Google Brain" in facts


def test_graph_search_returns_deduplicated_facts():
    graph = build_graph(TRIPLES)
    facts = graph_search(graph, "What does the Transformer achieve?")
    assert facts == ["Transformer achieves 28.4 BLEU"]


def test_graph_search_returns_empty_for_no_match():
    graph = build_graph(TRIPLES)
    assert graph_search(graph, "completely unrelated question") == []
