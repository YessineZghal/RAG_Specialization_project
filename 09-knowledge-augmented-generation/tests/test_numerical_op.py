from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "reasoning-engine"))
from numerical_op import evaluate_numeric  # noqa: E402


def _build_graph():
    graph = nx.MultiDiGraph()
    graph.add_node("study-1", name="Study-1", type="Study", attributes={})
    graph.add_node("study-2", name="Study-2", type="Study", attributes={})
    graph.add_node("diabetes", name="Diabetes", type="Condition", attributes={})
    graph.add_node("pop-1", name="Pop-1", type="Population", attributes={"size": 300})
    graph.add_node("pop-2", name="Pop-2", type="Population", attributes={"size": 620})

    graph.add_edge("study-1", "diabetes", relation="STUDIES", doc_id="d1")
    graph.add_edge("study-2", "diabetes", relation="STUDIES", doc_id="d2")
    graph.add_edge("study-1", "pop-1", relation="HAS_POPULATION", doc_id="d1")
    graph.add_edge("study-2", "pop-2", relation="HAS_POPULATION", doc_id="d2")
    return graph


def test_evaluate_numeric_threshold_true_when_any_matched_population_exceeds_it():
    graph = _build_graph()
    result = evaluate_numeric(graph, "diabetes", {"attribute": "size", "op": ">", "value": 500})
    assert result.comparison_result is True
    assert result.values == {"pop-1": 300, "pop-2": 620}


def test_evaluate_numeric_threshold_false_when_no_matched_population_exceeds_it():
    graph = _build_graph()
    result = evaluate_numeric(graph, "diabetes", {"attribute": "size", "op": ">", "value": 1000})
    assert result.comparison_result is False


def test_evaluate_numeric_resolves_focus_hint_directly_on_a_study_node():
    graph = _build_graph()
    result = evaluate_numeric(graph, "study-1", {"attribute": "size", "op": ">", "value": 200})
    assert result.values == {"pop-1": 300}
    assert result.comparison_result is True


def test_evaluate_numeric_resolves_focus_hint_directly_on_a_population_node():
    graph = _build_graph()
    result = evaluate_numeric(graph, "pop-2", {"attribute": "size", "op": ">", "value": 500})
    assert result.values == {"pop-2": 620}


def test_evaluate_numeric_falls_back_to_every_population_node_when_hint_matches_nothing():
    graph = _build_graph()
    result = evaluate_numeric(graph, "cancer", {"attribute": "size", "op": ">", "value": 500})
    assert result.values == {"pop-1": 300, "pop-2": 620}
    assert result.comparison_result is True


def test_evaluate_numeric_max_returns_the_extreme_node():
    graph = _build_graph()
    result = evaluate_numeric(graph, "diabetes", {"attribute": "size", "op": "max"})
    assert result.extreme_node == "pop-2"
    assert result.comparison_result is None


def test_evaluate_numeric_min_returns_the_extreme_node():
    graph = _build_graph()
    result = evaluate_numeric(graph, "diabetes", {"attribute": "size", "op": "min"})
    assert result.extreme_node == "pop-1"


def test_evaluate_numeric_on_missing_attribute_reports_no_values():
    graph = _build_graph()
    result = evaluate_numeric(graph, "diabetes", {"attribute": "duration_weeks", "op": ">", "value": 4})
    assert result.values == {}
    assert "No 'duration_weeks' attribute" in result.explanation
