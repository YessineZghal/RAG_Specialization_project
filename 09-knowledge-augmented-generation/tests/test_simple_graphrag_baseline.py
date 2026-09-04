from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kag_eval.simple_graphrag_baseline import (
    baseline_answer,
    build_unconstrained_graph,
    extract_unconstrained,
)


def test_extract_unconstrained_keeps_whatever_types_the_model_proposes(fake_llm):
    # "Drug" and "MedicalCondition" are not in the level's real schema --
    # the whole point of this baseline is that nothing rejects them
    payload = {
        "entities": [
            {"name": "Metformin", "type": "Drug"},
            {"name": "Diabetes", "type": "MedicalCondition"},
        ],
        "relations": [{"subject": "Metformin", "relation": "treats", "object": "Diabetes"}],
    }
    llm = fake_llm(response=json.dumps(payload))

    entities, relations = extract_unconstrained("d1", "some text", llm)

    assert {e["type"] for e in entities} == {"Drug", "MedicalCondition"}
    assert relations[0]["relation"] == "treats"


def test_extract_unconstrained_drops_a_relation_whose_object_is_not_a_string(fake_llm):
    # a real crash caught by actually running this against live Ollama
    # output: an unconstrained prompt sometimes answers "object" with a
    # JSON list instead of one string -- must be dropped, not crash
    # downstream `.strip()` calls in `build_unconstrained_graph`
    payload = {
        "entities": [],
        "relations": [{"subject": "Study-1", "relation": "found", "object": ["A", "B"]}],
    }
    llm = fake_llm(response=json.dumps(payload))

    _, relations = extract_unconstrained("d1", "text", llm)

    assert relations == []


def test_extract_unconstrained_drops_malformed_entries_but_does_not_crash(fake_llm):
    payload = {"entities": [{"name": "X"}], "relations": []}  # missing "type"
    llm = fake_llm(response=json.dumps(payload))

    entities, relations = extract_unconstrained("d1", "text", llm)

    assert entities == []
    assert relations == []


def test_build_unconstrained_graph_adds_placeholder_nodes_for_undeclared_relation_endpoints(fake_llm):
    # unlike the schema-constrained builder, the baseline has no
    # validator to reject a relation whose entities were never declared
    # -- it creates a placeholder node instead, since nothing here checks
    payload = {
        "entities": [],
        "relations": [{"subject": "Metformin", "relation": "treats", "object": "Diabetes"}],
    }
    llm = fake_llm(response=json.dumps(payload))

    graph = build_unconstrained_graph({"d1": "text"}, llm)

    assert graph.number_of_nodes() == 2
    assert graph.nodes["metformin"]["type"] == "unknown"
    assert graph.number_of_edges() == 1


def test_baseline_answer_combines_retrieval_and_naive_graph_facts(fake_llm, fake_embedder):
    import networkx as nx

    graph = nx.MultiDiGraph()
    graph.add_node("study-1", name="Study-1", type="Study")
    graph.add_node("diabetes", name="Diabetes", type="unknown")
    graph.add_edge("study-1", "diabetes", relation="studies", doc_id="d1")

    corpus = {"d1": "Study-1 studied diabetes patients."}
    doc_ids = ["d1"]
    matrix = np.array([[1.0, 0.0]], dtype=np.float32)
    embedder = fake_embedder(vectors={"Does the study concern diabetes?": [1.0, 0.0]})
    llm = fake_llm(response="Reasoning...\nAnswer: Yes")

    answer = baseline_answer("Does the study concern diabetes?", corpus, doc_ids, matrix, graph, embedder, llm)

    assert answer.verdict == "yes"
    assert "d1" in answer.citations
    assert "Graph facts" in llm.calls[0]
