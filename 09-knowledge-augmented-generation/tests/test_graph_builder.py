from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from indexing.graph_builder import build_graph, load_graph, neighbors_by_relation, nodes_by_type, save_graph


def _doc_response(entities, relations):
    return json.dumps({"entities": entities, "relations": relations})


def test_build_graph_assembles_nodes_and_edges_across_documents(fake_llm):
    doc1 = _doc_response(
        [
            {"name": "Study-1", "type": "Study", "attributes": {}},
            {"name": "Diabetes", "type": "Condition", "attributes": {}},
        ],
        [{"subject": "Study-1", "relation": "STUDIES", "object": "Diabetes"}],
    )
    doc2 = _doc_response(
        [
            {"name": "Study-2", "type": "Study", "attributes": {}},
            {"name": "Population-A", "type": "Population", "attributes": {"size": 620}},
        ],
        [{"subject": "Study-2", "relation": "HAS_POPULATION", "object": "Population-A"}],
    )
    llm = fake_llm(responses=[doc1, doc2])
    corpus = {"d1": "abstract one", "d2": "abstract two"}

    graph, validator, mutual_index = build_graph(corpus, llm)

    assert graph.number_of_nodes() == 4
    assert graph.number_of_edges() == 2
    assert nodes_by_type(graph, "Study") == ["study-1", "study-2"]
    assert neighbors_by_relation(graph, "study-1", "STUDIES") == ["diabetes"]
    assert graph.nodes["population-a"]["attributes"]["size"] == 620
    assert mutual_index.docs_for_entity("Diabetes") == {"d1"}
    assert validator.accepted_entities == 4


def test_build_graph_merges_attributes_of_a_repeated_entity(fake_llm):
    doc1 = _doc_response([{"name": "Diabetes", "type": "Condition", "attributes": {"note": "first"}}], [])
    doc2 = _doc_response([{"name": "Diabetes", "type": "Condition", "attributes": {"extra": "second"}}], [])
    llm = fake_llm(responses=[doc1, doc2])
    corpus = {"d1": "one", "d2": "two"}

    graph, _, mutual_index = build_graph(corpus, llm)

    assert graph.number_of_nodes() == 1
    assert graph.nodes["diabetes"]["attributes"] == {"note": "first", "extra": "second"}
    assert mutual_index.docs_for_entity("Diabetes") == {"d1", "d2"}


def test_build_graph_skips_edges_to_rejected_entities(fake_llm):
    # "Drug" is not a schema type, so the entity is rejected -- the edge
    # referencing it must not silently create a placeholder node either
    doc = _doc_response(
        [
            {"name": "Metformin", "type": "Drug", "attributes": {}},
            {"name": "Diabetes", "type": "Condition", "attributes": {}},
        ],
        [{"subject": "Metformin", "relation": "TREATS", "object": "Diabetes"}],
    )
    llm = fake_llm(response=doc)
    corpus = {"d1": "abstract"}

    graph, validator, _ = build_graph(corpus, llm)

    assert graph.number_of_nodes() == 1
    assert graph.number_of_edges() == 0
    assert validator.rejected_entities == 1


def test_save_and_load_graph_round_trips(tmp_path, fake_llm):
    doc = _doc_response(
        [
            {"name": "Study-1", "type": "Study", "attributes": {}},
            {"name": "Diabetes", "type": "Condition", "attributes": {}},
        ],
        [{"subject": "Study-1", "relation": "STUDIES", "object": "Diabetes"}],
    )
    llm = fake_llm(response=doc)
    graph, _, mutual_index = build_graph({"d1": "abstract"}, llm)

    path = tmp_path / "graph.json"
    save_graph(graph, mutual_index, path)
    restored_graph, restored_index = load_graph(path)

    assert restored_graph.number_of_nodes() == graph.number_of_nodes()
    assert restored_graph.number_of_edges() == graph.number_of_edges()
    assert restored_index.docs_for_entity("Diabetes") == {"d1"}
