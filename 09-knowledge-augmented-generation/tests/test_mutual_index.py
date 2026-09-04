from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from indexing.mutual_index import MutualIndex  # noqa: E402


def test_add_entity_is_queryable_from_both_directions():
    index = MutualIndex()
    index.add_entity("Diabetes", "doc-1")
    index.add_entity("Diabetes", "doc-2")

    assert index.docs_for_entity("Diabetes") == {"doc-1", "doc-2"}
    assert index.docs_for_entity("diabetes") == {"doc-1", "doc-2"}  # case-insensitive key
    assert index.entities_for_doc("doc-1") == {"diabetes"}


def test_add_relation_is_queryable_by_exact_triple():
    index = MutualIndex()
    index.add_relation("Study-1", "STUDIES", "Diabetes", "doc-1")

    assert index.docs_for_relation("Study-1", "STUDIES", "Diabetes") == {"doc-1"}
    assert index.docs_for_relation("Study-1", "STUDIES", "Obesity") == set()


def test_source_text_for_entity_widens_back_to_real_documents():
    index = MutualIndex()
    index.add_entity("Diabetes", "doc-1")
    index.add_entity("Diabetes", "doc-2")
    corpus = {"doc-1": "First abstract.", "doc-2": "Second abstract."}

    text = index.source_text_for_entity("Diabetes", corpus)

    assert "First abstract." in text
    assert "Second abstract." in text


def test_summary_counts_are_correct():
    index = MutualIndex()
    index.add_entity("Diabetes", "doc-1")
    index.add_entity("Metformin", "doc-1")
    index.add_relation("Study-1", "STUDIES", "Diabetes", "doc-1")

    summary = index.summary()

    assert summary["n_entities_indexed"] == 2
    assert summary["n_docs_indexed"] == 1
    assert summary["n_relations_indexed"] == 1


def test_to_dict_and_from_dict_round_trip():
    index = MutualIndex()
    index.add_entity("Diabetes", "doc-1")
    index.add_relation("Study-1", "STUDIES", "Diabetes", "doc-1")

    restored = MutualIndex.from_dict(index.to_dict())

    assert restored.docs_for_entity("Diabetes") == {"doc-1"}
    assert restored.docs_for_relation("Study-1", "STUDIES", "Diabetes") == {"doc-1"}
