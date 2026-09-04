from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from schema.constrained_extraction import extract_from_document
from schema.domain_schema import SchemaValidator


def test_extract_from_document_keeps_well_formed_schema_conforming_output(fake_llm):
    payload = {
        "entities": [
            {"name": "Study-1", "type": "Study", "attributes": {}},
            {"name": "Diabetes", "type": "Condition", "attributes": {}},
        ],
        "relations": [{"subject": "Study-1", "relation": "STUDIES", "object": "Diabetes"}],
    }
    llm = fake_llm(response=json.dumps(payload))
    validator = SchemaValidator()

    entities, relations = extract_from_document("123", "some abstract text", llm, validator)

    assert {e.name for e in entities} == {"Study-1", "Diabetes"}
    assert len(relations) == 1
    assert relations[0].relation == "STUDIES"
    assert validator.accepted_entities == 2
    assert validator.accepted_relations == 1


def test_extract_from_document_rejects_an_out_of_schema_entity_type(fake_llm):
    payload = {
        "entities": [
            {"name": "Metformin", "type": "Drug", "attributes": {}},  # "Drug" is not in the schema
            {"name": "Diabetes", "type": "Condition", "attributes": {}},
        ],
        "relations": [],
    }
    llm = fake_llm(response=json.dumps(payload))
    validator = SchemaValidator()

    entities, _ = extract_from_document("123", "text", llm, validator)

    assert {e.name for e in entities} == {"Diabetes"}
    assert validator.rejected_entities == 1


def test_extract_from_document_rejects_a_relation_referencing_an_unknown_entity(fake_llm):
    payload = {
        "entities": [{"name": "Diabetes", "type": "Condition", "attributes": {}}],
        # "Study-1" was never declared as an entity
        "relations": [{"subject": "Study-1", "relation": "STUDIES", "object": "Diabetes"}],
    }
    llm = fake_llm(response=json.dumps(payload))
    validator = SchemaValidator()

    _, relations = extract_from_document("123", "text", llm, validator)

    assert relations == []
    assert validator.rejected_relations == 1


def test_extract_from_document_on_unparseable_response_returns_nothing_and_logs(fake_llm):
    llm = fake_llm(response="I could not extract anything, sorry.")
    validator = SchemaValidator()

    entities, relations = extract_from_document("123", "text", llm, validator)

    assert entities == []
    assert relations == []
    assert validator.rejection_log  # the failure was recorded, not silently dropped


def test_extract_from_document_tolerates_a_markdown_fenced_json_block(fake_llm):
    payload = {"entities": [{"name": "Aspirin", "type": "Intervention", "attributes": {}}], "relations": []}
    llm = fake_llm(response=f"```json\n{json.dumps(payload)}\n```")
    validator = SchemaValidator()

    entities, _ = extract_from_document("123", "text", llm, validator)

    assert len(entities) == 1
    assert entities[0].name == "Aspirin"


def test_extract_from_document_keeps_population_size_attribute(fake_llm):
    payload = {
        "entities": [{"name": "Trial cohort", "type": "Population", "attributes": {"size": 620}}],
        "relations": [],
    }
    llm = fake_llm(response=json.dumps(payload))
    validator = SchemaValidator()

    entities, _ = extract_from_document("123", "text", llm, validator)

    assert entities[0].attributes["size"] == 620
