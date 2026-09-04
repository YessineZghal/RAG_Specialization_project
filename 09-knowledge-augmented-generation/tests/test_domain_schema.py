from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from schema.domain_schema import SchemaValidator


def test_validate_entity_accepts_a_type_in_the_closed_schema():
    validator = SchemaValidator()
    assert validator.validate_entity("Metformin", "Intervention") is True
    assert validator.accepted_entities == 1
    assert validator.rejected_entities == 0


def test_validate_entity_rejects_a_type_outside_the_closed_schema():
    validator = SchemaValidator()
    assert validator.validate_entity("Metformin", "Drug") is False  # "Drug" is not one of the five types
    assert validator.rejected_entities == 1
    assert validator.rejection_log  # a human-readable reason was recorded


def test_validate_entity_rejects_an_empty_name():
    validator = SchemaValidator()
    assert validator.validate_entity("   ", "Condition") is False
    assert validator.rejected_entities == 1


def test_validate_relation_accepts_the_declared_subject_object_pair():
    validator = SchemaValidator()
    assert validator.validate_relation("Study", "STUDIES", "Condition") is True
    assert validator.accepted_relations == 1


def test_validate_relation_rejects_a_type_pair_the_schema_forbids():
    validator = SchemaValidator()
    # STUDIES is only (Study, Condition), never (Condition, Study)
    assert validator.validate_relation("Condition", "STUDIES", "Study") is False
    assert validator.rejected_relations == 1


def test_validate_relation_rejects_an_unknown_relation_type():
    validator = SchemaValidator()
    assert validator.validate_relation("Study", "CAUSES", "Condition") is False
    assert validator.rejected_relations == 1


def test_summary_computes_rejection_rates():
    validator = SchemaValidator()
    validator.validate_entity("A", "Condition")
    validator.validate_entity("B", "NotAType")
    summary = validator.summary()
    assert summary["accepted_entities"] == 1
    assert summary["rejected_entities"] == 1
    assert summary["entity_rejection_rate"] == 0.5


def test_summary_on_no_activity_does_not_divide_by_zero():
    validator = SchemaValidator()
    summary = validator.summary()
    assert summary["entity_rejection_rate"] == 0.0
    assert summary["relation_rejection_rate"] == 0.0
