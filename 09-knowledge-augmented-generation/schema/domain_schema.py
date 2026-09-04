"""The fixed biomedical schema every extraction in this level is
constrained against -- KAG's core departure from Levels 3/5/6's
unconstrained graph-rag, where the LLM is free to invent any entity or
relation type it likes.

Five entity types, five relation types, each relation with a fixed
allowed (subject_type, object_type) pair. Anything the extractor proposes
outside this closed set is *rejected*, not coerced -- `SchemaValidator`
below counts every rejection so the real cost of the constraint (recall
lost to things that do not fit the schema) is a measured number in the
evaluation, not an assumption.
"""

from __future__ import annotations

from dataclasses import dataclass, field

ENTITY_TYPES: frozenset[str] = frozenset(
    {"Condition", "Intervention", "Study", "Outcome", "Population"}
)

# relation_type -> (subject_type, object_type)
RELATION_SCHEMA: dict[str, tuple[str, str]] = {
    "STUDIES": ("Study", "Condition"),
    "USES_INTERVENTION": ("Study", "Intervention"),
    "HAS_POPULATION": ("Study", "Population"),
    "REPORTS_OUTCOME": ("Study", "Outcome"),
    "INTERVENTION_AFFECTS_OUTCOME": ("Intervention", "Outcome"),
}

RELATION_TYPES: frozenset[str] = frozenset(RELATION_SCHEMA.keys())


@dataclass(frozen=True)
class Entity:
    name: str
    type: str
    # Free-form numeric/text attributes, e.g. Population -> {"size": 500},
    # Outcome -> {"direction": "improved"}. Not schema-constrained itself
    # (attribute keys are extraction-dependent), only the entity `type` is.
    attributes: dict = field(default_factory=dict)

    def key(self) -> tuple[str, str]:
        return (self.name.strip().lower(), self.type)


@dataclass(frozen=True)
class Relation:
    subject: str  # entity name
    relation: str
    object: str  # entity name


class SchemaViolation(Exception):
    """Raised (and caught) whenever a proposed entity/relation does not
    fit the closed schema -- never silently coerced."""


class SchemaValidator:
    """Validates proposed entities/relations against the fixed schema and
    keeps a running, inspectable count of accepted vs. rejected items --
    the real, measurable cost of schema constraint this level's whole
    comparison against unconstrained graph-rag depends on."""

    def __init__(self) -> None:
        self.accepted_entities = 0
        self.rejected_entities = 0
        self.accepted_relations = 0
        self.rejected_relations = 0
        self.rejection_log: list[str] = []

    def validate_entity(self, name: str, entity_type: str) -> bool:
        if not name or not name.strip() or entity_type not in ENTITY_TYPES:
            self.rejected_entities += 1
            self.rejection_log.append(f"entity rejected: name={name!r} type={entity_type!r}")
            return False
        self.accepted_entities += 1
        return True

    def validate_relation(
        self, subject_type: str, relation_type: str, object_type: str
    ) -> bool:
        expected = RELATION_SCHEMA.get(relation_type)
        if expected is None or expected != (subject_type, object_type):
            self.rejected_relations += 1
            self.rejection_log.append(
                f"relation rejected: {subject_type} -{relation_type}-> {object_type} "
                f"(expected {expected})"
            )
            return False
        self.accepted_relations += 1
        return True

    def summary(self) -> dict:
        total_entities = self.accepted_entities + self.rejected_entities
        total_relations = self.accepted_relations + self.rejected_relations
        return {
            "accepted_entities": self.accepted_entities,
            "rejected_entities": self.rejected_entities,
            "entity_rejection_rate": (
                self.rejected_entities / total_entities if total_entities else 0.0
            ),
            "accepted_relations": self.accepted_relations,
            "rejected_relations": self.rejected_relations,
            "relation_rejection_rate": (
                self.rejected_relations / total_relations if total_relations else 0.0
            ),
        }
