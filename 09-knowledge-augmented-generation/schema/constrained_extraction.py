"""LLM-based extraction constrained to `domain_schema`'s fixed entity and
relation types.

The LLM is prompted with the closed schema (never asked to invent one),
its raw JSON is parsed defensively (same `_extract_json_object` pattern
used throughout this repo -- see 02-advanced-rag's `self_query.py` and
07-production-rag's `ragas_eval.py`), shape-checked with Pydantic, and
then every individual entity/relation is run through `SchemaValidator` --
so a well-formed JSON object that still proposes an out-of-schema type
(e.g. "Drug" instead of "Intervention", or a relation between two types
that schema forbids) is still rejected, not silently accepted just
because the JSON parsed.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from pydantic import BaseModel, ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kag_common.llm import OllamaLLM

from .domain_schema import Entity, Relation, SchemaValidator

EXTRACTION_PROMPT = """You extract structured facts from a biomedical research abstract using ONLY this fixed schema. Do not invent new entity or relation types.

Entity types (exactly these five):
- Condition: a disease, disorder, or medical condition studied
- Intervention: a drug, procedure, treatment, or test applied
- Study: the research study or trial itself
- Outcome: a measured result or finding
- Population: the group of subjects/patients studied, with size if stated

Relation types (exactly these five, each only between the listed types):
- STUDIES: (Study, Condition)
- USES_INTERVENTION: (Study, Intervention)
- HAS_POPULATION: (Study, Population)
- REPORTS_OUTCOME: (Study, Outcome)
- INTERVENTION_AFFECTS_OUTCOME: (Intervention, Outcome)

Give the Study entity a short descriptive name (e.g. "Study-{doc_id}").
For a Population entity, put the number of subjects (an integer, if the abstract states one) in attributes.size.

Respond with ONLY a JSON object of this exact shape:
{{
  "entities": [{{"name": "...", "type": "...", "attributes": {{}}}}],
  "relations": [{{"subject": "...", "relation": "...", "object": "..."}}]
}}

Abstract (id={doc_id}):
{text}

JSON:"""


class _ExtractedEntity(BaseModel):
    name: str
    type: str
    attributes: dict = {}


class _ExtractedRelation(BaseModel):
    subject: str
    relation: str
    object: str


class _ExtractionShape(BaseModel):
    entities: list[_ExtractedEntity] = []
    relations: list[_ExtractedRelation] = []


def _extract_json_object(raw: str) -> dict:
    """Same defensive JSON extraction pattern as
    02-advanced-rag/metadata-filtering/self_query.py and
    07-production-rag/production_eval/ragas_eval.py."""
    candidate = raw.strip().strip("`")
    if candidate.lower().startswith("json"):
        candidate = candidate[4:].strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}


def extract_from_document(
    doc_id: str,
    text: str,
    llm: OllamaLLM,
    validator: SchemaValidator,
) -> tuple[list[Entity], list[Relation]]:
    """Extract schema-constrained entities and relations from one
    document. Every candidate is validated against `domain_schema`
    before being kept -- rejections are counted on `validator`, not
    dropped silently."""
    prompt = EXTRACTION_PROMPT.format(doc_id=doc_id, text=text)
    raw = llm.complete(prompt, temperature=0.0)
    payload = _extract_json_object(raw)

    if not payload:
        # no JSON object found at all -- an empty dict would otherwise
        # validate "successfully" as zero entities/relations, hiding a
        # real parse failure behind a result indistinguishable from "this
        # document genuinely has nothing to extract"
        validator.rejection_log.append(f"doc {doc_id}: unparseable extraction response")
        return [], []

    try:
        shape = _ExtractionShape.model_validate(payload)
    except ValidationError:
        # The whole response didn't even fit the expected JSON shape --
        # treat every proposed item as rejected up front (nothing to
        # individually validate), rather than crashing the whole pipeline
        # on one bad document.
        validator.rejection_log.append(f"doc {doc_id}: unparseable extraction response")
        return [], []

    entity_types: dict[str, str] = {}
    entities: list[Entity] = []
    for candidate in shape.entities:
        if not validator.validate_entity(candidate.name, candidate.type):
            continue
        key = candidate.name.strip().lower()
        entity_types[key] = candidate.type
        entities.append(Entity(name=candidate.name.strip(), type=candidate.type, attributes=candidate.attributes))

    relations: list[Relation] = []
    for candidate in shape.relations:
        subject_type = entity_types.get(candidate.subject.strip().lower())
        object_type = entity_types.get(candidate.object.strip().lower())
        if subject_type is None or object_type is None:
            # references an entity that was itself rejected (or never
            # declared) -- can't be schema-checked, so it's rejected too
            validator.rejected_relations += 1
            validator.rejection_log.append(
                f"relation rejected: {candidate.subject} -{candidate.relation}-> "
                f"{candidate.object} (unknown entity)"
            )
            continue
        if not validator.validate_relation(subject_type, candidate.relation, object_type):
            continue
        relations.append(Relation(subject=candidate.subject.strip(), relation=candidate.relation, object=candidate.object.strip()))

    return entities, relations
