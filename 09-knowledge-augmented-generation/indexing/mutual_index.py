"""Bidirectional entity <-> source-document mapping.

KAG's "mutual indexing" links every KG node back to the real source text
it was extracted from, so KG-reasoning answers can cite an actual PubMed
abstract instead of asserting a bare triple with no provenance -- and so
the retrieval operator can widen a KG hit back out to its full source
text when the graph alone is not enough context to answer.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field


def _entity_key(name: str) -> str:
    return name.strip().lower()


@dataclass
class MutualIndex:
    entity_to_docs: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    doc_to_entities: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    # (subject_key, relation, object_key) -> doc_ids that produced it
    relation_to_docs: dict[tuple[str, str, str], set[str]] = field(default_factory=lambda: defaultdict(set))

    def add_entity(self, name: str, doc_id: str) -> None:
        key = _entity_key(name)
        self.entity_to_docs[key].add(doc_id)
        self.doc_to_entities[doc_id].add(key)

    def add_relation(self, subject: str, relation: str, obj: str, doc_id: str) -> None:
        triple_key = (_entity_key(subject), relation, _entity_key(obj))
        self.relation_to_docs[triple_key].add(doc_id)

    def docs_for_entity(self, name: str) -> set[str]:
        return set(self.entity_to_docs.get(_entity_key(name), set()))

    def entities_for_doc(self, doc_id: str) -> set[str]:
        return set(self.doc_to_entities.get(doc_id, set()))

    def docs_for_relation(self, subject: str, relation: str, obj: str) -> set[str]:
        return set(self.relation_to_docs.get((_entity_key(subject), relation, _entity_key(obj)), set()))

    def source_text_for_entity(self, name: str, corpus: dict[str, str]) -> str:
        """Widen a KG entity back out to the real source text it came
        from -- what the retrieval operator falls back to when graph
        structure alone can't answer a question."""
        doc_ids = self.docs_for_entity(name)
        return "\n\n".join(corpus[d] for d in sorted(doc_ids) if d in corpus)

    def summary(self) -> dict:
        return {
            "n_entities_indexed": len(self.entity_to_docs),
            "n_docs_indexed": len(self.doc_to_entities),
            "n_relations_indexed": len(self.relation_to_docs),
        }

    def to_dict(self) -> dict:
        """JSON-serializable snapshot -- tuple relation keys become a
        joined string since JSON object keys must be strings."""
        return {
            "entity_to_docs": {k: sorted(v) for k, v in self.entity_to_docs.items()},
            "relation_to_docs": {
                "\x1f".join(k): sorted(v) for k, v in self.relation_to_docs.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MutualIndex":
        index = cls()
        for entity_key, doc_ids in data.get("entity_to_docs", {}).items():
            for doc_id in doc_ids:
                index.entity_to_docs[entity_key].add(doc_id)
                index.doc_to_entities[doc_id].add(entity_key)
        for joined_key, doc_ids in data.get("relation_to_docs", {}).items():
            subject_key, relation, object_key = joined_key.split("\x1f")
            index.relation_to_docs[(subject_key, relation, object_key)] |= set(doc_ids)
        return index
