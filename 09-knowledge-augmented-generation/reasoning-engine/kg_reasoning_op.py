"""The `kg_reasoning` operator: follow structured relations in the
schema-constrained graph rather than re-reading free text.

Node lookup is a simple substring match against a `focus_hint` phrase --
deliberately not another LLM call. A real production KAG-style system
would use entity linking; here the point is showing what graph structure
buys once an entity is found, not re-solving entity linking (out of
scope, and the level's Common Failure Modes section says so explicitly).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from indexing.mutual_index import MutualIndex  # noqa: E402
from schema.domain_schema import RELATION_SCHEMA  # noqa: E402


@dataclass(frozen=True)
class KgEvidence:
    facts: list[str]
    doc_ids: set[str] = field(default_factory=set)


def find_matching_nodes(graph: nx.MultiDiGraph, focus_hint: str | None) -> list[str]:
    if not focus_hint:
        return []
    hint = focus_hint.strip().lower()
    if not hint:
        return []
    return [n for n, data in graph.nodes(data=True) if hint in n or hint in data.get("name", "").lower()]


def reason_over_graph(
    graph: nx.MultiDiGraph,
    mutual_index: MutualIndex,
    focus_hint: str | None,
) -> KgEvidence:
    """Traverse every schema relation out of (and into) nodes matching
    `focus_hint`, turning each edge into one plain-English fact string
    plus the real document(s) it was extracted from."""
    matches = find_matching_nodes(graph, focus_hint)
    facts: list[str] = []
    doc_ids: set[str] = set()

    for node in matches:
        node_name = graph.nodes[node].get("name", node)
        for _, target, data in graph.out_edges(node, data=True):
            relation = data.get("relation")
            if relation not in RELATION_SCHEMA:
                continue
            target_name = graph.nodes[target].get("name", target)
            facts.append(f"{node_name} --{relation}--> {target_name}")
            doc_ids |= mutual_index.docs_for_relation(node_name, relation, target_name)
        for source, _, data in graph.in_edges(node, data=True):
            relation = data.get("relation")
            if relation not in RELATION_SCHEMA:
                continue
            source_name = graph.nodes[source].get("name", source)
            facts.append(f"{source_name} --{relation}--> {node_name}")
            doc_ids |= mutual_index.docs_for_relation(source_name, relation, node_name)

    return KgEvidence(facts=facts, doc_ids=doc_ids)
