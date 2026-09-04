"""Build a schema-constrained knowledge graph (networkx) from a corpus of
real PubMed abstracts, populating a `MutualIndex` at the same time.

`networkx` stands in for KAG's own OpenSPG engine -- this level's README
is explicit that a hand-rolled graph on a plain, inspectable library is
the point here (understand the mechanism), not adopting the paper's
production graph store.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kag_common.llm import OllamaLLM  # noqa: E402
from schema.constrained_extraction import extract_from_document  # noqa: E402
from schema.domain_schema import SchemaValidator  # noqa: E402

from .mutual_index import MutualIndex

logger = logging.getLogger(__name__)


def build_graph(
    corpus: dict[str, str],
    llm: OllamaLLM | None = None,
    validator: SchemaValidator | None = None,
    mutual_index: MutualIndex | None = None,
) -> tuple[nx.MultiDiGraph, SchemaValidator, MutualIndex]:
    """Run schema-constrained extraction over every document in `corpus`
    and assemble the results into one shared graph + mutual index.

    Returns `(graph, validator, mutual_index)` so callers can inspect the
    real rejection counts (`validator.summary()`) alongside the graph
    itself -- the schema's real recall cost is a first-class output here,
    not an afterthought.
    """
    llm = llm or OllamaLLM()
    validator = validator or SchemaValidator()
    mutual_index = mutual_index or MutualIndex()

    graph = nx.MultiDiGraph()

    for doc_id, text in corpus.items():
        entities, relations = extract_from_document(doc_id, text, llm, validator)

        for entity in entities:
            node_key = entity.name.strip().lower()
            if graph.has_node(node_key):
                # merge attributes from a later mention rather than
                # overwrite -- e.g. a Population's size stated in one
                # document should survive even if a later document
                # mentions the same-named entity without it
                merged = {**graph.nodes[node_key].get("attributes", {}), **entity.attributes}
                graph.nodes[node_key]["attributes"] = merged
            else:
                graph.add_node(node_key, name=entity.name, type=entity.type, attributes=dict(entity.attributes))
            mutual_index.add_entity(entity.name, doc_id)

        for relation in relations:
            subject_key = relation.subject.strip().lower()
            object_key = relation.object.strip().lower()
            if not graph.has_node(subject_key) or not graph.has_node(object_key):
                continue  # defensive: extraction should not reference undeclared entities, but don't trust blindly
            graph.add_edge(subject_key, object_key, relation=relation.relation, doc_id=doc_id)
            mutual_index.add_relation(relation.subject, relation.relation, relation.object, doc_id)

    logger.info(
        "Built KG: %d nodes, %d edges from %d documents. Validator: %s",
        graph.number_of_nodes(),
        graph.number_of_edges(),
        len(corpus),
        validator.summary(),
    )
    return graph, validator, mutual_index


def save_graph(graph: nx.MultiDiGraph, mutual_index: MutualIndex, path: Path) -> None:
    """Persist the graph + mutual index together as one JSON file --
    node-link format (plain, human-readable) rather than a pickle, so a
    cached graph can be inspected directly."""
    payload = {
        "graph": nx.node_link_data(graph, edges="edges"),
        "mutual_index": mutual_index.to_dict(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def load_graph(path: Path) -> tuple[nx.MultiDiGraph, MutualIndex]:
    payload = json.loads(path.read_text())
    graph = nx.node_link_graph(payload["graph"], edges="edges", multigraph=True, directed=True)
    mutual_index = MutualIndex.from_dict(payload["mutual_index"])
    return graph, mutual_index


def nodes_by_type(graph: nx.MultiDiGraph, entity_type: str) -> list[str]:
    """Return node keys whose `type` attribute matches `entity_type`."""
    return [n for n, data in graph.nodes(data=True) if data.get("type") == entity_type]


def neighbors_by_relation(graph: nx.MultiDiGraph, node_key: str, relation: str) -> list[str]:
    """Return target node keys reachable from `node_key` via edges typed `relation`."""
    if not graph.has_node(node_key):
        return []
    targets = []
    for _, target, data in graph.out_edges(node_key, data=True):
        if data.get("relation") == relation:
            targets.append(target)
    return targets
