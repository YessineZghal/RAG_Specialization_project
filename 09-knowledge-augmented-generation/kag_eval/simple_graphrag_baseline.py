"""A self-contained, *unconstrained* graph-rag baseline -- the same
"LLM freely invents entity/relation types, graph is a flat fact dump"
shape as Levels 3/5/6's own graph-rag, rebuilt fresh here (not imported
from those levels) so this level's comparison against it is a fair,
apples-to-apples real measurement rather than a citation of numbers from
a different dataset.

No schema, no mutual index, no logical-form parser, no operator router:
one extraction prompt with no fixed vocabulary, one flat graph, one
generic "here is retrieved text plus whatever graph facts mention words
from the question, answer yes/no/maybe" prompt. This is deliberately the
*simple* baseline the KAG paper (and this level) argues schema
constraints and hybrid reasoning improve on -- it should not be
artificially handicapped, only genuinely unconstrained.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kag_common.answer_parsing import parse_yes_no_maybe  # noqa: E402
from kag_common.embed import OllamaEmbedder, cosine_search  # noqa: E402
from kag_common.llm import OllamaLLM  # noqa: E402

logger = logging.getLogger(__name__)

UNCONSTRAINED_EXTRACTION_PROMPT = """Extract the key entities and relationships mentioned in this biomedical abstract. You may use any entity types and relationship types that seem natural -- there is no fixed schema.

Respond with ONLY a JSON object of this shape:
{{"entities": [{{"name": "...", "type": "..."}}], "relations": [{{"subject": "...", "relation": "...", "object": "..."}}]}}

Abstract:
{text}

JSON:"""

BASELINE_ANSWER_PROMPT = """Answer the following biomedical research question using the evidence below (retrieved passages and/or graph facts).

Evidence:
{evidence}

Question: {question}

Think briefly, then give your final verdict on its own last line, in exactly this form: "Answer: Yes", "Answer: No", or "Answer: Maybe"."""


def _extract_json_object(raw: str) -> dict:
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


def extract_unconstrained(doc_id: str, text: str, llm: OllamaLLM) -> tuple[list[dict], list[dict]]:
    """No schema check at all -- whatever the model proposes is kept
    as-is, the defining trait of the baseline this level compares against.

    Still requires `name`/`type` (entities) and `subject`/`relation`/`object`
    (relations) to be plain strings -- a real crash caught by actually
    running this against live output: with no schema and no Pydantic model
    forcing a shape, an unconstrained prompt occasionally answers a
    "what is the object of this relation" field with a JSON *list* (e.g.
    multiple co-occurring findings) instead of one string. The
    schema-constrained side never hits this because `_ExtractedRelation`
    is a typed Pydantic model that would reject the same list outright --
    this is real, measured evidence that the unconstrained baseline is not
    just less structured, it is also more brittle to parse safely.
    """
    raw = llm.complete(UNCONSTRAINED_EXTRACTION_PROMPT.format(text=text), temperature=0.0)
    payload = _extract_json_object(raw)
    entities = payload.get("entities") or []
    relations = payload.get("relations") or []
    entities = [
        e for e in entities
        if isinstance(e, dict) and isinstance(e.get("name"), str) and e.get("name") and isinstance(e.get("type"), str) and e.get("type")
    ]
    relations = [
        r for r in relations
        if isinstance(r, dict)
        and isinstance(r.get("subject"), str) and r.get("subject")
        and isinstance(r.get("relation"), str) and r.get("relation")
        and isinstance(r.get("object"), str) and r.get("object")
    ]
    return entities, relations


def build_unconstrained_graph(corpus: dict[str, str], llm: OllamaLLM | None = None) -> nx.MultiDiGraph:
    llm = llm or OllamaLLM()
    graph = nx.MultiDiGraph()

    for doc_id, text in corpus.items():
        entities, relations = extract_unconstrained(doc_id, text, llm)
        for entity in entities:
            key = entity["name"].strip().lower()
            if not graph.has_node(key):
                graph.add_node(key, name=entity["name"], type=entity["type"])
        for relation in relations:
            subject_key = relation["subject"].strip().lower()
            object_key = relation["object"].strip().lower()
            if not graph.has_node(subject_key):
                graph.add_node(subject_key, name=relation["subject"], type="unknown")
            if not graph.has_node(object_key):
                graph.add_node(object_key, name=relation["object"], type="unknown")
            graph.add_edge(subject_key, object_key, relation=relation["relation"], doc_id=doc_id)

    logger.info(
        "Built unconstrained baseline graph: %d nodes, %d edges from %d documents.",
        graph.number_of_nodes(),
        graph.number_of_edges(),
        len(corpus),
    )
    return graph


def _graph_facts_for_question(graph: nx.MultiDiGraph, question: str, max_facts: int = 15) -> list[str]:
    """No focus-hint parser here -- the baseline's own naive way of
    deciding which graph facts are relevant: does any word from the
    question appear in the node's name? This is intentionally cruder
    than `kg_reasoning_op.find_matching_nodes`'s schema-aware traversal."""
    words = {w.lower() for w in re.findall(r"[a-zA-Z]{4,}", question)}
    facts = []
    for source, target, data in graph.edges(data=True):
        source_name = graph.nodes[source].get("name", source)
        target_name = graph.nodes[target].get("name", target)
        haystack = f"{source_name} {target_name}".lower()
        if any(w in haystack for w in words):
            facts.append(f"{source_name} --{data.get('relation')}--> {target_name}")
        if len(facts) >= max_facts:
            break
    return facts


@dataclass(frozen=True)
class BaselineAnswer:
    verdict: str | None
    raw_response: str
    citations: frozenset[str]


def baseline_answer(
    question: str,
    corpus: dict[str, str],
    doc_ids: list[str],
    matrix: np.ndarray,
    graph: nx.MultiDiGraph,
    embedder: OllamaEmbedder | None = None,
    llm: OllamaLLM | None = None,
    top_k: int = 3,
) -> BaselineAnswer:
    embedder = embedder or OllamaEmbedder()
    llm = llm or OllamaLLM()

    query_vector = np.asarray(embedder.embed_one(question), dtype=np.float32)
    hits = cosine_search(query_vector, doc_ids, matrix, top_k=top_k)
    citations = {doc_id for doc_id, _ in hits}

    evidence_parts = [f"[retrieved doc {doc_id}] {corpus[doc_id]}" for doc_id, _ in hits if doc_id in corpus]
    graph_facts = _graph_facts_for_question(graph, question)
    if graph_facts:
        evidence_parts.append("Graph facts:\n" + "\n".join(graph_facts))

    evidence_text = "\n\n".join(evidence_parts)
    raw = llm.complete(BASELINE_ANSWER_PROMPT.format(evidence=evidence_text, question=question), temperature=0.0)
    return BaselineAnswer(verdict=parse_yes_no_maybe(raw), raw_response=raw, citations=frozenset(citations))
