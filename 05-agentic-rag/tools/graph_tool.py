"""`graph_search(entity)` — look up facts connected to an entity in a
small knowledge graph, extracted from the same TriviaQA corpus by a local
LLM. Same extraction approach as Level 3's `graph-rag/`, reimplemented
here to keep this level self-contained.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agentic_common.llm import OllamaLLM  # noqa: E402

EXTRACTION_PROMPT = """Extract factual (subject, relation, object) triples from the text below.
Only extract facts that are explicitly stated. Use short, consistent entity names.

Respond with ONLY a JSON array, like:
[{{"subject": "A", "relation": "works at", "object": "B"}}]

Text:
{text}

JSON:"""


def extract_triples(text: str, llm: OllamaLLM | None = None) -> list[dict]:
    llm = llm or OllamaLLM()
    raw = llm.complete(EXTRACTION_PROMPT.format(text=text), temperature=0.0)
    candidate = raw.strip().strip("`")
    if candidate.lower().startswith("json"):
        candidate = candidate[4:].strip()
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not match:
            return []
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []
    if not isinstance(parsed, list):
        return []
    return [
        {"subject": str(t["subject"]).strip(), "relation": str(t["relation"]).strip(), "object": str(t["object"]).strip()}
        for t in parsed
        if isinstance(t, dict) and {"subject", "relation", "object"} <= t.keys()
    ]


def build_graph(triples: list[dict]) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    for t in triples:
        graph.add_edge(t["subject"], t["object"], relation=t["relation"])
    return graph


class GraphTool:
    name = "graph_search"
    description = "Look up facts connected to a named entity in the knowledge graph."

    def __init__(self, graph: nx.MultiDiGraph) -> None:
        self.graph = graph

    def __call__(self, entity: str) -> list[str]:
        query_lower = entity.lower()
        matches = [n for n in self.graph.nodes if n.lower() in query_lower or query_lower in n.lower()]
        facts = []
        for node in matches[:5]:
            for _, target, data in self.graph.out_edges(node, data=True):
                facts.append(f"{node} {data['relation']} {target}")
            for source, _, data in self.graph.in_edges(node, data=True):
                facts.append(f"{source} {data['relation']} {node}")
        seen, unique = set(), []
        for fact in facts:
            if fact not in seen:
                seen.add(fact)
                unique.append(fact)
        return unique
