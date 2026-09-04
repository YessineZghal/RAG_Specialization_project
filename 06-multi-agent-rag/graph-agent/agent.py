"""Graph Agent — looks up facts connected to a company/ticker in a small
knowledge graph, extracted from the financial corpus by a local LLM.
Same extraction approach as Levels 3 and 5's graph tools, reimplemented
here to keep this level self-contained.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from multiagent_common.agent_result import AgentResult  # noqa: E402
from multiagent_common.llm import OllamaLLM  # noqa: E402

EXTRACTION_PROMPT = """Extract factual (subject, relation, object) triples about companies from the text below.
Only extract facts that are explicitly stated. Use short, consistent entity names (e.g. company ticker or name).

Respond with ONLY a JSON array, like:
[{{"subject": "NVIDIA", "relation": "focuses on", "object": "PC graphics"}}]

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


class GraphAgent:
    name = "graph-agent"

    def __init__(self, graph: nx.MultiDiGraph, llm: OllamaLLM | None = None) -> None:
        self.graph = graph
        self.llm = llm or OllamaLLM()

    def run(self, task: str) -> AgentResult:
        task_lower = task.lower()
        matches = [n for n in self.graph.nodes if n.lower() in task_lower or task_lower.find(n.lower()) != -1]
        facts: list[str] = []
        for node in matches[:5]:
            for _, target, data in self.graph.out_edges(node, data=True):
                facts.append(f"{node} {data['relation']} {target}")
            for source, _, data in self.graph.in_edges(node, data=True):
                facts.append(f"{source} {data['relation']} {node}")
        seen, unique_facts = set(), []
        for fact in facts:
            if fact not in seen:
                seen.add(fact)
                unique_facts.append(fact)

        if not unique_facts:
            return AgentResult(self.name, task, "No relevant facts found in the knowledge graph.", success=False)

        context = "\n".join(unique_facts)
        answer = self.llm.complete(f"Known facts:\n{context}\n\nTask: {task}\nAnswer using only these facts:")
        return AgentResult(self.name, task, answer, evidence=unique_facts)
