"""Query the knowledge graph: find nodes matching an entity mentioned in
the question, then return the facts (edges) connected to them.

Entity matching here is deliberately simple (case-insensitive substring),
consistent with this level's "naive-first" philosophy — Level 5+ agentic
patterns can layer smarter entity linking on top of the same graph.
"""

from __future__ import annotations

import networkx as nx


def find_matching_nodes(graph: nx.MultiDiGraph, query: str) -> list[str]:
    query_lower = query.lower()
    return [
        node
        for node in graph.nodes
        if node.lower() in query_lower or query_lower.find(node.lower().split()[0]) != -1
    ]


def describe_node(graph: nx.MultiDiGraph, node: str) -> list[str]:
    """All facts where `node` is the subject or the object, as readable strings."""
    facts = []
    for _, target, data in graph.out_edges(node, data=True):
        facts.append(f"{node} {data['relation']} {target}")
    for source, _, data in graph.in_edges(node, data=True):
        facts.append(f"{source} {data['relation']} {node}")
    return facts


def graph_search(graph: nx.MultiDiGraph, query: str, max_nodes: int = 5) -> list[str]:
    """Return readable facts relevant to `query` — the "retrieved context"
    for a graph-backed answer.
    """
    matches = find_matching_nodes(graph, query)[:max_nodes]
    facts: list[str] = []
    for node in matches:
        facts.extend(describe_node(graph, node))
    # De-duplicate while preserving order (a fact can be reachable from both endpoints).
    seen = set()
    unique_facts = []
    for fact in facts:
        if fact not in seen:
            seen.add(fact)
            unique_facts.append(fact)
    return unique_facts
