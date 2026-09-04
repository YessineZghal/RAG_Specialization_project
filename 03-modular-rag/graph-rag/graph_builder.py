"""Build a queryable knowledge graph from extracted triples, using
`networkx` — a directed multigraph, since two entities can be connected
by more than one relation (e.g. "Google Brain" --includes--> "X" and
"X" --published--> "paper").
"""

from __future__ import annotations

import networkx as nx


def build_graph(triples: list[dict]) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    for triple in triples:
        graph.add_edge(triple["subject"], triple["object"], relation=triple["relation"])
    return graph


def build_graph_from_chunks(
    chunks: dict[str, dict], extractor=None, limit: int | None = None
) -> nx.MultiDiGraph:
    """Run entity extraction over a set of text chunks and merge everything
    into one graph. `extractor` defaults to `entity_extraction.extract_triples`.
    """
    if extractor is None:
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from entity_extraction import extract_triples

        extractor = extract_triples

    graph = nx.MultiDiGraph()
    items = list(chunks.items())[:limit] if limit else list(chunks.items())
    for chunk_id, chunk in items:
        for triple in extractor(chunk["text"]):
            graph.add_edge(
                triple["subject"], triple["object"], relation=triple["relation"], source=chunk_id
            )
    return graph
