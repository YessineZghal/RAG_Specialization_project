"""Graph-of-Thoughts' graph structure -- the same reasoning-step content
as Tree-of-Thought's nodes, but organized as a general directed graph
instead of a tree. The one structural difference that actually matters:
a node here can have **more than one parent**, which is what lets two
separate branches merge into a single synthesized node (see
`graph_search.py`'s aggregation step) -- something a tree, by
definition, cannot represent.

Pure data structure, no LLM calls -- kept separate from `graph_search.py`
so this part is trivial to unit test on its own.
"""

from __future__ import annotations

import networkx as nx


class ThoughtGraph:
    def __init__(self) -> None:
        self.graph = nx.DiGraph()
        self._next_id = 0

    def add_thought(self, text: str, parents: list[int] | None = None, score: float = 0.0) -> int:
        node_id = self._next_id
        self._next_id += 1
        self.graph.add_node(node_id, text=text, score=score)
        for parent_id in parents or []:
            self.graph.add_edge(parent_id, node_id)
        return node_id

    def text(self, node_id: int) -> str:
        return self.graph.nodes[node_id]["text"]

    def score(self, node_id: int) -> float:
        return self.graph.nodes[node_id]["score"]

    def parents(self, node_id: int) -> list[int]:
        return list(self.graph.predecessors(node_id))

    def path_to(self, node_id: int) -> list[str]:
        """The reasoning text along one path from a root down to
        `node_id`, root first. A merged node (more than one parent) just
        follows its *first* parent back -- enough to build a coherent
        prompt; the merged node's own text already carries what the
        *other* parent contributed, since `graph_search.py`'s aggregation
        step writes that synthesis directly into the merged node's text.
        """
        path = [self.text(node_id)]
        current = node_id
        while True:
            parents = self.parents(current)
            if not parents:
                break
            current = parents[0]
            path.append(self.text(current))
        return list(reversed(path))

    def __len__(self) -> int:
        return self.graph.number_of_nodes()
