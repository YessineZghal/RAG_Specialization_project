"""The `numerical_calculation` operator: genuinely deterministic Python
over numeric attributes already sitting in the graph (e.g. a
Population's `size`) -- no LLM call at all. This is the operator whose
whole point is that a language model should never be asked to do
arithmetic or threshold comparison when the graph already has the
number: KAG's own motivation for a dedicated numerical operator instead
of leaving every comparison to free-text reasoning.

Supports two shapes of `numeric_comparison`:
  - a threshold check: {"attribute": "size", "op": ">", "value": 500}
  - an aggregation: {"attribute": "size", "op": "max"} / "min" (value ignored)
       -> used for "the largest study" style questions, returning which
          node holds the extreme value so a follow-up KG lookup can use it
"""

from __future__ import annotations

import operator as op_module
import sys
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kg_reasoning_op import find_matching_nodes  # noqa: E402

_COMPARATORS = {
    ">": op_module.gt,
    "<": op_module.lt,
    ">=": op_module.ge,
    "<=": op_module.le,
    "==": op_module.eq,
}


@dataclass(frozen=True)
class NumericResult:
    values: dict[str, float] = field(default_factory=dict)
    comparison_result: bool | None = None
    extreme_node: str | None = None
    explanation: str = ""


def _population_nodes_for(graph: nx.MultiDiGraph, matched: list[str]) -> set[str]:
    population_nodes: set[str] = set()
    studies: set[str] = set()

    for node in matched:
        node_type = graph.nodes[node].get("type")
        if node_type == "Population":
            population_nodes.add(node)
        elif node_type == "Study":
            studies.add(node)
        elif node_type == "Condition":
            for source, _, data in graph.in_edges(node, data=True):
                if data.get("relation") == "STUDIES":
                    studies.add(source)

    for study in studies:
        for _, target, data in graph.out_edges(study, data=True):
            if data.get("relation") == "HAS_POPULATION":
                population_nodes.add(target)

    return population_nodes


def evaluate_numeric(
    graph: nx.MultiDiGraph,
    focus_hint: str | None,
    comparison: dict,
) -> NumericResult:
    attribute = comparison["attribute"]
    comparison_op = comparison["op"]

    matched = find_matching_nodes(graph, focus_hint) if focus_hint else []
    population_nodes = _population_nodes_for(graph, matched)
    if not population_nodes:
        # no focus, or focus didn't resolve to any Population -- fall
        # back to every Population node in the graph rather than
        # returning an empty (and misleadingly confident) result
        population_nodes = {n for n, d in graph.nodes(data=True) if d.get("type") == "Population"}

    values = {
        node: graph.nodes[node]["attributes"][attribute]
        for node in population_nodes
        if attribute in graph.nodes[node].get("attributes", {})
    }

    if not values:
        return NumericResult(explanation=f"No '{attribute}' attribute found on any matched Population node.")

    if comparison_op in ("max", "min"):
        pick = max if comparison_op == "max" else min
        extreme_node = pick(values, key=values.get)
        return NumericResult(
            values=values,
            extreme_node=extreme_node,
            explanation=f"{comparison_op}({attribute}) = {values[extreme_node]} at {extreme_node!r}.",
        )

    comparator = _COMPARATORS.get(comparison_op)
    if comparator is None:
        return NumericResult(values=values, explanation=f"Unsupported comparison op: {comparison_op!r}")

    threshold = comparison["value"]
    result = any(comparator(v, threshold) for v in values.values())
    return NumericResult(
        values=values,
        comparison_result=result,
        explanation=f"{attribute} {comparison_op} {threshold} -> {result} (values seen: {values}).",
    )
