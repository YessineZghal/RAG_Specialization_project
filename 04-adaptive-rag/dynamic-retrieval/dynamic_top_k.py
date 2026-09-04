"""Map a classified complexity tier to a Top-K — instead of using the
same fixed Top-K for every question regardless of how much evidence it
actually needs (Level 1 and 2's approach).
"""

from __future__ import annotations

TOP_K_BY_COMPLEXITY = {
    "none": 0,
    "simple": 3,
    "complex": 8,
    "multi_hop": 5,  # per hop, not total -- see multi-hop-rag/
}


def dynamic_top_k(complexity: str) -> int:
    return TOP_K_BY_COMPLEXITY.get(complexity, TOP_K_BY_COMPLEXITY["simple"])
