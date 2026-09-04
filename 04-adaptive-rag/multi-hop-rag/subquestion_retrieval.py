"""Retrieve for each sub-question in sequence, carrying the previous hop's
top evidence forward as context for the next hop's query — this is what
actually lets the second retrieval step "know" the bridge entity resolved
by the first one, instead of retrieving each sub-question independently
and hoping they happen to overlap.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from adaptive_common.llm import OllamaLLM

sys.path.insert(0, str(Path(__file__).resolve().parent))
from planner import plan_subquestions


def multi_hop_retrieve(
    question: str,
    retriever,
    llm: OllamaLLM | None = None,
    top_k_per_hop: int = 5,
    corpus: dict[str, str] | None = None,
) -> list[tuple[str, float]]:
    """Returns a deduplicated `[(doc_id, score), ...]` list pooling evidence
    from every hop, best score per doc kept, sorted descending.
    """
    llm = llm or OllamaLLM()
    subquestions = plan_subquestions(question, llm=llm)

    pooled: dict[str, float] = {}
    bridge_context = ""
    for subq in subquestions:
        query = f"{bridge_context} {subq}".strip() if bridge_context else subq
        hop_results = retriever.search(query, top_k=top_k_per_hop)
        for doc_id, score in hop_results:
            pooled[doc_id] = max(pooled.get(doc_id, 0.0), score)

        if corpus and hop_results:
            top_doc_id = hop_results[0][0]
            bridge_context = corpus.get(top_doc_id, "")[:300]

    return sorted(pooled.items(), key=lambda item: item[1], reverse=True)
