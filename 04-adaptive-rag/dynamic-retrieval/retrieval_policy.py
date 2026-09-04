"""The adaptive decision policy: route a question to a retrieval strategy
based on its classified complexity, instead of always running the same
fixed pipeline (see `../README.md` for the decision table this implements).
"""

from __future__ import annotations

import sys
from pathlib import Path

_LEVEL_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_LEVEL_DIR))
sys.path.insert(0, str(_LEVEL_DIR / "query-classification"))
sys.path.insert(0, str(_LEVEL_DIR / "multi-hop-rag"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # this file's own dir, for dynamic_top_k
from adaptive_common.llm import OllamaLLM
from classifier import classify_rule
from dynamic_top_k import dynamic_top_k


def multi_query_retrieve(question: str, retriever, llm: OllamaLLM, top_k: int) -> list[tuple[str, float]]:
    """"complex" strategy: paraphrase the question a couple of ways, fuse
    the rankings with RRF, so a comparison question isn't limited to
    whichever single phrasing the retriever handles best.
    """
    prompt = (
        "Generate 2 different phrasings of this question, one per line, no numbering:\n\n"
        f"{question}"
    )
    variants = [question] + [
        line.strip("-* ") for line in llm.complete(prompt).splitlines() if line.strip()
    ][:2]

    rankings = [retriever.search(v, top_k=top_k) for v in variants]
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, (doc_id, _score) in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (60 + rank)
    ranked = sorted(fused.items(), key=lambda item: item[1], reverse=True)
    return ranked[:top_k]


def run_policy(question: str, retriever, llm: OllamaLLM | None = None) -> dict:
    """Classify, then execute the matching strategy. Returns a dict with
    everything relevant for transparency: the classification, the
    resulting Top-K, the strategy name, and the retrieved doc ids.
    """
    llm = llm or OllamaLLM()
    complexity = classify_rule(question)
    top_k = dynamic_top_k(complexity)

    if complexity == "none":
        return {"complexity": complexity, "strategy": "no_retrieval", "top_k": 0, "results": []}

    if complexity == "simple":
        results = retriever.search(question, top_k=top_k)
        return {"complexity": complexity, "strategy": "single_retrieval", "top_k": top_k, "results": results}

    if complexity == "complex":
        results = multi_query_retrieve(question, retriever, llm, top_k)
        return {"complexity": complexity, "strategy": "multi_query_fusion", "top_k": top_k, "results": results}

    # multi_hop
    from subquestion_retrieval import multi_hop_retrieve

    results = multi_hop_retrieve(question, retriever, llm, top_k_per_hop=top_k)
    return {"complexity": complexity, "strategy": "multi_hop_retrieval", "top_k": top_k, "results": results}
