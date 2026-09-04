"""Multi-query retrieval — ask the LLM for several rephrasings of the same
question, retrieve for each, and fuse the rankings with RRF (see
`../hybrid-search/rrf.py`). Covers cases where a single phrasing of the
query just doesn't overlap well with how the answer happens to be worded
in the corpus.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))  # 02-advanced-rag/  (for common.llm)
sys.path.insert(0, str(_HERE.parent / "hybrid-search"))  # for top-level `rrf`
from common.llm import OllamaLLM
from rrf import reciprocal_rank_fusion

MULTI_QUERY_PROMPT = (
    "Generate {n} different search queries that would each help answer the "
    "following question, phrased differently from each other. Return ONLY "
    "the queries, one per line, with no numbering or extra text.\n\n"
    "Question: {query}"
)


def generate_queries(query: str, n: int = 3, llm: OllamaLLM | None = None) -> list[str]:
    llm = llm or OllamaLLM()
    response = llm.complete(MULTI_QUERY_PROMPT.format(n=n, query=query))
    queries = [line.strip("-*\"' ").strip() for line in response.splitlines() if line.strip()]
    return queries[:n] or [query]


def multi_query_search(
    query: str,
    retriever,
    n: int = 3,
    top_k: int = 10,
    candidate_k: int = 20,
    llm: OllamaLLM | None = None,
) -> list[tuple[str, float]]:
    variants = [query, *generate_queries(query, n=n, llm=llm)]
    rankings = [retriever.search(variant, top_k=candidate_k) for variant in variants]
    fused = reciprocal_rank_fusion(rankings)
    return fused[:top_k]
