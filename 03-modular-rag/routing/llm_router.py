"""LLM-based query router — handles the ambiguous cases `rule_router.py`
can't (or shouldn't try to). Slower and costs a model call, but reasons
about intent instead of matching keywords.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modular_common.llm import OllamaLLM  # noqa: E402
from routing.rule_router import ROUTES  # noqa: E402

ROUTER_PROMPT = """You are a query router for a system with five backends:

- documents: questions about the content of a specific research paper (concepts, methods, results, definitions)
- sql: questions answerable by querying a music store database (artists, albums, tracks, customers, invoices, sales counts/totals)
- graph: questions about relationships between entities (who is affiliated with what, who authored/created what, what is connected to what)
- web: questions needing current/recent/fresh information not in a static paper (news, "latest", "today", recent years)
- api: questions about looking up a paper's metadata (arXiv id, DOI, citation info, publication details)

Classify the following question into EXACTLY ONE of: documents, sql, graph, web, api.
Respond with only the single word.

Question: {question}
Backend:"""


def llm_route(question: str, llm: OllamaLLM | None = None) -> str:
    llm = llm or OllamaLLM()
    response = llm.complete(ROUTER_PROMPT.format(question=question)).strip().lower()
    for route in ROUTES:
        if route in response:
            return route
    return "documents"  # safe default if the model answers off-script
