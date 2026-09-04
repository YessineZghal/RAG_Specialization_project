"""Query rewriting — ask the LLM to turn a short/ambiguous user query into a
clearer search query before embedding or tokenizing it.

Useful for conversational queries ("what about the enterprise one?") that
carry implicit context a raw retriever has no way to resolve on its own.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.llm import OllamaLLM

REWRITE_PROMPT = (
    "Rewrite the following search query to be clearer and more specific for a "
    "document search engine. Keep it factual — do not add information that "
    "isn't implied by the original query. Return ONLY the rewritten query, "
    "nothing else.\n\nOriginal query: {query}\n\nRewritten query:"
)


def rewrite_query(query: str, llm: OllamaLLM | None = None) -> str:
    llm = llm or OllamaLLM()
    return llm.complete(REWRITE_PROMPT.format(query=query)).strip().strip('"')
