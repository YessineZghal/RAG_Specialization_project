"""Step-back prompting — ask a more general question first.

Some questions are hard to retrieve for directly because the answer lives
in a passage about the broader topic, not the specific detail asked about.
Step-back prompting asks the LLM to abstract the question up one level,
retrieves for *that*, and returns both the general and the original-query
results so a generator can draw on whichever context actually contains
the answer.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.llm import OllamaLLM

STEP_BACK_PROMPT = (
    "Given the following specific question, write a single more general "
    "'step back' question that captures the broader topic or underlying "
    "concept it depends on. Return ONLY the step-back question.\n\n"
    "Specific question: {query}\n\nStep-back question:"
)


def step_back_query(query: str, llm: OllamaLLM | None = None) -> str:
    llm = llm or OllamaLLM()
    return llm.complete(STEP_BACK_PROMPT.format(query=query)).strip().strip('"')


def step_back_search(query: str, retriever, top_k: int = 10, llm: OllamaLLM | None = None) -> dict:
    llm = llm or OllamaLLM()
    step_back_q = step_back_query(query, llm=llm)
    return {
        "step_back_question": step_back_q,
        "step_back_results": retriever.search(step_back_q, top_k=top_k),
        "original_results": retriever.search(query, top_k=top_k),
    }
