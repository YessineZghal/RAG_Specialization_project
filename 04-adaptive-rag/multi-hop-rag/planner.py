"""Decompose a multi-hop question into an ordered chain of sub-questions.

A bridge question ("What nationality is the director of the film that
starred X?") can't be answered by one retrieval step because the thing you
actually need to look up (the director's name) isn't in the question —
you have to resolve an earlier step first to know what to search for next.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from adaptive_common.llm import OllamaLLM  # noqa: E402

PLANNER_PROMPT = """This question requires finding one fact, then using it to find another.
Break it into exactly 2 sequential sub-questions: the first finds the missing intermediate
entity, the second uses that entity to answer the original question.

Respond with exactly 2 lines, no numbering, no extra text.

Question: {question}
Sub-questions:"""


def plan_subquestions(question: str, llm: OllamaLLM | None = None) -> list[str]:
    llm = llm or OllamaLLM()
    response = llm.complete(PLANNER_PROMPT.format(question=question))
    subquestions = [line.strip("-* ") for line in response.splitlines() if line.strip()]
    return subquestions[:2] if len(subquestions) >= 2 else [question]
