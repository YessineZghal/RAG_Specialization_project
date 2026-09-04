"""Sketch a rough plan before the agent starts calling tools — not a rigid
script (the ReAct loop in `agents/rag_agent.py` still decides tool calls
dynamically), but a short, logged statement of intent that makes the
agent's later choices explainable rather than opaque.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agentic_common.llm import OllamaLLM

PLAN_PROMPT = """You have these tools available: vector_search, get_document, sql_query, web_search, graph_search.

Write a short 2-3 step plan (one short line per step) for how you would answer this question.
Do not answer the question yet -- just the plan.

Question: {question}
Plan:"""


def make_plan(question: str, llm: OllamaLLM | None = None) -> list[str]:
    llm = llm or OllamaLLM()
    response = llm.complete(PLAN_PROMPT.format(question=question))
    steps = [
        line.strip("-*0123456789. ")
        for line in response.splitlines()
        if line.strip() and not line.strip().rstrip(":").lower().startswith(("here", "plan"))
    ]
    return steps or [f"Search for information about: {question}"]
