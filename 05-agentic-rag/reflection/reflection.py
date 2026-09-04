"""Reflection — after producing an answer, have the model step back and
judge whether its own approach actually worked, distinct from Self-RAG's
narrower "is this answer grounded in the context" check (Level 4). This
also asks: did I use the right tool? Should I try a different one?
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agentic_common.llm import OllamaLLM  # noqa: E402

REFLECTION_PROMPT = """Question: {question}
Tools used: {tools_used}
Evidence gathered: {evidence}
Answer given: {answer}

Reflect: does the evidence actually support this answer, and were the right tools used?
Respond with only one word: satisfactory or unsatisfactory.
Judgment:"""


def reflect(question: str, tools_used: list[str], evidence: str, answer: str, llm: OllamaLLM | None = None) -> dict:
    llm = llm or OllamaLLM()
    response = llm.complete(
        REFLECTION_PROMPT.format(
            question=question, tools_used=", ".join(tools_used) or "none", evidence=evidence[:2000], answer=answer
        )
    ).strip().lower()

    satisfactory = bool(re.search(r"\bsatisfactory\b", response)) and not re.search(r"\bunsatisfactory\b", response)
    return {"satisfactory": satisfactory, "raw_judgment": response}
