"""Tree-of-Thought's thought generator -- given the reasoning path built
so far, propose several different possible next steps instead of
committing to just one, the way Chain-of-Thought does. Each proposal
becomes a new branch `tree_search.py` can independently score and either
continue exploring or prune away.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reasoning_common.llm import OllamaLLM  # noqa: E402

GENERATE_PROMPT = (
    "You are reasoning step by step to answer a yes/no question using "
    "only the given context.\n\n"
    "Context:\n{context}\n\nQuestion: {question}\n\n"
    "Reasoning so far:\n{partial}\n\n"
    "Propose {k} different possible NEXT reasoning steps that could follow "
    "from the reasoning so far -- genuinely different from each other, "
    "each a single short sentence. Return ONLY the steps, one per line, "
    "no numbering."
)


def format_partial_path(path: list[str]) -> str:
    return "\n".join(path) if path else "(nothing yet -- this is the first step)"


def generate_thoughts(
    question: str,
    context: str,
    partial_path: list[str],
    k: int = 3,
    llm: OllamaLLM | None = None,
) -> list[str]:
    llm = llm or OllamaLLM()
    prompt = GENERATE_PROMPT.format(
        context=context, question=question, partial=format_partial_path(partial_path), k=k
    )
    raw = llm.complete(prompt)
    thoughts = [line.strip("-*0123456789. \t") for line in raw.splitlines() if line.strip()]
    thoughts = [t for t in thoughts if t]
    return thoughts[:k] if thoughts else [raw.strip()]
