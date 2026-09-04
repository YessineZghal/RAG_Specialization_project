"""Chain-of-Thought (CoT) -- the baseline every other strategy in this
level is measured against. One linear prompt asks the model to reason
step by step before committing to a final answer, in a single LLM call.
No branching, no backtracking, no evaluator -- whatever reasoning path
the model picks first is the one it follows to the end.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reasoning_common.answer_parsing import parse_yes_no
from reasoning_common.llm import OllamaLLM

COT_PROMPT = (
    "Answer the following yes/no question using only the given context. "
    "Think through it step by step, showing your reasoning, then give "
    "your final answer on its own last line, in exactly this form: "
    "'Answer: Yes' or 'Answer: No'.\n\n"
    "Context:\n{context}\n\nQuestion: {question}\n\nReasoning:"
)


def cot_answer(question: str, context: str, llm: OllamaLLM | None = None) -> dict:
    """One linear reasoning chain, one LLM call. Returns the parsed
    boolean answer (`None` if the model's output could not be parsed),
    the raw reasoning text, and `llm_calls` -- every strategy in this
    level reports this same field so `evaluation/cost_tracker.py` can
    compare them on equal terms.
    """
    llm = llm or OllamaLLM()
    raw = llm.complete(COT_PROMPT.format(context=context, question=question))
    return {"answer": parse_yes_no(raw), "reasoning": raw, "llm_calls": 1}
