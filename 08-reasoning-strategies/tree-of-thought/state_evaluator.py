"""Tree-of-Thought's state evaluator -- scores how promising a partial
reasoning path looks, so `tree_search.py` can keep exploring the
best-scoring branches and prune the rest instead of exploring every
branch equally deep.

This evaluator is itself an LLM call, judging the model's own reasoning.
Expect the same class of unreliability every prior level found in its own
judge calls (Level 4's CRAG grading, Level 7's faithfulness judge) --
verify this against real data before trusting it blindly; see the
README's Common Failure Modes once this level has actually been run.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reasoning_common.llm import OllamaLLM

from thought_generator import format_partial_path

EVALUATE_PROMPT = (
    "Context:\n{context}\n\nQuestion: {question}\n\n"
    "Reasoning path so far:\n{path}\n\n"
    "On a scale from 1 (unpromising -- off-track, or contradicts the "
    "context) to 10 (very promising -- clearly grounded in the context "
    "and heading toward a correct final answer), how good is this "
    "reasoning path? Respond with ONLY a single integer, nothing else."
)

_NUMBER_RE = re.compile(r"(\d+(?:\.\d+)?)")


def evaluate_state(
    question: str,
    context: str,
    path: list[str],
    llm: OllamaLLM | None = None,
) -> float:
    """Returns a score in `[0.0, 1.0]` (the raw 1-10 rating, normalized).
    A response with no parseable number scores `0.0` -- treated as
    unpromising rather than crashing the search, the same
    degrade-gracefully-on-malformed-output discipline used throughout
    this repo.
    """
    llm = llm or OllamaLLM()
    prompt = EVALUATE_PROMPT.format(context=context, question=question, path=format_partial_path(path))
    raw = llm.complete(prompt)
    match = _NUMBER_RE.search(raw)
    if not match:
        return 0.0
    raw_score = float(match.group(1))
    return max(0.0, min(1.0, raw_score / 10.0))
