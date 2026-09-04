"""Verify a generated answer against **real ground truth** — TriviaQA
ships multiple accepted phrasings (`normalized_aliases`) per question, so
this is genuine automatic correctness checking, not a proxy like "did we
retrieve the right document" (every prior level's evaluation approach).

Only usable during evaluation/testing, where the real answer is known —
never available to the agent itself while it's answering.
"""

from __future__ import annotations

import re


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def verify_answer(generated_answer: str, aliases: list[str]) -> bool:
    """True if any accepted alias appears in the generated answer (after
    normalization) -- a substring check is intentional here: a correct
    answer is often embedded in a longer sentence ("The answer is X.").
    """
    normalized_answer = normalize(generated_answer)
    return any(normalize(alias) in normalized_answer for alias in aliases if alias.strip())
