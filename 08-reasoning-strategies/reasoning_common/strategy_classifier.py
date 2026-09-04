"""Classify a question into which reasoning strategy it needs -- the
missing mini project from this level's build order: "an end-to-end
example choosing a strategy per question type" (`TASK.md`).

Four categories, mapped one-to-one onto this level's four strategies:

  - simple        -> cot   (one fact settles it, no branching needed)
  - comparative   -> tot   (weigh competing hypotheses before committing)
  - combinatorial -> got   (combine independent facts into one conclusion)
  - multi_hop     -> hgot  (splits into distinct sub-questions, answered
                            and retrieved separately, then combined)

This is a real classification decision with its own error rate, same
failure class as Level 4's query classifier and Level 3's backend router
(both measured, not assumed correct) and Level 9's logical-form parser
(fail-open on an unparseable response). `classify_strategy` follows the
same fail-open discipline: an unparseable or ambiguous response falls
back to `"cot"` -- this level's own real evaluation
(`notebooks/04_reasoning_vs_plain_rag_eval.ipynb`) found CoT the
strongest, cheapest strategy on real StrategyQA questions, so a
classifier failure defaulting to it is a safe failure, not an arbitrary
one.
"""

from __future__ import annotations

import re

from .llm import OllamaLLM

CLASSIFY_PROMPT = """Classify the following yes/no question into exactly one of these four categories, based on what kind of reasoning it needs:

- simple: a single, direct fact settles it -- no branching or combining needed
- comparative: requires weighing two or more competing hypotheses or explanations before committing to one
- combinatorial: requires combining two or more independent facts together to reach one conclusion
- multi_hop: naturally splits into distinct sub-questions that need to be answered separately, then combined

Respond with ONLY the category name (simple, comparative, combinatorial, or multi_hop), nothing else.

Question: {question}

Category:"""

_CATEGORY_TO_STRATEGY = {
    "simple": "cot",
    "comparative": "tot",
    "combinatorial": "got",
    "multi_hop": "hgot",
}

_CATEGORY_RE = re.compile(r"\b(simple|comparative|combinatorial|multi_hop|multi-hop)\b", re.IGNORECASE)

FALLBACK_STRATEGY = "cot"


def classify_strategy(question: str, llm: OllamaLLM | None = None) -> str:
    """Return one of `"cot"`, `"tot"`, `"got"`, `"hgot"`. Falls open to
    `"cot"` on an unparseable or ambiguous response -- never raises."""
    llm = llm or OllamaLLM()
    raw = llm.complete(CLASSIFY_PROMPT.format(question=question), temperature=0.0)

    matches = {m.group(1).lower().replace("-", "_") for m in _CATEGORY_RE.finditer(raw)}
    if len(matches) != 1:
        return FALLBACK_STRATEGY  # none, or more than one category mentioned -- genuinely ambiguous

    category = next(iter(matches))
    return _CATEGORY_TO_STRATEGY.get(category, FALLBACK_STRATEGY)
