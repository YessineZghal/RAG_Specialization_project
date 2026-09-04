"""Classify a query into one of four complexity tiers, each mapped to a
different retrieval strategy in `dynamic-retrieval/retrieval_policy.py`:

- "none"      — conversational, no retrieval needed
- "simple"    — a single factual lookup, one retrieval step
- "complex"   — needs comparing/combining more than one fact at once
                (HotpotQA's "comparison" questions land here)
- "multi_hop" — needs a chain of lookups where the answer to one step
                names the next thing to look up (HotpotQA's "bridge"
                questions land here)

Real HotpotQA questions are inherently 2-document questions by
construction — there is no genuinely single-hop question in the dataset
itself, which is exactly why "none" and "simple" examples in this level's
evaluation are hand-authored rather than pulled from HotpotQA (see
`../README.md#dataset`).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from adaptive_common.llm import OllamaLLM  # noqa: E402

LABELS = ("none", "simple", "complex", "multi_hop")

_CONVERSATIONAL_RE = re.compile(
    r"^\s*(hi|hello|hey|thanks|thank you|bye|goodbye|good (morning|evening|afternoon))\b",
    re.IGNORECASE,
)
_COMPARISON_RE = re.compile(
    r"\b(both|compare|comparison|same|older|younger|which of|more than|less than|"
    r"as (much|many|old|young) as|earlier|later|first)\b|"
    # "Which X, A or B?" / "Who is older, A or B?" -- an enumerated-alternatives
    # question, the comma is what distinguishes this from an incidental "or"
    # elsewhere in an otherwise simple question.
    r"\b(which|who)\b.{0,80},.{0,80}\bor\b",
    re.IGNORECASE,
)
# Indirect/descriptive reference to an unnamed entity ("the director who...",
# "the film that...") is the classic bridge-question shape: you must resolve
# who/what is being described before you can look up the actual answer.
_BRIDGE_RE = re.compile(
    r"\b(the \w+ (who|that|which)|who (directed|wrote|created|founded).{0,40}(that|which)|"
    r"what is the .{0,30} of the \w+ (who|that|which))",
    re.IGNORECASE,
)


def classify_rule(question: str) -> str:
    # NB: do NOT use "lacks a '?'" as a signal for "none" -- real HotpotQA
    # questions sometimes lack trailing punctuation entirely (a benign data
    # quirk), which made this misclassify genuine questions as small talk.
    if _CONVERSATIONAL_RE.search(question):
        return "none"
    if _COMPARISON_RE.search(question):
        return "complex"
    if _BRIDGE_RE.search(question):
        return "multi_hop"
    return "simple"


# Zero-shot classification badly under-uses "complex" with small local
# models (measured: 0/15 correct on real comparison questions, defaulting
# to "simple" almost every time) -- a few worked examples per category
# fixed it completely (measured: 15/15). Keep the examples; don't regress
# to a bare category-definition prompt.
CLASSIFY_PROMPT = """Classify the question into exactly one category: none, simple, complex, or multi_hop.

Examples:
Q: Hi, how are you?
Category: none

Q: What is the capital of France?
Category: simple

Q: Were both Gabriela Mistral and G. K. Chesterton authors?
Category: complex

Q: Which magazine was published weekly, Aeon or Life?
Category: complex

Q: What nationality is the director of the film that starred an actor born in 1990?
Category: multi_hop

Q: Peter Hobbs founded the company that is based in what town in Manchester?
Category: multi_hop

Now classify this one. Respond with only one word.

Q: {question}
Category:"""


def classify_llm(question: str, llm: OllamaLLM | None = None) -> str:
    llm = llm or OllamaLLM()
    response = llm.complete(CLASSIFY_PROMPT.format(question=question)).strip().lower()
    # Models reliably "correct" multi_hop to the more standard English
    # spelling "multi-hop" despite the prompt spelling it with an
    # underscore -- normalize before matching rather than silently losing
    # every multi-hop classification to the default.
    normalized = response.replace("-", "_").replace(" ", "_")
    for label in LABELS:
        if label in normalized:
            return label
    return "simple"  # safe default if the model answers off-script


def classify_ensemble(question: str, llm: OllamaLLM | None = None) -> str:
    """Combine both classifiers using their *measured* relative strengths
    rather than guessing: on a real 15-question sample per HotpotQA type,
    the rule classifier caught more genuine multi_hop (bridge) questions
    (6/15 vs. the few-shot LLM's 3/15 — the LLM persistently over-applies
    "complex" once it has seen a "complex" example), while the LLM
    classifier was far more reliable at complex vs. simple (15/15 vs. the
    rule classifier's 12/15). So: trust the rule classifier's "none" and
    "multi_hop" calls (cheap and relatively strong there), and defer to the
    LLM for anything it doesn't confidently flag as either.
    """
    rule_result = classify_rule(question)
    if rule_result in ("none", "multi_hop"):
        return rule_result
    return classify_llm(question, llm=llm)
