"""Cost tracking -- an independent way to count real LLM calls, on top of
each strategy's own self-reported `llm_calls` field (returned by
`cot_answer`, `tree_of_thought_search`, `graph_of_thought_search`, and
`hgot_answer`). Wrapping the actual LLM client, rather than trusting only
each strategy's own internal counting, means a bug in one strategy's
count can never silently make it look cheaper than it really was --
`CostTrackingLLM.call_count` is ground truth, counted at the one place
every call actually passes through.

This independent count is exactly what this level's own Evaluation
section needs to report honestly: ToT and GoT are *expected* to cost
several times a single CoT call (branching and evaluating both cost real
calls) -- the question this level exists to answer with a real number,
not an assumption, is by how much, measured against Level 7's own
finding that a single generation call already costs several seconds on
this repo's CPU-bound Ollama setup.
"""

from __future__ import annotations


class CostTrackingLLM:
    """Duck-types any of this repo's `OllamaLLM`-shaped clients (or a
    test double): forwards every call unchanged, counting how many there
    were.
    """

    def __init__(self, llm) -> None:
        self._llm = llm
        self.call_count = 0

    def complete(self, prompt: str, system: str | None = None, temperature: float = 0.2) -> str:
        self.call_count += 1
        return self._llm.complete(prompt, system=system, temperature=temperature)

    def reset(self) -> None:
        self.call_count = 0
