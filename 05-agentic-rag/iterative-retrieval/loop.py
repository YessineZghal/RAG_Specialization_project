"""The retrieve -> inspect -> decide -> retrieve-again loop, as a
standalone, reusable piece — this is the "evidence gathering" sub-routine
`agents/rag_agent.py` calls into, separated out so it can be tested and
reasoned about on its own.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agentic_common.llm import OllamaLLM

SUFFICIENCY_PROMPT = """Question: {question}

Evidence gathered so far:
{evidence}

Is this evidence sufficient to answer the question? Respond with only one word: yes or no.
Judgment:"""


def is_evidence_sufficient(question: str, evidence: str, llm: OllamaLLM | None = None) -> bool:
    if not evidence.strip():
        return False
    llm = llm or OllamaLLM()
    response = llm.complete(SUFFICIENCY_PROMPT.format(question=question, evidence=evidence)).strip().lower()
    # Word-boundary matching, not substring containment -- "no" is a
    # literal substring of "not", which bit exactly this kind of check
    # once already in this repo (see 04-adaptive-rag's "relevant" being a
    # substring of "irrelevant"). Check "no" first: a response like
    # "no, not sufficient" must not match "yes" via some other substring.
    if re.search(r"\bno\b", response):
        return False
    return bool(re.search(r"\byes\b", response))


def iterative_retrieve(
    question: str,
    vector_tool,
    llm: OllamaLLM | None = None,
    max_iterations: int = 3,
    top_k: int = 3,
) -> dict:
    """Keep retrieving with the vector tool, widening/rephrasing the query
    each round, until the LLM judges the pooled evidence sufficient or the
    iteration budget runs out.
    """
    llm = llm or OllamaLLM()
    query = question
    pooled: dict[str, dict] = {}
    rounds = []

    for i in range(max_iterations):
        results = vector_tool(query, top_k=top_k)
        for r in results:
            pooled[r["chunk_id"]] = r
        evidence = "\n\n".join(r["text"] for r in pooled.values())
        sufficient = is_evidence_sufficient(question, evidence, llm=llm)
        rounds.append({"iteration": i, "query": query, "n_new": len(results), "sufficient": sufficient})

        if sufficient:
            break
        query = llm.complete(
            f"The evidence gathered so far did not fully answer this question: {question}\n"
            f"Suggest a different, more specific search query. Return ONLY the query."
        ).strip()

    return {"question": question, "evidence": list(pooled.values()), "rounds": rounds}
