"""A generic retrieve-judge-retry loop: keep retrying with a rewritten
query until either the evidence looks trustworthy or a retry budget runs
out — then, only then, fall back to the web.

This is the piece that ties `corrective-rag/`, `self-rag/`'s rewrite step,
and `web_fallback.py` together into one policy, matching the level's
"weak retrieved evidence -> retry / rewrite / web fallback" decision rule.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from adaptive_common.llm import OllamaLLM  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from web_fallback import web_fallback_context  # noqa: E402

REWRITE_PROMPT = (
    "Rewrite this question with different, more specific search terms. "
    "Return ONLY the rewritten question.\n\nQuestion: {question}"
)


def retrieve_with_retry(
    question: str,
    retriever,
    corpus: dict[str, str],
    llm: OllamaLLM | None = None,
    top_k: int = 5,
    max_retries: int = 2,
    min_relevant: int = 1,
) -> dict:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "corrective-rag"))
    from crag import corrective_retrieve

    llm = llm or OllamaLLM()
    current_question = question
    history = []

    for attempt in range(max_retries + 1):
        graded = corrective_retrieve(
            current_question, retriever, corpus, llm=llm, top_k=top_k, min_relevant=min_relevant
        )
        history.append({"attempt": attempt, "query": current_question, "confidence": graded["confidence"]})

        if graded["trustworthy"]:
            return {"question": question, "source": "corpus", "result": graded, "history": history}

        current_question = llm.complete(REWRITE_PROMPT.format(question=current_question)).strip()

    # Exhausted retries — fall back to the web rather than answer from weak evidence.
    web_context = web_fallback_context(question)
    return {
        "question": question,
        "source": "web_fallback",
        "result": {"context": web_context},
        "history": history,
    }
