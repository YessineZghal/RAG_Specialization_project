"""Self-RAG — generate an answer, then have the model critique whether its
*own* answer is actually supported by the retrieved context, and retry
with a rewritten query if it isn't.

This catches a failure mode CRAG doesn't: CRAG grades the *evidence*
before generation; Self-RAG checks the *answer* after generation — an LLM
can still drift away from good evidence during generation, especially a
smaller local model.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from adaptive_common.llm import OllamaLLM

ANSWER_PROMPT = """Context:
{context}

Question: {question}
Answer using only the context above:"""

CRITIQUE_PROMPT = """Context:
{context}

Question: {question}
Proposed answer: {answer}

Is this answer fully supported by the context, with no unsupported claims?
Respond with only one word: grounded or ungrounded.
Judgment:"""

REWRITE_PROMPT = """The following question was not answerable from the retrieved context.
Rewrite it to be more specific or use different terms that might match the source text better.
Return ONLY the rewritten question.

Original question: {question}"""


def generate_answer(question: str, context: str, llm: OllamaLLM) -> str:
    return llm.complete(ANSWER_PROMPT.format(context=context, question=question))


def critique_answer(question: str, context: str, answer: str, llm: OllamaLLM) -> bool:
    response = llm.complete(
        CRITIQUE_PROMPT.format(context=context, question=question, answer=answer)
    ).strip().lower()
    return "ungrounded" not in response and "grounded" in response


def self_rag_answer(
    question: str,
    retriever,
    corpus: dict[str, str],
    llm: OllamaLLM | None = None,
    top_k: int = 5,
    max_retries: int = 1,
) -> dict:
    llm = llm or OllamaLLM()
    current_question = question
    attempts = []

    for attempt in range(max_retries + 1):
        results = retriever.search(current_question, top_k=top_k)
        context = "\n\n".join(corpus.get(doc_id, "") for doc_id, _ in results)

        answer = generate_answer(question, context, llm)
        grounded = critique_answer(question, context, answer, llm)
        attempts.append(
            {"attempt": attempt, "query_used": current_question, "answer": answer, "grounded": grounded}
        )

        if grounded:
            break
        current_question = llm.complete(REWRITE_PROMPT.format(question=question)).strip()

    return {"question": question, "final_answer": attempts[-1]["answer"], "grounded": attempts[-1]["grounded"], "attempts": attempts}
