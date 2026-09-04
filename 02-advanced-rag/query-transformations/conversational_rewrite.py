"""Conversational query rewriting — resolve pronouns and follow-up phrasing
against prior conversation turns before retrieval ever runs.

`query_rewrite.py` already asks the LLM to clean up one query in
isolation. This module is the same idea applied to a chatbot's actual
situation: the current question is often incomplete on its own ("What
about the enterprise plan?" only makes sense after a prior question about
pricing). A retriever has no memory — it only ever sees the one string it
is given — so resolving "it," "that one," or "the enterprise plan" against
history has to happen before retrieval, not after.

If there is no history yet (the first turn of a conversation), the
question is already standalone and is returned unchanged without an LLM
call — there is nothing to resolve.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.llm import OllamaLLM  # noqa: E402

CONVERSATIONAL_REWRITE_PROMPT = (
    "The following is a conversation between a user and an assistant, "
    "followed by the user's next question. Rewrite the next question as a "
    "single, standalone question that makes sense without the conversation "
    "-- resolve any pronoun or implicit reference (\"it\", \"that one\", "
    "\"the other one\") using the conversation. Do not answer the question, "
    "only rewrite it. Return ONLY the rewritten question.\n\n"
    "Conversation so far:\n{history}\n\n"
    "Next question: {question}\n\nStandalone question:"
)


def format_history(history: list[tuple[str, str]]) -> str:
    """`history`: a list of `(speaker, text)` turns, oldest first, e.g.
    `[("user", "What is our refund policy?"), ("assistant", "30 days.")]`.
    """
    return "\n".join(f"{speaker}: {text}" for speaker, text in history)


def rewrite_with_history(
    question: str,
    history: list[tuple[str, str]],
    llm: OllamaLLM | None = None,
) -> str:
    if not history:
        return question
    llm = llm or OllamaLLM()
    prompt = CONVERSATIONAL_REWRITE_PROMPT.format(history=format_history(history), question=question)
    return llm.complete(prompt).strip().strip('"')


def conversational_search(
    question: str,
    history: list[tuple[str, str]],
    retriever,
    top_k: int = 10,
    llm: OllamaLLM | None = None,
) -> dict:
    """Rewrite `question` against `history`, then retrieve on the
    rewritten form. Returns both the rewritten question and the results
    so a caller can show what was actually resolved.
    """
    rewritten = rewrite_with_history(question, history, llm=llm)
    return {
        "rewritten_question": rewritten,
        "results": retriever.search(rewritten, top_k=top_k),
    }
