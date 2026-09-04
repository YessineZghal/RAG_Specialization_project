"""Long-context baseline — skip chunking and retrieval entirely and hand
the whole corpus to the LLM directly, letting it pick which document(s)
answer the question.

Every other technique in this level assumes retrieval is necessary and
tries to make it better. This module asks the opposite question: modern
LLMs have large context windows — for a *small enough* corpus, is
retrieval even needed at all, or can the model just be shown everything
and asked to find the answer itself?

Returns the same `{query_id: [doc_id, ...]}` shape `evaluation/recall_at_k.py`,
`mrr.py`, and `ndcg.py` already expect, so this baseline can be scored
with the exact same metrics as dense, sparse, and hybrid retrieval and
compared on equal terms.

**A real ceiling, not a hidden one**: this only works while the whole
corpus fits comfortably in the model's context window. scifact's full
1,000-document subset does not — this baseline is meant to be run against
a small sample (a few dozen documents), specifically to measure *how*
retrieval's advantage grows as the corpus grows past what a single
prompt can hold, not to replace retrieval as this level's default.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.llm import OllamaLLM  # noqa: E402

LONG_CONTEXT_PROMPT = (
    "Below are several documents, each labeled with its ID in square "
    "brackets. Read the question, then decide which document or documents "
    "actually answer it, ranked from most to least relevant.\n\n"
    "{documents}\n\n"
    "Question: {query}\n\n"
    "Respond with ONLY a comma-separated list of document IDs, most "
    "relevant first (for example: doc-12, doc-4). If no document answers "
    "the question, respond with exactly: NONE"
)


def format_documents(corpus: dict[str, str]) -> str:
    return "\n\n".join(f"[{doc_id}] {text}" for doc_id, text in corpus.items())


def long_context_answer(query: str, corpus: dict[str, str], llm: OllamaLLM | None = None) -> list[str]:
    """Ask the LLM to rank every document in `corpus` for `query`,
    directly, with no retrieval step at all.
    """
    llm = llm or OllamaLLM()
    prompt = LONG_CONTEXT_PROMPT.format(documents=format_documents(corpus), query=query)
    response = llm.complete(prompt).strip()

    if response.upper().startswith("NONE"):
        return []

    # Real, observed failure mode: the documents in the prompt are labeled
    # "[doc_id]" (see `format_documents`), and the model sometimes echoes
    # that exact bracket notation back even though the prompt asks for a
    # plain comma-separated list ("[25439264], [4442799]" instead of
    # "25439264, 4442799"). Strip brackets and stray punctuation from each
    # candidate before checking it against the real corpus ids, or a
    # perfectly correct answer gets thrown away as "hallucinated" just
    # because of its formatting.
    candidate_ids = [
        doc_id.strip().strip("[]() ") for doc_id in response.split(",") if doc_id.strip()
    ]
    # A model can invent a doc_id that does not exist, or wrap its answer
    # in extra words despite the prompt -- keep only ids that are real,
    # in the order the model gave them.
    return [doc_id for doc_id in candidate_ids if doc_id in corpus]


def long_context_search(
    queries: dict[str, str],
    corpus: dict[str, str],
    llm: OllamaLLM | None = None,
) -> dict[str, list[str]]:
    """Run `long_context_answer` for every query in `queries`. Intended to
    be scored with `evaluation/recall_at_k.py`, `mrr.py`, and `ndcg.py`
    exactly like any other method's results.
    """
    llm = llm or OllamaLLM()
    return {query_id: long_context_answer(query, corpus, llm=llm) for query_id, query in queries.items()}
