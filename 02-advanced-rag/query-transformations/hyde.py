"""HyDE — Hypothetical Document Embeddings.

Short queries and long documents don't always embed close together, even
when the document answers the query perfectly (a question and its answer
are phrased very differently). HyDE sidesteps this by asking the LLM to
*hallucinate* a plausible answer passage first, then embedding **that**
instead of the raw query — a fake document is often closer, in embedding
space, to the real one than the question was.

Only meaningful for embedding-based (dense) retrieval — pass a
`DenseRetriever`, not BM25.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.llm import OllamaLLM  # noqa: E402

HYDE_PROMPT = (
    "Write a short, factual passage (2-4 sentences) that would directly "
    "answer the following question, written as if it were an excerpt from a "
    "reference document. Do not mention that this is hypothetical or that "
    "you are an AI.\n\nQuestion: {query}\n\nPassage:"
)


def generate_hypothetical_document(query: str, llm: OllamaLLM | None = None) -> str:
    llm = llm or OllamaLLM()
    return llm.complete(HYDE_PROMPT.format(query=query)).strip()


def hyde_search(
    query: str,
    dense_retriever,
    top_k: int = 10,
    llm: OllamaLLM | None = None,
) -> list[tuple[str, float]]:
    hypothetical_doc = generate_hypothetical_document(query, llm=llm)
    return dense_retriever.search(hypothetical_doc, top_k=top_k)
