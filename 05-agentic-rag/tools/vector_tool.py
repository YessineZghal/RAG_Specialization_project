"""`vector_search(query)` — semantic search over the TriviaQA chunk corpus.

Wraps `agentic_common.retrieval.DenseRetriever` behind the exact call
signature an agent invokes as a tool: one string in, a list of
(chunk_id, text, score) results out — no retriever internals leak through.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agentic_common.retrieval import DenseRetriever  # noqa: E402


class VectorTool:
    name = "vector_search"
    description = "Search the local document corpus for passages relevant to a query."

    def __init__(self, retriever: DenseRetriever, corpus: dict[str, dict]) -> None:
        self.retriever = retriever
        self.corpus = corpus

    def __call__(self, query: str, top_k: int = 5) -> list[dict]:
        results = self.retriever.search(query, top_k=top_k)
        return [
            {
                "chunk_id": chunk_id,
                "text": self.corpus[chunk_id]["text"],
                "article_title": self.corpus[chunk_id]["article_title"],
                "score": score,
            }
            for chunk_id, score in results
        ]


class GetDocumentTool:
    """`get_document(document_id)` — fetch a full source article by title,
    reassembled from its chunks. Distinct from `vector_search`: this is
    for "give me the whole thing" once the agent knows *which* document it
    wants, not "find something relevant."
    """

    name = "get_document"
    description = "Fetch the full text of a specific source article by its title."

    def __init__(self, data) -> None:
        self.data = data  # agentic_common.dataset.TriviaData

    def __call__(self, document_id: str) -> str:
        text = self.data.get_document(document_id)
        return text or f"No document found with title {document_id!r}."
