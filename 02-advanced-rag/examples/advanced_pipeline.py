#!/usr/bin/env python3
"""Full advanced RAG pipeline: hybrid retrieval (dense + BM25 via RRF),
cross-encoder reranking, then generation — on the real, open BeIR/scifact
dataset.

Usage:
    cd 02-advanced-rag
    uv sync --extra sentence-transformers
    uv run python examples/advanced_pipeline.py "Is chronic rhinosinusitis associated with elevated ILC2s?"

First run downloads + embeds the scifact subset (a few minutes); every run
after that loads the cached corpus/embeddings from data/cache/ in seconds
(see common/dataset.py, common/embed.py).
"""

from __future__ import annotations

import sys
from pathlib import Path

_LEVEL_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_LEVEL_DIR))
sys.path.insert(0, str(_LEVEL_DIR / "hybrid-search"))  # hyphenated: top-level module import

from bm25_vector import HybridRetriever  # noqa: E402
from common.dataset import prepare  # noqa: E402
from common.llm import OllamaLLM  # noqa: E402
from reranking.cross_encoder import CrossEncoderReranker  # noqa: E402
from retrieval.dense import DenseRetriever  # noqa: E402
from retrieval.sparse import BM25Retriever  # noqa: E402

SYSTEM_PROMPT = (
    "You are a helpful scientific assistant. Answer using ONLY the provided "
    "context. If the context does not support an answer, say so. Cite "
    "source numbers like [Source 1]."
)


def build_prompt(question: str, sources: list[tuple[str, str, float]]) -> str:
    context = "\n\n".join(f"[Source {i}] {text}" for i, (_, text, _) in enumerate(sources, start=1))
    return f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer, citing source numbers:"


def main() -> None:
    question = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "Is chronic rhinosinusitis associated with elevated group 2 innate lymphoid cells?"
    )

    print("Loading scifact subset (cached after first run)...")
    data = prepare()
    print(f"Corpus: {len(data.corpus)} docs · {len(data.queries)} labeled eval queries available.\n")

    corpus_texts = {doc_id: data.corpus_text(doc_id) for doc_id in data.doc_ids()}

    print("Building dense retriever (embeddings cached after first run)...")
    dense = DenseRetriever.from_corpus(corpus_texts)
    print("Building sparse (BM25) retriever...")
    sparse = BM25Retriever.from_corpus(corpus_texts)
    hybrid = HybridRetriever(dense, sparse)

    candidates = hybrid.search(question, top_k=20)
    print(f"Hybrid retrieval: {len(candidates)} candidates (RRF-fused dense + BM25).")

    print("Reranking with a cross-encoder...")
    reranker = CrossEncoderReranker()
    candidate_texts = [(doc_id, corpus_texts[doc_id]) for doc_id, _ in candidates]
    reranked = reranker.rerank(question, candidate_texts, top_k=5)
    sources = [(doc_id, corpus_texts[doc_id], score) for doc_id, score in reranked]

    llm = OllamaLLM()
    answer = llm.complete(build_prompt(question, sources), system=SYSTEM_PROMPT)

    print(f"\nQ: {question}")
    print(f"A: {answer}\n")
    print("Top reranked sources:")
    for i, (doc_id, _text, score) in enumerate(sources, start=1):
        title = data.corpus[doc_id]["title"][:90]
        print(f"  [{i}] rerank_score={score:.3f} doc={doc_id} — {title}")


if __name__ == "__main__":
    main()
