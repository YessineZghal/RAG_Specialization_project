from __future__ import annotations

import numpy as np

from retrieval.dense import DenseRetriever
from retrieval.sparse import BM25Retriever


def test_dense_retriever_ranks_by_cosine_similarity(fake_embedder, tiny_corpus):
    doc_ids = list(tiny_corpus.keys())
    matrix = np.array([fake_embedder.embed_one(tiny_corpus[d]) for d in doc_ids])
    retriever = DenseRetriever(doc_ids, matrix, fake_embedder)

    # Strong exact-token overlap with doc-refunds — the hash-trick fake
    # embedder has no stemming/synonyms, so the query must share literal
    # words, not just paraphrase the intended document.
    results = retriever.search("refunds are processed within thirty days of purchase", top_k=2)

    assert results[0][0] == "doc-refunds"
    assert len(results) == 2
    assert results[0][1] >= results[1][1]


def test_dense_retriever_top_k_clamped(fake_embedder, tiny_corpus):
    doc_ids = list(tiny_corpus.keys())
    matrix = np.array([fake_embedder.embed_one(tiny_corpus[d]) for d in doc_ids])
    retriever = DenseRetriever(doc_ids, matrix, fake_embedder)

    results = retriever.search("anything", top_k=100)
    assert len(results) == len(doc_ids)


def test_bm25_retriever_finds_exact_keyword_matches(tiny_corpus):
    retriever = BM25Retriever.from_corpus(tiny_corpus)

    results = retriever.search("probation period ninety day", top_k=1)

    assert results[0][0] == "doc-onboarding"


def test_bm25_retriever_returns_zero_scores_for_no_overlap(tiny_corpus):
    retriever = BM25Retriever.from_corpus(tiny_corpus)
    results = retriever.search("zzz nonexistent qqq", top_k=3)
    assert all(score == 0.0 for _, score in results)
