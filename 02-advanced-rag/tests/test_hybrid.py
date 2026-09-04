from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "hybrid-search"))
from bm25_vector import HybridRetriever  # noqa: E402

from retrieval.dense import DenseRetriever
from retrieval.sparse import BM25Retriever


def test_hybrid_retriever_combines_dense_and_sparse(fake_embedder, tiny_corpus):
    doc_ids = list(tiny_corpus.keys())
    matrix = np.array([fake_embedder.embed_one(tiny_corpus[d]) for d in doc_ids])
    dense = DenseRetriever(doc_ids, matrix, fake_embedder)
    sparse = BM25Retriever.from_corpus(tiny_corpus)
    hybrid = HybridRetriever(dense, sparse, candidate_k=3)

    results = hybrid.search("refund days purchase", top_k=3)

    assert len(results) > 0
    assert results[0][0] == "doc-refunds"


def test_hybrid_retriever_still_surfaces_bm25_only_matches(tiny_corpus, fake_embedder):
    # A rare, exact keyword that BM25 will find but a weak fake embedder might not rank first.
    doc_ids = list(tiny_corpus.keys())
    matrix = np.array([fake_embedder.embed_one(tiny_corpus[d]) for d in doc_ids])
    dense = DenseRetriever(doc_ids, matrix, fake_embedder)
    sparse = BM25Retriever.from_corpus(tiny_corpus)
    hybrid = HybridRetriever(dense, sparse, candidate_k=3)

    result_ids = {doc_id for doc_id, _ in hybrid.search("seven business days shipping", top_k=3)}
    assert "doc-shipping" in result_ids
