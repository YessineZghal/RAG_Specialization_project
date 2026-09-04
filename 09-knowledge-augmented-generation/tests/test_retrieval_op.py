from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "reasoning-engine"))
from retrieval_op import retrieve


def test_retrieve_returns_the_closest_documents_by_cosine_similarity(fake_embedder):
    embedder = fake_embedder(vectors={"query": [1.0, 0.0, 0.0]})
    doc_ids = ["d1", "d2"]
    matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    corpus = {"d1": "closely related text", "d2": "unrelated text"}

    hits = retrieve("query", corpus, doc_ids, matrix, embedder, top_k=1)

    assert len(hits) == 1
    assert hits[0].doc_id == "d1"
    assert hits[0].text == "closely related text"


def test_retrieve_respects_top_k(fake_embedder):
    embedder = fake_embedder(vectors={"query": [1.0, 0.0, 0.0]})
    doc_ids = ["d1", "d2", "d3"]
    matrix = np.array([[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    corpus = {"d1": "a", "d2": "b", "d3": "c"}

    hits = retrieve("query", corpus, doc_ids, matrix, embedder, top_k=2)

    assert len(hits) == 2
    assert {h.doc_id for h in hits} == {"d1", "d2"}
