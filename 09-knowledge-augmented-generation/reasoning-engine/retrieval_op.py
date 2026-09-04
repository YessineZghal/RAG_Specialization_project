"""The `retrieval` operator: plain dense retrieval over the real
PubMedQA abstract corpus -- the same brute-force cosine search every
prior level uses, reused here as one of four operators instead of the
whole pipeline.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kag_common.embed import OllamaEmbedder, cosine_search  # noqa: E402


@dataclass(frozen=True)
class RetrievalHit:
    doc_id: str
    text: str
    score: float


def retrieve(
    question: str,
    corpus: dict[str, str],
    doc_ids: list[str],
    matrix: np.ndarray,
    embedder: OllamaEmbedder | None = None,
    top_k: int = 3,
) -> list[RetrievalHit]:
    embedder = embedder or OllamaEmbedder()
    query_vector = np.asarray(embedder.embed_one(question), dtype=np.float32)
    hits = cosine_search(query_vector, doc_ids, matrix, top_k=top_k)
    return [RetrievalHit(doc_id=doc_id, text=corpus[doc_id], score=score) for doc_id, score in hits if doc_id in corpus]
