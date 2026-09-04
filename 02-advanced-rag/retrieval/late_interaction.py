"""Late-interaction retrieval (the mechanism behind ColBERT).

`dense.py` represents a whole document as **one** vector and compares it
to the query's one vector. Late interaction represents a document as
**several** vectors and compares every one of them against every vector
on the query side, keeping only the best match for each query vector:

    MaxSim(Q, D) = sum over each query vector q  of  ( max over every
                   document vector d  of  cosine(q, d) )

Summing the best match per query vector (instead of averaging everything
into one vector first, the way dense retrieval does) means a document
only has to be strongly relevant to *part* of the query to score well —
useful when a question has several distinct parts and no single sentence
in a document answers all of them at once.

**Disclosed approximation**: real ColBERT computes MaxSim over per-*token*
embeddings from a model specifically trained to produce them. Ollama's
embeddings endpoint returns one pooled vector per input text and has no
per-token mode, so this module approximates the same MaxSim *mechanism*
at a coarser granularity: each document is represented by one vector per
**sentence** (via `chunking.semantic.split_sentences`), and each query by
one vector per **word**. This is real multi-vector, real MaxSim scoring —
just at sentence/word granularity instead of the sub-word granularity a
purpose-built ColBERT model uses.

**Performance note**: embedding one vector per document sentence and one
per query word means many more Ollama calls than `dense.py`'s one call
per document. Cache the per-sentence vectors the same way
`common/embed.py` caches whole-document vectors, and keep
`max_sentences_per_doc` modest — this is meant to demonstrate the MaxSim
mechanism clearly, not to replace dense retrieval as this level's default
for the full corpus.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def maxsim_score(query_vectors: list[np.ndarray], doc_vectors: list[np.ndarray]) -> float:
    """MaxSim(Q, D): for every query vector, the best-matching document
    vector's similarity, summed across all query vectors.
    """
    if not query_vectors or not doc_vectors:
        return 0.0
    return sum(max(_cosine(q, d) for d in doc_vectors) for q in query_vectors)


class LateInteractionRetriever:
    def __init__(
        self,
        doc_ids: list[str],
        doc_vectors: list[list[np.ndarray]],
        embed_fn: Callable[[str], list[float]],
    ) -> None:
        """`doc_vectors[i]` is the list of per-sentence vectors for `doc_ids[i]`."""
        self.doc_ids = doc_ids
        self.doc_vectors = doc_vectors
        self.embed_fn = embed_fn

    @classmethod
    def from_corpus(
        cls,
        corpus_texts: dict[str, str],
        embed_fn: Callable[[str], list[float]],
        max_sentences_per_doc: int = 8,
    ) -> "LateInteractionRetriever":
        from chunking.semantic import split_sentences

        doc_ids = list(corpus_texts.keys())
        doc_vectors: list[list[np.ndarray]] = []
        for doc_id in doc_ids:
            text = corpus_texts[doc_id]
            sentences = split_sentences(text)[:max_sentences_per_doc] or [text]
            doc_vectors.append([np.array(embed_fn(sentence)) for sentence in sentences])
        return cls(doc_ids, doc_vectors, embed_fn)

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        query_words = query.split()
        query_vectors = [np.array(self.embed_fn(word)) for word in query_words]
        if not query_vectors:
            query_vectors = [np.array(self.embed_fn(query))]

        scores = [
            (doc_id, maxsim_score(query_vectors, vectors))
            for doc_id, vectors in zip(self.doc_ids, self.doc_vectors, strict=True)
        ]
        scores.sort(key=lambda pair: pair[1], reverse=True)
        return scores[: min(top_k, len(scores))]
