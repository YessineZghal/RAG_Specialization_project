"""Sparse (keyword) retrieval via BM25.

Where dense retrieval matches on *meaning*, BM25 matches on *exact term
overlap*, weighted by term rarity (IDF) and document length. It has no
notion of synonyms, but it is unbeatable for exact identifiers, numbers,
and rare technical terms that an embedding model may blur together — the
motivation for `hybrid-search/`.
"""

from __future__ import annotations

import sys
from pathlib import Path

from rank_bm25 import BM25Okapi

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # repo root
from shared.utils.text import tokenize  # noqa: E402


class BM25Retriever:
    def __init__(self, doc_ids: list[str], corpus_texts: dict[str, str]) -> None:
        self.doc_ids = doc_ids
        tokenized_corpus = [tokenize(corpus_texts[doc_id]) for doc_id in doc_ids]
        self._bm25 = BM25Okapi(tokenized_corpus)

    @classmethod
    def from_corpus(cls, corpus_texts: dict[str, str]) -> "BM25Retriever":
        return cls(list(corpus_texts.keys()), corpus_texts)

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        scores = self._bm25.get_scores(tokenize(query))
        top_k = min(top_k, len(self.doc_ids))
        top_indices = scores.argsort()[::-1][:top_k]
        return [(self.doc_ids[i], float(scores[i])) for i in top_indices]
