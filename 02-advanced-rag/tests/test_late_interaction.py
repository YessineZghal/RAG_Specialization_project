from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from retrieval.late_interaction import LateInteractionRetriever, maxsim_score


def test_maxsim_score_sums_the_best_match_per_query_vector():
    query_vectors = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    doc_vectors = [np.array([1.0, 0.0]), np.array([0.0, 0.5])]

    score = maxsim_score(query_vectors, doc_vectors)

    # query vector 1 matches doc vector 1 perfectly (cosine 1.0); query
    # vector 2 matches doc vector 2 perfectly too (same direction, cosine
    # 1.0 regardless of magnitude) -- so the sum is 1.0 + 1.0. `approx`
    # absorbs the tiny epsilon `_cosine` adds to its denominator to avoid
    # dividing by zero on a zero vector.
    assert score == pytest.approx(2.0)


def test_maxsim_score_is_zero_for_empty_input():
    assert maxsim_score([], [np.array([1.0, 0.0])]) == 0.0
    assert maxsim_score([np.array([1.0, 0.0])], []) == 0.0


_VECTORS = {
    "Cats are mammals.": [1.0, 0.0],
    "Dogs are mammals too.": [0.9, 0.1],
    "The sun is a star.": [0.0, 1.0],
    "Stars produce light.": [0.1, 0.9],
    "cats": [1.0, 0.0],
    "star": [0.0, 1.0],
}


def _controlled_embed(text: str) -> list[float]:
    return _VECTORS[text]


def test_from_corpus_builds_one_vector_per_sentence_per_document():
    corpus = {
        "doc-a": "Cats are mammals. Dogs are mammals too.",
        "doc-b": "The sun is a star. Stars produce light.",
    }
    retriever = LateInteractionRetriever.from_corpus(corpus, embed_fn=_controlled_embed)

    assert retriever.doc_ids == ["doc-a", "doc-b"]
    assert len(retriever.doc_vectors[0]) == 2  # two sentences in doc-a
    assert len(retriever.doc_vectors[1]) == 2  # two sentences in doc-b


def test_search_ranks_the_document_matching_the_querys_topic_first():
    corpus = {
        "doc-a": "Cats are mammals. Dogs are mammals too.",
        "doc-b": "The sun is a star. Stars produce light.",
    }
    retriever = LateInteractionRetriever.from_corpus(corpus, embed_fn=_controlled_embed)

    results = retriever.search("cats", top_k=2)

    assert results[0][0] == "doc-a"
    assert results[0][1] > results[1][1]


def test_search_respects_top_k():
    corpus = {
        "doc-a": "Cats are mammals. Dogs are mammals too.",
        "doc-b": "The sun is a star. Stars produce light.",
    }
    retriever = LateInteractionRetriever.from_corpus(corpus, embed_fn=_controlled_embed)

    results = retriever.search("cats", top_k=1)

    assert len(results) == 1
    assert results[0][0] == "doc-a"
