from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation.long_context_baseline import (
    format_documents,
    long_context_answer,
    long_context_search,
)


def test_format_documents_labels_each_document_with_its_id():
    corpus = {"doc-1": "First document text.", "doc-2": "Second document text."}
    formatted = format_documents(corpus)
    assert "[doc-1] First document text." in formatted
    assert "[doc-2] Second document text." in formatted


def test_long_context_answer_parses_a_clean_comma_separated_list(fake_llm):
    corpus = {"doc-1": "...", "doc-2": "...", "doc-3": "..."}
    llm = fake_llm(response="doc-2, doc-1")

    result = long_context_answer("some question", corpus, llm=llm)

    assert result == ["doc-2", "doc-1"]


def test_long_context_answer_returns_empty_list_for_none(fake_llm):
    corpus = {"doc-1": "..."}
    llm = fake_llm(response="NONE")
    assert long_context_answer("unanswerable question", corpus, llm=llm) == []


def test_long_context_answer_strips_brackets_the_model_echoes_from_the_prompt(fake_llm):
    # Real failure caught by actually running this against a live model:
    # `format_documents` labels each document "[doc_id]", and the model
    # sometimes echoes that exact bracket notation back in its answer
    # instead of the plain comma-separated list the prompt asks for. A
    # correct answer must not be discarded just because of this.
    corpus = {"25439264": "...", "4442799": "..."}
    llm = fake_llm(response="[25439264], [4442799]")

    result = long_context_answer("some question", corpus, llm=llm)

    assert result == ["25439264", "4442799"]


def test_long_context_answer_filters_out_hallucinated_doc_ids(fake_llm):
    corpus = {"doc-1": "...", "doc-2": "..."}
    llm = fake_llm(response="doc-1, doc-99, doc-2")

    result = long_context_answer("some question", corpus, llm=llm)

    # "doc-99" was never in the corpus -- a real failure mode when a model
    # is simply asked to output IDs from memory instead of choosing from
    # a fixed, retrieved candidate list.
    assert result == ["doc-1", "doc-2"]


def test_long_context_answer_sends_the_whole_corpus_in_the_prompt(fake_llm):
    corpus = {"doc-1": "UNIQUE MARKER TEXT"}
    llm = fake_llm(response="doc-1")

    long_context_answer("some question", corpus, llm=llm)

    assert "UNIQUE MARKER TEXT" in llm.calls[0]["prompt"]
    assert "doc-1" in llm.calls[0]["prompt"]


def test_long_context_search_runs_every_query_against_the_same_corpus(fake_llm):
    corpus = {"doc-1": "...", "doc-2": "..."}
    llm = fake_llm(response="doc-1")

    results = long_context_search({"q1": "question one", "q2": "question two"}, corpus, llm=llm)

    assert results == {"q1": ["doc-1"], "q2": ["doc-1"]}
    assert len(llm.calls) == 2
