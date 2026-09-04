from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "corrective-rag"))
from crag import corrective_retrieve, grade_passage  # noqa: E402


def test_grade_passage_parses_relevant(fake_llm):
    llm = fake_llm(response="relevant")
    assert grade_passage("q", "p", llm=llm) == "relevant"


def test_grade_passage_parses_irrelevant(fake_llm):
    llm = fake_llm(response="This passage is irrelevant to the question.")
    assert grade_passage("q", "p", llm=llm) == "irrelevant"


def test_grade_passage_defaults_to_ambiguous_on_unparseable_output(fake_llm):
    llm = fake_llm(response="I cannot tell.")
    assert grade_passage("q", "p", llm=llm) == "ambiguous"


def test_corrective_retrieve_trustworthy_when_all_relevant(fake_retriever, tiny_corpus, fake_llm):
    llm = fake_llm(response="relevant")
    result = corrective_retrieve("Where is Russell Hobbs based?", fake_retriever, tiny_corpus, llm=llm, top_k=2)
    assert result["confidence"] == 1.0
    assert result["trustworthy"] is True


def test_corrective_retrieve_untrustworthy_when_all_irrelevant(fake_retriever, tiny_corpus, fake_llm):
    llm = fake_llm(response="irrelevant")
    result = corrective_retrieve("anything", fake_retriever, tiny_corpus, llm=llm, top_k=2)
    assert result["confidence"] == 0.0
    assert result["trustworthy"] is False


def test_corrective_retrieve_respects_min_relevant(fake_retriever, tiny_corpus, fake_llm):
    # 1 relevant out of 2 -- enough for the default (min_relevant=1)...
    llm = fake_llm(responses=["relevant", "irrelevant"])
    result = corrective_retrieve("q", fake_retriever, tiny_corpus, llm=llm, top_k=2)
    assert result["n_relevant"] == 1
    assert result["confidence"] == 0.5
    assert result["trustworthy"] is True

    # ...but not enough if the caller demands at least 2.
    llm2 = fake_llm(responses=["relevant", "irrelevant"])
    result2 = corrective_retrieve("q", fake_retriever, tiny_corpus, llm=llm2, top_k=2, min_relevant=2)
    assert result2["trustworthy"] is False
