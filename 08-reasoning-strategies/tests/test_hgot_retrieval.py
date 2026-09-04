from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "graph-of-thought"))
from hgot_retrieval import answer_subquestion, decompose_question, hgot_answer  # noqa: E402


def test_decompose_question_splits_lines_and_strips_numbering(fake_llm):
    llm = fake_llm(response="1. first sub-question\n2. second sub-question")
    subs = decompose_question("original question", n=2, llm=llm)
    assert subs == ["first sub-question", "second sub-question"]


def test_decompose_question_falls_back_to_the_original_question_if_unparseable(fake_llm):
    llm = fake_llm(response="   ")
    subs = decompose_question("original question", n=3, llm=llm)
    assert subs == ["original question"]


def test_answer_subquestion_retrieves_and_builds_context_from_the_real_corpus(fake_llm, fake_retriever):
    retriever = fake_retriever(results=[("fact-1", 0.9), ("fact-2", 0.8)])
    corpus = {"fact-1": "The sky is blue.", "fact-2": "Water is wet."}
    llm = fake_llm(response="Yes, based on the evidence.")

    result = answer_subquestion("is this true?", retriever, corpus, top_k=2, llm=llm)

    assert result["evidence_ids"] == ["fact-1", "fact-2"]
    assert result["answer_text"] == "Yes, based on the evidence."
    prompt = llm.calls[0]
    assert "The sky is blue." in prompt
    assert "Water is wet." in prompt


def test_answer_subquestion_handles_no_retrieved_evidence_gracefully(fake_llm, fake_retriever):
    retriever = fake_retriever(results=[])
    llm = fake_llm(response="Cannot answer without evidence.")

    result = answer_subquestion("is this true?", retriever, corpus={}, llm=llm)

    assert result["evidence_ids"] == []
    assert "(no evidence retrieved)" in llm.calls[0]


def test_hgot_answer_retrieves_separately_for_each_subquestion(fake_llm, fake_retriever):
    llm = fake_llm(
        responses=[
            "sub-question one\nsub-question two",  # decompose
            "Sub-answer one.",  # answer sub-question one
            "Sub-answer two.",  # answer sub-question two
            "Answer: Yes",  # final vote
        ]
    )
    retriever = fake_retriever(results=[("fact-1", 0.9)])
    corpus = {"fact-1": "Some real fact."}

    result = hgot_answer("original question", retriever, corpus, n_subquestions=2, llm=llm)

    assert result["sub_questions"] == ["sub-question one", "sub-question two"]
    assert retriever.calls == ["sub-question one", "sub-question two"]  # a real, separate search per sub-question
    assert result["answer"] is True
    assert result["llm_calls"] == 4  # 1 decompose + 2 sub-answers + 1 vote
    assert result["cited_evidence_ids"] == ["fact-1"]


def test_hgot_answer_deduplicates_evidence_cited_by_multiple_subquestions(fake_llm, fake_retriever):
    llm = fake_llm(
        responses=["sub A\nsub B", "answer A", "answer B", "Answer: No"]
    )
    # Both sub-questions happen to retrieve the same fact -- the citation
    # list should not list it twice.
    retriever = fake_retriever(results=[("fact-1", 0.9), ("fact-2", 0.8)])
    corpus = {"fact-1": "...", "fact-2": "..."}

    result = hgot_answer("question", retriever, corpus, n_subquestions=2, llm=llm)

    assert result["cited_evidence_ids"] == ["fact-1", "fact-2"]
