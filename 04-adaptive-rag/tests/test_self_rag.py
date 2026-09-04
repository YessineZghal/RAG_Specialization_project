from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "self-rag"))
from self_rag import critique_answer, self_rag_answer  # noqa: E402


def test_critique_answer_true_when_grounded(fake_llm):
    llm = fake_llm(response="grounded")
    assert critique_answer("q", "context", "answer", llm=llm) is True


def test_critique_answer_false_when_ungrounded(fake_llm):
    llm = fake_llm(response="ungrounded")
    assert critique_answer("q", "context", "answer", llm=llm) is False


def test_self_rag_answer_stops_after_first_grounded_attempt(fake_retriever, tiny_corpus, fake_llm):
    # generate -> "grounded" critique -> should stop, no rewrite needed
    llm = fake_llm(responses=["some answer", "grounded"])
    result = self_rag_answer("q", fake_retriever, tiny_corpus, llm=llm, top_k=2, max_retries=2)
    assert result["grounded"] is True
    assert len(result["attempts"]) == 1


def test_self_rag_answer_retries_when_ungrounded(fake_retriever, tiny_corpus, fake_llm):
    # attempt 1: answer -> ungrounded -> rewrite; attempt 2: answer -> grounded -> stop
    llm = fake_llm(responses=["answer1", "ungrounded", "rewritten question", "answer2", "grounded"])
    result = self_rag_answer("q", fake_retriever, tiny_corpus, llm=llm, top_k=2, max_retries=2)
    assert len(result["attempts"]) == 2
    assert result["grounded"] is True
    assert result["final_answer"] == "answer2"


def test_self_rag_answer_gives_up_after_max_retries(fake_retriever, tiny_corpus, fake_llm):
    llm = fake_llm(responses=["a1", "ungrounded", "rw1", "a2", "ungrounded"])
    result = self_rag_answer("q", fake_retriever, tiny_corpus, llm=llm, top_k=2, max_retries=1)
    assert len(result["attempts"]) == 2  # 1 original + 1 retry, then stop
    assert result["grounded"] is False
