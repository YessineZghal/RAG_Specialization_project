from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "query-transformations"))
from hyde import generate_hypothetical_document, hyde_search
from multi_query import generate_queries, multi_query_search
from query_rewrite import rewrite_query
from step_back import step_back_query, step_back_search


class FakeRetriever:
    def __init__(self):
        self.calls: list[str] = []

    def search(self, query: str, top_k: int):
        self.calls.append(query)
        return [(f"doc-for-{query[:10]}", 1.0)][:top_k]


def test_rewrite_query_strips_quotes_and_whitespace(fake_llm):
    llm = fake_llm(response='"a cleaner query"  ')
    assert rewrite_query("orig", llm=llm) == "a cleaner query"
    assert "orig" in llm.calls[0]["prompt"]


def test_generate_queries_splits_lines_and_strips_bullets(fake_llm):
    llm = fake_llm(response="- first query\n- second query\n* third query")
    queries = generate_queries("original question", n=3, llm=llm)
    assert queries == ["first query", "second query", "third query"]


def test_generate_queries_falls_back_to_original_if_llm_returns_nothing(fake_llm):
    llm = fake_llm(response="   ")
    assert generate_queries("original question", llm=llm) == ["original question"]


def test_multi_query_search_queries_retriever_for_every_variant(fake_llm):
    llm = fake_llm(response="variant one\nvariant two")
    retriever = FakeRetriever()

    multi_query_search("original", retriever, n=2, llm=llm)

    assert retriever.calls == ["original", "variant one", "variant two"]


def test_generate_hypothetical_document_returns_llm_output(fake_llm):
    llm = fake_llm(response="A hypothetical answer passage.")
    assert generate_hypothetical_document("some question", llm=llm) == "A hypothetical answer passage."


def test_hyde_search_embeds_the_hypothetical_document_not_the_query(fake_llm):
    llm = fake_llm(response="hypothetical passage text")
    retriever = FakeRetriever()

    hyde_search("original question", retriever, llm=llm)

    assert retriever.calls == ["hypothetical passage text"]


def test_step_back_query_strips_quotes(fake_llm):
    llm = fake_llm(response='"What is the general category?"')
    assert step_back_query("specific question", llm=llm) == "What is the general category?"


def test_step_back_search_queries_both_general_and_original(fake_llm):
    llm = fake_llm(response="general question")
    retriever = FakeRetriever()

    result = step_back_search("specific question", retriever, llm=llm)

    assert result["step_back_question"] == "general question"
    assert set(retriever.calls) == {"general question", "specific question"}
