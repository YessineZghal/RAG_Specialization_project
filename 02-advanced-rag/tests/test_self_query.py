from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "metadata-filtering"))
from self_query import SelfQueryFilters, parse_self_query, self_query_search


class FakeRetriever:
    def __init__(self):
        self.calls: list[str] = []

    def search(self, query: str, top_k: int):
        self.calls.append(query)
        return [("a", 0.9), ("b", 0.8), ("c", 0.7), ("d", 0.6)][:top_k]


def test_parse_self_query_reads_a_clean_json_response(fake_llm):
    llm = fake_llm(
        response='{"semantic_query": "AI papers", "min_year": 2023, "max_year": null, "length_bucket": "short"}'
    )
    parsed = parse_self_query("AI papers from 2023 or later, short ones", llm=llm)

    assert parsed.semantic_query == "AI papers"
    assert parsed.min_year == 2023
    assert parsed.max_year is None
    assert parsed.length_bucket == "short"


def test_parse_self_query_coerces_a_numeric_string_year():
    # Pydantic coerces "2023" (a string) to the integer 2023 -- this is the
    # exact real-world case this module exists to handle: an LLM does not
    # reliably distinguish JSON numbers from JSON strings.
    from self_query import SelfQueryFilters as _Filters

    parsed = _Filters(semantic_query="AI papers", min_year="2023")
    assert parsed.min_year == 2023
    assert isinstance(parsed.min_year, int)


def test_parse_self_query_recovers_json_embedded_in_prose(fake_llm):
    llm = fake_llm(response='Sure, here it is:\n{"semantic_query": "refund policy"}\nHope that helps.')
    parsed = parse_self_query("refund policy", llm=llm)
    assert parsed.semantic_query == "refund policy"


def test_parse_self_query_falls_back_to_the_raw_query_on_malformed_json(fake_llm):
    llm = fake_llm(response="I cannot produce JSON for this.")
    parsed = parse_self_query("original question", llm=llm)
    assert parsed == SelfQueryFilters(semantic_query="original question")


def test_parse_self_query_falls_back_when_a_field_has_the_wrong_shape(fake_llm):
    # "length_bucket" here is not one of the three allowed values -- a real
    # failure mode this module must not crash on.
    llm = fake_llm(
        response='{"semantic_query": "AI papers", "length_bucket": "extremely long"}'
    )
    parsed = parse_self_query("AI papers", llm=llm)
    assert parsed.semantic_query == "AI papers"
    assert parsed.length_bucket is None


def test_self_query_search_uses_the_semantic_part_for_retrieval(fake_llm):
    llm = fake_llm(response='{"semantic_query": "refund policy"}')
    retriever = FakeRetriever()

    self_query_search("what is the refund policy", retriever, metadata={}, llm=llm)

    assert retriever.calls == ["refund policy"]


def test_self_query_search_applies_the_extracted_year_filter(fake_llm):
    llm = fake_llm(response='{"semantic_query": "papers", "min_year": 2021}')
    retriever = FakeRetriever()
    metadata = {"a": {"year": 2019}, "b": {"year": 2021}, "c": {"year": 2022}, "d": {"year": 2023}}

    result = self_query_search("papers since 2021", retriever, metadata=metadata, llm=llm)

    assert [doc_id for doc_id, _ in result["results"]] == ["b", "c", "d"]
    assert result["parsed_filters"].min_year == 2021


def test_self_query_search_applies_the_extracted_length_filter(fake_llm):
    llm = fake_llm(response='{"semantic_query": "papers", "length_bucket": "short"}')
    retriever = FakeRetriever()
    metadata = {
        "a": {"length_bucket": "short"},
        "b": {"length_bucket": "long"},
        "c": {"length_bucket": "short"},
        "d": {"length_bucket": "medium"},
    }

    result = self_query_search("short papers", retriever, metadata=metadata, llm=llm)

    assert [doc_id for doc_id, _ in result["results"]] == ["a", "c"]


def test_self_query_search_applies_no_filter_when_none_were_extracted(fake_llm):
    llm = fake_llm(response='{"semantic_query": "anything"}')
    retriever = FakeRetriever()

    result = self_query_search("anything", retriever, metadata={}, top_k=4, llm=llm)

    assert [doc_id for doc_id, _ in result["results"]] == ["a", "b", "c", "d"]
