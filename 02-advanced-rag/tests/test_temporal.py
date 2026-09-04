from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "metadata-filtering"))
from temporal import DateRange, build_temporal_metadata, filter_by_date_range, temporal_search


def test_build_temporal_metadata_is_deterministic_for_a_fixed_seed():
    corpus = {"doc-a": {"text": "..."}, "doc-b": {"text": "..."}}
    first = build_temporal_metadata(corpus, seed=42)
    second = build_temporal_metadata(corpus, seed=42)
    assert first == second


def test_build_temporal_metadata_assigns_years_within_range():
    corpus = {f"doc-{i}": {"text": "..."} for i in range(20)}
    metadata = build_temporal_metadata(corpus, min_year=2018, max_year=2020, seed=1)
    years = {entry["year"] for entry in metadata.values()}
    assert years <= {2018, 2019, 2020}


def test_date_range_as_of_keeps_only_documents_on_or_before_that_year():
    date_range = DateRange.as_of(2020)
    assert date_range.contains(2018) is True
    assert date_range.contains(2020) is True
    assert date_range.contains(2021) is False


def test_date_range_supports_both_bounds():
    date_range = DateRange(after=2019, before=2021)
    assert date_range.contains(2018) is False
    assert date_range.contains(2019) is True
    assert date_range.contains(2020) is True
    assert date_range.contains(2021) is True
    assert date_range.contains(2022) is False


def test_filter_by_date_range_drops_documents_outside_the_range():
    candidates = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
    metadata = {"a": {"year": 2015}, "b": {"year": 2020}, "c": {"year": 2023}}

    filtered = filter_by_date_range(candidates, metadata, DateRange(after=2018, before=2021))

    assert [doc_id for doc_id, _ in filtered] == ["b"]


def test_filter_by_date_range_drops_documents_with_no_year_metadata():
    candidates = [("a", 0.9), ("b", 0.8)]
    metadata = {"a": {"year": 2020}}  # "b" has no metadata entry at all

    filtered = filter_by_date_range(candidates, metadata, DateRange())

    assert [doc_id for doc_id, _ in filtered] == ["a"]


def test_temporal_search_retrieves_wide_then_filters_then_truncates():
    class FakeRetriever:
        def search(self, query, top_k):
            return [("a", 0.9), ("b", 0.8), ("c", 0.7), ("d", 0.6)][:top_k]

    metadata = {"a": {"year": 2010}, "b": {"year": 2020}, "c": {"year": 2021}, "d": {"year": 2022}}

    # after=2015 excludes "a" (2010); before=2021 excludes "d" (2022) --
    # leaves "b" and "c" as the only two in range, in retrieval order.
    results = temporal_search(
        FakeRetriever(), "query", metadata, DateRange(after=2015, before=2021), top_k=2, candidate_k=4
    )

    assert [doc_id for doc_id, _ in results] == ["b", "c"]
