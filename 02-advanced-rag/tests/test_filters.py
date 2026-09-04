from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "metadata-filtering"))
from filters import build_length_metadata, filter_by_metadata, filtered_search


def test_build_length_metadata_buckets_by_word_count():
    corpus = {
        "short-doc": {"text": "a b c"},
        "long-doc": {"text": " ".join("w" for _ in range(300))},
    }
    metadata = build_length_metadata(corpus)
    assert metadata["short-doc"]["length_bucket"] == "short"
    assert metadata["long-doc"]["length_bucket"] == "long"


def test_filter_by_metadata_drops_non_matching_docs():
    candidates = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
    metadata = {"a": {"bucket": "keep"}, "b": {"bucket": "drop"}, "c": {"bucket": "keep"}}

    filtered = filter_by_metadata(candidates, metadata, lambda m: m.get("bucket") == "keep")

    assert [doc_id for doc_id, _ in filtered] == ["a", "c"]


def test_filtered_search_retrieves_wide_then_filters_then_truncates():
    class FakeRetriever:
        def search(self, query, top_k):
            return [("a", 0.9), ("b", 0.8), ("c", 0.7), ("d", 0.6)][:top_k]

    metadata = {"a": {"ok": False}, "b": {"ok": True}, "c": {"ok": True}, "d": {"ok": True}}

    results = filtered_search(
        FakeRetriever(), "query", metadata, lambda m: m["ok"], top_k=2, candidate_k=4
    )

    assert [doc_id for doc_id, _ in results] == ["b", "c"]
