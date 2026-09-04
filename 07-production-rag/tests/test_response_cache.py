"""caching/response_cache.py -- exact-match, SHA256-keyed, Redis-backed."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from caching.response_cache import ResponseCache


def test_miss_on_empty_cache(fake_redis):
    cache = ResponseCache(store=fake_redis())
    assert cache.get("What is RAG?") is None


def test_set_then_get_is_a_hit(fake_redis):
    cache = ResponseCache(store=fake_redis())
    cache.set("What is RAG?", {"answer": "Retrieval-Augmented Generation"})
    assert cache.get("What is RAG?") == {"answer": "Retrieval-Augmented Generation"}


def test_key_is_case_and_whitespace_insensitive(fake_redis):
    cache = ResponseCache(store=fake_redis())
    cache.set("What is RAG?", {"answer": "..."})
    assert cache.get("  what is rag?  ") is not None


def test_different_queries_do_not_collide(fake_redis):
    cache = ResponseCache(store=fake_redis())
    cache.set("query one", {"answer": "one"})
    assert cache.get("query two") is None


def test_same_query_different_namespaces_do_not_collide(fake_redis):
    # Real motivation: personalization can make the same question generate
    # a different, correct answer per user -- their cache entries must not
    # overwrite or leak into each other.
    cache = ResponseCache(store=fake_redis())
    cache.set("favorite topics", {"answer": "sports"}, namespace="alice")
    cache.set("favorite topics", {"answer": "science"}, namespace="bob")

    assert cache.get("favorite topics", namespace="alice") == {"answer": "sports"}
    assert cache.get("favorite topics", namespace="bob") == {"answer": "science"}


def test_namespaced_entry_is_invisible_without_the_namespace(fake_redis):
    cache = ResponseCache(store=fake_redis())
    cache.set("favorite topics", {"answer": "sports"}, namespace="alice")
    assert cache.get("favorite topics") is None  # no namespace -> the shared, un-namespaced cache


def test_omitting_namespace_matches_prior_behavior_exactly(fake_redis):
    cache = ResponseCache(store=fake_redis())
    cache.set("What is RAG?", {"answer": "..."})
    assert cache.get("What is RAG?", namespace=None) is not None
