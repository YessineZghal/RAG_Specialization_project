"""caching/semantic_cache.py -- cosine-similarity match above a threshold.

Uses hand-built vectors (via the `fake_embedder` fixture) so the threshold
logic is tested exactly, without depending on a real embedding model's
actual similarity scores -- those are measured separately and documented
in README.md#caching (paraphrases ~0.95, unrelated ~0.39 with nomic-embed-text).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from caching.semantic_cache import SemanticCache  # noqa: E402

_VECTORS = {
    "capital of france": [1.0, 0.0, 0.0],
    # cos([1,0,0], [0.95, 0.3122, 0]) == 0.95 exactly -- matching this
    # level's real *measured* nomic-embed-text paraphrase similarity
    # (~0.95, see README.md#caching), not an arbitrarily-close vector.
    "what is the capital of france?": [0.95, 0.3122, 0.0],
    "weather in paris today": [0.0, 1.0, 0.0],  # orthogonal -> unrelated
}


def _cache(fake_embedder, fake_redis, threshold: float = 0.9) -> SemanticCache:
    return SemanticCache(store=fake_redis(), embedder=fake_embedder(_VECTORS), threshold=threshold)


def test_miss_on_empty_cache(fake_embedder, fake_redis):
    cache = _cache(fake_embedder, fake_redis)
    assert cache.get("capital of france") is None


def test_near_duplicate_phrasing_hits_above_threshold(fake_embedder, fake_redis):
    cache = _cache(fake_embedder, fake_redis, threshold=0.9)
    cache.set("capital of france", {"answer": "Paris"})
    hit = cache.get("what is the capital of france?")
    assert hit is not None
    assert hit["answer"] == {"answer": "Paris"}
    assert hit["similarity"] > 0.9


def test_unrelated_query_does_not_hit_even_with_entries_present(fake_embedder, fake_redis):
    cache = _cache(fake_embedder, fake_redis, threshold=0.9)
    cache.set("capital of france", {"answer": "Paris"})
    assert cache.get("weather in paris today") is None


def test_threshold_is_configurable_and_enforced(fake_embedder, fake_redis):
    # Same near-duplicate pair, but with an unreachably strict threshold --
    # this is the exact miscalibration this level's config.py documents
    # fixing (0.97 was too strict for a real ~0.95 paraphrase score).
    cache = _cache(fake_embedder, fake_redis, threshold=0.999)
    cache.set("capital of france", {"answer": "Paris"})
    assert cache.get("what is the capital of france?") is None


def test_namespaces_keep_separate_entry_lists(fake_embedder, fake_redis):
    # Real motivation: personalization can make the same question generate
    # a different, correct answer per user -- their semantic-cache entries
    # must not be compared against each other at all.
    cache = _cache(fake_embedder, fake_redis, threshold=0.9)
    cache.set("capital of france", {"answer": "alice's answer"}, namespace="alice")

    assert cache.get("what is the capital of france?", namespace="alice") is not None
    assert cache.get("what is the capital of france?", namespace="bob") is None
    assert cache.get("what is the capital of france?") is None  # un-namespaced cache is separate too
