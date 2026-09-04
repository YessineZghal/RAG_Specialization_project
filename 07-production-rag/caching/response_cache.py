"""Response cache — exact-match: the same query string returns the cached
answer instantly, skipping retrieval and generation entirely. Redis-backed
so it's shared across every worker process, not just the one that first
answered it.

**Why `namespace` exists**: once `security/personalization.py` can make
the same question rank sources differently (and so, downstream, generate
a different answer) for different users, caching by question text alone
would leak one user's personalized answer to a different user who happens
to type the same question. Passing a `namespace` (in practice, the
requesting user's id) keeps each user's cache entries separate; omitting
it (the default) reproduces the exact pre-personalization behavior, one
shared cache for everyone, unchanged.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from production_common.config import settings

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "retrieval-infrastructure"))
from redis_store import RedisStore

DEFAULT_TTL_SECONDS = 3600


class ResponseCache:
    def __init__(self, store: RedisStore | None = None, ttl_seconds: int = DEFAULT_TTL_SECONDS) -> None:
        self.store = store or RedisStore()
        self.ttl_seconds = ttl_seconds

    @staticmethod
    def _key(query: str, namespace: str | None = None) -> str:
        normalized = query.strip().lower()
        if namespace:
            normalized = f"{namespace}:{normalized}"
        digest = hashlib.sha256(normalized.encode()).hexdigest()
        return f"response_cache:{digest}"

    def get(self, query: str, namespace: str | None = None) -> dict | None:
        cached = self.store.get(self._key(query, namespace))
        return json.loads(cached) if cached else None

    def set(self, query: str, answer: dict, namespace: str | None = None) -> None:
        self.store.set(self._key(query, namespace), json.dumps(answer), ttl_seconds=self.ttl_seconds)
