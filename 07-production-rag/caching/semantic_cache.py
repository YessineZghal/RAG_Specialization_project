"""Semantic cache — unlike `response_cache.py`'s exact string match, this
catches *rephrased* duplicate queries ("What's the capital of France?" vs.
"capital of france") by comparing embeddings. Entries live in a single
Redis key as a JSON list (fine at the scale a cache actually needs to be
useful — a hot-query cache with millions of entries would want a real
vector index instead, e.g. Qdrant with a low top_k=1 threshold search).

**Why `namespace` exists**: same reason as `response_cache.py` -- once
personalization can make the same question generate a different answer
for different users, a shared entry list would surface one user's
personalized answer to another. Passing a `namespace` stores that user's
entries under their own Redis key, so similarity search only ever
compares against that same user's own history. Omitting it reproduces
the original, single shared cache for everyone.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from production_common.config import settings
from production_common.embed import OllamaEmbedder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "retrieval-infrastructure"))
from redis_store import RedisStore

CACHE_KEY = "semantic_cache:entries"
MAX_ENTRIES = 500


def _cache_key(namespace: str | None) -> str:
    return f"{CACHE_KEY}:{namespace}" if namespace else CACHE_KEY


class SemanticCache:
    def __init__(
        self,
        store: RedisStore | None = None,
        embedder: OllamaEmbedder | None = None,
        threshold: float | None = None,
    ) -> None:
        self.store = store or RedisStore()
        self.embedder = embedder or OllamaEmbedder()
        self.threshold = threshold if threshold is not None else settings.semantic_cache_threshold

    def _load_entries(self, namespace: str | None = None) -> list[dict]:
        raw = self.store.get(_cache_key(namespace))
        return json.loads(raw) if raw else []

    def _save_entries(self, entries: list[dict], namespace: str | None = None) -> None:
        self.store.set(_cache_key(namespace), json.dumps(entries[-MAX_ENTRIES:]))

    def get(self, query: str, namespace: str | None = None) -> dict | None:
        entries = self._load_entries(namespace)
        if not entries:
            return None

        query_vector = np.array(self.embedder.embed_one(query))
        query_norm = np.linalg.norm(query_vector) + 1e-12

        best_entry, best_score = None, -1.0
        for entry in entries:
            vector = np.array(entry["vector"])
            score = float(np.dot(vector, query_vector) / (np.linalg.norm(vector) * query_norm + 1e-12))
            if score > best_score:
                best_entry, best_score = entry, score

        if best_entry is not None and best_score >= self.threshold:
            return {"answer": best_entry["answer"], "matched_query": best_entry["query"], "similarity": best_score}
        return None

    def set(self, query: str, answer: dict, namespace: str | None = None) -> None:
        vector = self.embedder.embed_one(query)
        entries = self._load_entries(namespace)
        entries.append({"query": query, "answer": answer, "vector": vector})
        self._save_entries(entries, namespace)
