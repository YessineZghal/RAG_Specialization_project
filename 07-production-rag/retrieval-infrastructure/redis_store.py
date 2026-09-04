"""Redis client — shared, low-latency state: response/semantic caching
(`caching/`) and, in a multi-worker deployment, anything else that needs
to be visible across processes instead of living in one worker's memory.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from production_common.config import settings


class RedisStore:
    def __init__(self, url: str | None = None) -> None:
        import redis

        self.client = redis.Redis.from_url(url or settings.redis_url, decode_responses=True)

    def ping(self) -> bool:
        return self.client.ping()

    def get(self, key: str) -> str | None:
        return self.client.get(key)

    def set(self, key: str, value: str, ttl_seconds: int | None = None) -> None:
        self.client.set(key, value, ex=ttl_seconds)

    def delete(self, key: str) -> None:
        self.client.delete(key)

    def scan_keys(self, pattern: str) -> list[str]:
        return list(self.client.scan_iter(match=pattern))
