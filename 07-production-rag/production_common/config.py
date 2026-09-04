"""Environment-driven configuration for Level 7 — unique package name
(`production_common`) per the naming lesson from Levels 3-6.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

LEVEL_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = LEVEL_DIR.parent

load_dotenv(REPO_ROOT / ".env")


def _get_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


@dataclass(frozen=True)
class Settings:
    # Ollama (same defaults as every prior level)
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    ollama_embed_model: str = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    ollama_chat_model: str = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2")

    # Document backend: SQuAD -- a fresh, widely-known open QA dataset
    # (Wikipedia paragraphs + real ground-truth answers), used here as the
    # underlying corpus a production API actually serves.
    hf_dataset_name: str = os.getenv("HF_DATASET_NAME", "rajpurkar/squad")
    n_contexts: int = _get_int("PROD_N_CONTEXTS", 300)
    dataset_seed: int = _get_int("PROD_DATASET_SEED", 42)

    # Isolated ports for this level's own docker-compose stack -- deliberately
    # different from any other locally-running Qdrant/Postgres/Redis
    # instance (this machine has unrelated ones already running).
    qdrant_url: str = os.getenv("PROD_QDRANT_URL", "http://localhost:16333")
    qdrant_collection: str = os.getenv("PROD_QDRANT_COLLECTION", "production_rag")
    postgres_dsn: str = os.getenv(
        "PROD_POSTGRES_DSN", "postgresql://postgres:postgres@localhost:15432/production_rag"
    )
    redis_url: str = os.getenv("PROD_REDIS_URL", "redis://localhost:16379/0")

    # API
    api_key: str = os.getenv("PROD_API_KEY", "dev-local-key")
    # A separate key for the admin-only /admin/ingest route (security/permissions.py's
    # "ingest" action) -- kept distinct from api_key so a regular caller's
    # key can never perform an admin action, not even by accident.
    admin_api_key: str = os.getenv("PROD_ADMIN_API_KEY", "dev-admin-key")
    top_k: int = _get_int("PROD_TOP_K", 5)
    # Measured, not guessed: real paraphrase pairs scored ~0.95 cosine
    # similarity with nomic-embed-text, unrelated queries ~0.39 -- a huge
    # gap. The original default of 0.97 was picked without measuring and
    # was strict enough to miss genuine paraphrases (see README.md#caching).
    semantic_cache_threshold: float = float(os.getenv("PROD_SEMANTIC_CACHE_THRESHOLD", "0.92"))

    # Paths
    level_dir: Path = LEVEL_DIR
    cache_dir: Path = LEVEL_DIR / "data" / "cache"


settings = Settings()
