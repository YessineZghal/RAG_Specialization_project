"""Environment-driven configuration for Level 2, mirroring Level 1's pattern
(01-naive-rag/src/config.py) — same env vars where they overlap, so a single
repo-root `.env` configures both levels.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

LEVEL_DIR = Path(__file__).resolve().parent.parent  # .../02-advanced-rag
REPO_ROOT = LEVEL_DIR.parent

load_dotenv(REPO_ROOT / ".env")


def _get_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


@dataclass(frozen=True)
class Settings:
    # Ollama (same defaults as Level 1)
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    ollama_embed_model: str = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    ollama_chat_model: str = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2")

    # Rerankers (sentence-transformers CrossEncoder — optional extra)
    cross_encoder_model: str = os.getenv(
        "CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    bge_reranker_model: str = os.getenv("BGE_RERANKER_MODEL", "BAAI/bge-reranker-base")

    # Dataset: BeIR/scifact — open-source IR benchmark with real relevance
    # judgments (qrels), so Level 2's metrics are measured against ground
    # truth rather than a heuristic (contrast with Level 1's best-effort
    # expected_sources.jsonl).
    hf_corpus_dataset: str = os.getenv("HF_CORPUS_DATASET", "BeIR/scifact")
    hf_qrels_dataset: str = os.getenv("HF_QRELS_DATASET", "BeIR/scifact-qrels")
    corpus_size: int = _get_int("CORPUS_SIZE", 1000)  # relevant docs + distractors
    corpus_seed: int = _get_int("CORPUS_SEED", 42)

    # Paths
    level_dir: Path = LEVEL_DIR
    data_dir: Path = LEVEL_DIR / "data"
    cache_dir: Path = LEVEL_DIR / "data" / "cache"  # gitignored, rebuildable


settings = Settings()
