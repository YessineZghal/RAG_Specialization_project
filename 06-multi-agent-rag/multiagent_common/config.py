"""Environment-driven configuration for Level 6 — unique package name
(`multiagent_common`) per the naming lesson from Levels 3-5 (see
03-modular-rag/README.md#folder-structure): same-named shared packages or
top-level modules across levels collide in `sys.modules` during a
combined test run, regardless of `sys.path` order.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

LEVEL_DIR = Path(__file__).resolve().parent.parent  # .../06-multi-agent-rag
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

    # Document backend: real questions over real SEC 10-K filings, across
    # 69 real public companies -- a fresh open dataset and a genuinely
    # "business research" domain, unlike any prior level.
    hf_dataset_name: str = os.getenv("HF_DATASET_NAME", "virattt/financial-qa-10K")
    n_questions: int = _get_int("MULTIAGENT_N_QUESTIONS", 200)
    dataset_seed: int = _get_int("MULTIAGENT_DATASET_SEED", 42)

    # SQL backend: Sakila (DVD rental store) -- a third distinct SQL
    # domain in this repo, after Chinook (music, Level 3) and Northwind
    # (trade, Level 5).
    sakila_url: str = os.getenv(
        "MULTIAGENT_SAKILA_URL",
        "https://github.com/bradleygrant/sakila-sqlite3/raw/main/sakila_master.db",
    )
    sakila_path: Path = LEVEL_DIR / "data" / "sql" / "sakila.sqlite"

    max_steps: int = _get_int("MULTIAGENT_MAX_STEPS", 4)

    # Paths
    level_dir: Path = LEVEL_DIR
    data_dir: Path = LEVEL_DIR / "data"
    cache_dir: Path = LEVEL_DIR / "data" / "cache"


settings = Settings()
