"""Environment-driven configuration for Level 5 — same pattern as every
prior level, unique package name (`agentic_common`) per the lesson from
Level 3 (see 03-modular-rag/README.md#folder-structure).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

LEVEL_DIR = Path(__file__).resolve().parent.parent  # .../05-agentic-rag
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

    # Document/vector backend: TriviaQA -- open, real trivia questions with
    # full Wikipedia articles as source evidence AND real answer aliases
    # (multiple acceptable phrasings), which is what makes automatic
    # answer verification in verification/answer_verifier.py possible
    # without a heuristic.
    hf_dataset_name: str = os.getenv("HF_DATASET_NAME", "mandarjoshi/trivia_qa")
    hf_dataset_config: str = os.getenv("HF_DATASET_CONFIG", "rc.wikipedia")
    n_questions: int = _get_int("AGENTIC_N_QUESTIONS", 50)
    dataset_seed: int = _get_int("AGENTIC_DATASET_SEED", 42)
    chunk_size: int = _get_int("AGENTIC_CHUNK_SIZE", 200)
    chunk_overlap: int = _get_int("AGENTIC_CHUNK_OVERLAP", 20)

    # SQL backend: Northwind -- a different open sample database (business
    # orders/products/suppliers) from Level 3's Chinook (music store).
    northwind_url: str = os.getenv(
        "AGENTIC_NORTHWIND_URL",
        "https://raw.githubusercontent.com/jpwhite3/northwind-SQLite3/main/dist/northwind.db",
    )
    northwind_path: Path = LEVEL_DIR / "data" / "sql" / "northwind.sqlite"

    # Agent loop
    max_steps: int = _get_int("AGENTIC_MAX_STEPS", 5)

    # Paths
    level_dir: Path = LEVEL_DIR
    data_dir: Path = LEVEL_DIR / "data"
    cache_dir: Path = LEVEL_DIR / "data" / "cache"


settings = Settings()
