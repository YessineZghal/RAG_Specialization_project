"""Environment-driven configuration for Level 8 -- unique package name
(`reasoning_common`) per the naming lesson repeated at every prior level
(most recently restated in Level 7's README): a same-named shared package
across levels silently collides once every level's directory sits on
`sys.path` at once, in a combined `pytest` run.
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

    # Primary dataset: StrategyQA -- real yes/no questions whose answers
    # require an *implicit* multi-step reasoning strategy the question
    # itself never states, grounded in real supporting facts. Not used by
    # any prior level.
    hf_dataset_name: str = os.getenv("HF_DATASET_NAME", "ChilleD/StrategyQA")
    n_questions: int = _get_int("REASONING_N_QUESTIONS", 120)
    dataset_seed: int = _get_int("REASONING_DATASET_SEED", 42)

    # Calibration dataset (optional, no retrieval): GSM8K -- isolates
    # reasoning-strategy quality from retrieval quality.
    hf_calibration_dataset: str = os.getenv("HF_CALIBRATION_DATASET", "openai/gsm8k")

    # Paths
    level_dir: Path = LEVEL_DIR
    cache_dir: Path = LEVEL_DIR / "data" / "cache"


settings = Settings()
