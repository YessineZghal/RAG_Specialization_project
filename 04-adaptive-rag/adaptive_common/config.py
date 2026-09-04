"""Environment-driven configuration for Level 4, following the same
pattern as Levels 1-3 (each level names its shared package uniquely —
`adaptive_common` here — after Level 3 hit a real `sys.modules` collision
between two same-named `common` packages; see 03-modular-rag/README.md).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

LEVEL_DIR = Path(__file__).resolve().parent.parent  # .../04-adaptive-rag
REPO_ROOT = LEVEL_DIR.parent

load_dotenv(REPO_ROOT / ".env")


def _get_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


@dataclass(frozen=True)
class Settings:
    # Ollama (same defaults as Levels 1-3)
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    ollama_embed_model: str = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    ollama_chat_model: str = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2")

    # Dataset: HotpotQA — a real, open multi-hop QA benchmark with genuine
    # ground truth (each question ships its exact supporting-fact
    # paragraphs), a different HF dataset and domain from every prior level.
    hf_dataset_name: str = os.getenv("HF_DATASET_NAME", "hotpotqa/hotpot_qa")
    hf_dataset_config: str = os.getenv("HF_DATASET_CONFIG", "distractor")
    n_questions: int = _get_int("ADAPTIVE_N_QUESTIONS", 200)
    dataset_seed: int = _get_int("ADAPTIVE_DATASET_SEED", 42)

    # Paths
    level_dir: Path = LEVEL_DIR
    data_dir: Path = LEVEL_DIR / "data"
    cache_dir: Path = LEVEL_DIR / "data" / "cache"


settings = Settings()
