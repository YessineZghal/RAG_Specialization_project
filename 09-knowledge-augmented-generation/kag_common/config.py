"""Environment-driven configuration for Level 9 -- unique package name
(`kag_common`) per the naming lesson repeated at every prior level.
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

    # Sixth fresh open source: real biomedical research questions with real
    # yes/no/maybe answers, grounded in real PubMed abstracts -- a
    # professional domain, matching KAG's own published deployments
    # (Ant Group's E-Government and E-Health products). Not used by any
    # prior level.
    hf_dataset_name: str = os.getenv("HF_DATASET_NAME", "qiaojin/PubMedQA")
    hf_dataset_config: str = os.getenv("HF_DATASET_CONFIG", "pqa_labeled")
    # Kept modest on purpose: unlike Level 8's corpus build (no LLM calls
    # at all), every document here costs at least one real extraction
    # call -- and the real evaluation run builds *two* full graphs
    # (schema-constrained + unconstrained baseline) plus 2-3 answer calls
    # per question, so this number directly multiplies total LLM calls.
    n_documents: int = _get_int("KAG_N_DOCUMENTS", 25)
    dataset_seed: int = _get_int("KAG_DATASET_SEED", 42)

    # Paths
    level_dir: Path = LEVEL_DIR
    cache_dir: Path = LEVEL_DIR / "data" / "cache"


settings = Settings()
