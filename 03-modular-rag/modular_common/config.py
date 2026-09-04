"""Environment-driven configuration for Level 3, mirroring the pattern in
Level 1 (`01-naive-rag/src/config.py`) and Level 2 (`02-advanced-rag/common/config.py`).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

LEVEL_DIR = Path(__file__).resolve().parent.parent  # .../03-modular-rag
REPO_ROOT = LEVEL_DIR.parent

load_dotenv(REPO_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    # Ollama (same defaults as Levels 1-2)
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    ollama_embed_model: str = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    ollama_chat_model: str = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2")
    # A vision-language model, used only by multimodal-rag/vision_embedding.py
    # to describe an image's actual visual content (see that module's
    # docstring for why this is a two-step approximation, not true visual
    # embedding). Not required for anything else in this level.
    ollama_vision_model: str = os.getenv("OLLAMA_VISION_MODEL", "moondream")

    # Document backend: a real, open-access PDF (Google grants reproduction
    # rights for its tables/figures "for use in journalistic or scholarly
    # works" — see page 1). Downloaded on first use, never bundled.
    pdf_url: str = os.getenv(
        "MODULAR_RAG_PDF_URL", "https://arxiv.org/pdf/1706.03762"
    )
    pdf_path: Path = LEVEL_DIR / "data" / "pdfs" / "attention_is_all_you_need.pdf"

    # SQL backend: the open-source (MIT-licensed) Chinook sample database.
    chinook_url: str = os.getenv(
        "MODULAR_RAG_CHINOOK_URL",
        "https://raw.githubusercontent.com/lerocha/chinook-database/master/"
        "ChinookDatabase/DataSources/Chinook_Sqlite.sqlite",
    )
    chinook_path: Path = LEVEL_DIR / "data" / "sql" / "chinook.sqlite"

    # Paths
    level_dir: Path = LEVEL_DIR
    data_dir: Path = LEVEL_DIR / "data"
    images_dir: Path = LEVEL_DIR / "data" / "pdfs" / "images"
    cache_dir: Path = LEVEL_DIR / "data" / "cache"


settings = Settings()
