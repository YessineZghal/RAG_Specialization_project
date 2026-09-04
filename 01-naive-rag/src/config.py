"""Central, environment-driven configuration for the Level 1 pipeline.

Every value has a sensible default so the pipeline runs with zero
configuration once Ollama is installed and running. Override anything by
copying `../../.env.example` to `../../.env` (or exporting the variable
directly) — see the repo-root `.env.example` for the full list.

Deliberately plain dataclasses + `os.getenv`, no pydantic: Level 1 is the
"naive" level, so the config layer stays as transparent as the pipeline
it configures.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

LEVEL_DIR = Path(__file__).resolve().parent.parent  # .../01-naive-rag
REPO_ROOT = LEVEL_DIR.parent
SHARED_DIR = REPO_ROOT / "shared"

# Load ../../.env (repo root) if present, without overriding variables
# already set in the real environment.
load_dotenv(REPO_ROOT / ".env")


def _get_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


@dataclass(frozen=True)
class Settings:
    # Ollama
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    ollama_embed_model: str = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    ollama_chat_model: str = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2")

    # Backends
    embedding_backend: str = os.getenv("EMBEDDING_BACKEND", "ollama")
    sentence_transformer_model: str = os.getenv(
        "SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2"
    )
    generation_backend: str = os.getenv("GENERATION_BACKEND", "ollama")
    vector_store_backend: str = os.getenv("VECTOR_STORE_BACKEND", "memory")

    # Qdrant
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "naive_rag_level1")

    # Chunking + retrieval
    chunk_size: int = _get_int("CHUNK_SIZE", 500)
    chunk_overlap: int = _get_int("CHUNK_OVERLAP", 50)
    top_k: int = _get_int("TOP_K", 3)

    # Dataset (open-source, fetched at runtime — see README.md#dataset)
    hf_dataset_name: str = os.getenv("HF_DATASET_NAME", "rag-datasets/rag-mini-wikipedia")
    hf_dataset_config: str = os.getenv("HF_DATASET_CONFIG", "text-corpus")
    hf_dataset_split: str = os.getenv("HF_DATASET_SPLIT", "passages")

    # Paths (relative to this level's folder)
    level_dir: Path = LEVEL_DIR
    data_dir: Path = LEVEL_DIR / "data"
    sample_docs_dir: Path = LEVEL_DIR / "data" / "sample_docs"
    index_dir: Path = LEVEL_DIR / "data" / "index"
    shared_dir: Path = SHARED_DIR
    shared_eval_dir: Path = SHARED_DIR / "evaluation"


settings = Settings()
