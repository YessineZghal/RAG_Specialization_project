#!/usr/bin/env python3
"""Same pipeline as simple_rag.py, backed by Qdrant instead of in-memory numpy.

Swapping the vector store is a one-line change (`vector_store=...`) because
`InMemoryVectorStore` and `QdrantVectorStore` share the same interface —
see src/retrieve.py.

Requirements:
    docker compose up -d qdrant        # from the repo root
    uv sync --extra qdrant
    ollama serve
    ollama pull nomic-embed-text
    ollama pull llama3.2

Usage:
    cd 01-naive-rag
    uv run python examples/rag_with_qdrant.py "What is the refund period?"
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings  # noqa: E402
from src.ingest import load_from_directory  # noqa: E402
from src.pipeline import RAGPipeline  # noqa: E402
from src.retrieve import QdrantVectorStore  # noqa: E402


def main() -> None:
    question = sys.argv[1] if len(sys.argv) > 1 else "What is the refund period?"

    documents = load_from_directory(settings.sample_docs_dir)
    print(f"Loaded {len(documents)} documents from {settings.sample_docs_dir}")

    pipeline = RAGPipeline(vector_store=QdrantVectorStore())
    n_chunks = pipeline.build_index(documents)
    print(f"Indexed {n_chunks} chunks into Qdrant collection '{settings.qdrant_collection}'.\n")

    answer = pipeline.ask(question)

    print(f"Q: {answer.question}")
    print(f"A: {answer.answer}\n")
    print("Sources:")
    for i, retrieved in enumerate(answer.sources, start=1):
        print(f"  [{i}] score={retrieved.score:.3f} doc={retrieved.chunk.document_id}")


if __name__ == "__main__":
    main()
