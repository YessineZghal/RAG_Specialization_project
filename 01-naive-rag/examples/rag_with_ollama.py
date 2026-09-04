#!/usr/bin/env python3
"""Same pipeline, but on the full open-source Wikipedia dataset, mixing backends:
sentence-transformers for embeddings, Ollama for generation.

This shows that `Embedder` and `Generator` are independent, swappable
pieces (src/embed.py, src/generate.py) — you are not locked into one
vendor/tool for the whole pipeline.

Requirements:
    uv sync --extra sentence-transformers
    ollama serve
    ollama pull llama3.2

Usage:
    cd 01-naive-rag
    uv run python examples/rag_with_ollama.py "Who was Ada Lovelace?"
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.embed import SentenceTransformerEmbedder  # noqa: E402
from src.generate import OllamaGenerator  # noqa: E402
from src.ingest import load_from_hf_dataset  # noqa: E402
from src.pipeline import RAGPipeline  # noqa: E402


def main() -> None:
    question = sys.argv[1] if len(sys.argv) > 1 else "Who was Ada Lovelace?"

    # This is the one script in Level 1 that pulls the full open dataset —
    # it is only downloaded the first time you run it, and only because you
    # ran it (see src/ingest.py:load_from_hf_dataset).
    documents = load_from_hf_dataset(limit=500)
    print(f"Loaded {len(documents)} documents from {documents[0].metadata['source']}")

    pipeline = RAGPipeline(
        embedder=SentenceTransformerEmbedder(),
        generator=OllamaGenerator(),
    )
    n_chunks = pipeline.build_index(documents)
    print(f"Indexed {n_chunks} chunks.\n")

    answer = pipeline.ask(question)

    print(f"Q: {answer.question}")
    print(f"A: {answer.answer}\n")
    print("Sources:", answer.source_document_ids())


if __name__ == "__main__":
    main()
