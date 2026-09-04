#!/usr/bin/env python3
"""Minimal end-to-end naive RAG: in-memory store, Ollama for embeddings + generation.

This is the smallest complete loop in the repo — read it top to bottom to
see every stage of Level 1 with nothing hidden.

Requirements:
    ollama serve                       # in another terminal
    ollama pull nomic-embed-text
    ollama pull llama3.2

Usage:
    cd 01-naive-rag
    uv run python examples/simple_rag.py "What is the refund period?"
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # make `src` importable

from src.ingest import load_from_directory
from src.pipeline import RAGPipeline
from src.config import settings


def main() -> None:
    question = sys.argv[1] if len(sys.argv) > 1 else "What is the refund period?"

    # 1-6: load -> chunk -> embed -> store, all inside build_index().
    # Uses the small, hand-written docs in data/sample_docs/ so this example
    # runs fully offline (no dataset download) — swap in
    # `src.ingest.load_from_hf_dataset()` for the full open Wikipedia corpus.
    documents = load_from_directory(settings.sample_docs_dir)
    print(f"Loaded {len(documents)} documents from {settings.sample_docs_dir}")

    pipeline = RAGPipeline()  # defaults: Ollama embeddings, Ollama generation, in-memory store
    n_chunks = pipeline.build_index(documents)
    print(f"Indexed {n_chunks} chunks.\n")

    # 7-9: retrieve -> prompt -> generate.
    answer = pipeline.ask(question)

    print(f"Q: {answer.question}")
    print(f"A: {answer.answer}\n")
    print("Sources:")
    for i, retrieved in enumerate(answer.sources, start=1):
        print(f"  [{i}] score={retrieved.score:.3f} doc={retrieved.chunk.document_id}")
        print(f"      {retrieved.chunk.text[:120]}...")


if __name__ == "__main__":
    main()
