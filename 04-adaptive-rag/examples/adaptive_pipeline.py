#!/usr/bin/env python3
"""Full adaptive pipeline: classify a question, run the matching retrieval
strategy, apply CRAG grading, and fall back if evidence is weak — on the
real, open HotpotQA dataset.

Usage:
    cd 04-adaptive-rag
    uv run python examples/adaptive_pipeline.py "hi there!"
    uv run python examples/adaptive_pipeline.py "What is the capital of France?"
    uv run python examples/adaptive_pipeline.py "Peter Hobbs founded the company that is based in what town in Manchester?"
"""

from __future__ import annotations

import sys
from pathlib import Path

LEVEL_DIR = Path(__file__).resolve().parent.parent
for sub in ["", "query-classification", "dynamic-retrieval", "corrective-rag", "fallback-strategies"]:
    sys.path.insert(0, str(LEVEL_DIR / sub) if sub else str(LEVEL_DIR))

from adaptive_common.dataset import prepare
from adaptive_common.llm import OllamaLLM
from adaptive_common.retrieval import DenseRetriever
from classifier import classify_ensemble
from crag import corrective_retrieve
from retry import retrieve_with_retry


def main() -> None:
    question = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "Peter Hobbs founded the company that is based in what town in Manchester?"
    )
    llm = OllamaLLM()

    print("Loading HotpotQA subset (cached after first run)...")
    data = prepare()
    print(f"Corpus: {len(data.corpus)} paragraphs · {len(data.questions)} labeled questions available.\n")

    retriever = DenseRetriever.from_corpus(data.corpus)

    complexity = classify_ensemble(question, llm=llm)
    print(f"Q: {question}")
    print(f"Classified as: {complexity}\n")

    if complexity == "none":
        print("A:", llm.complete(question))
        return

    graded = corrective_retrieve(question, retriever, data.corpus, llm=llm, top_k=5)
    print(f"CRAG confidence: {graded['confidence']:.2f} (trustworthy={graded['trustworthy']})")

    if graded["trustworthy"]:
        context = "\n\n".join(
            data.corpus[g["doc_id"]] for g in graded["graded_results"] if g["grade"] == "relevant"
        )
        answer = llm.complete(f"Context:\n{context}\n\nQuestion: {question}\nAnswer:")
        print("\nA:", answer)
    else:
        print("Evidence too weak -- retrying with rewrite, then falling back to the web if needed.")
        outcome = retrieve_with_retry(question, retriever, data.corpus, llm=llm, top_k=5)
        print(f"Resolved via: {outcome['source']}")
        if outcome["source"] == "web_fallback":
            answer = llm.complete(f"Context:\n{outcome['result']['context']}\n\nQuestion: {question}\nAnswer:")
        else:
            context = "\n\n".join(
                data.corpus[g["doc_id"]]
                for g in outcome["result"]["graded_results"]
                if g["grade"] == "relevant"
            )
            answer = llm.complete(f"Context:\n{context}\n\nQuestion: {question}\nAnswer:")
        print("\nA:", answer)


if __name__ == "__main__":
    main()
