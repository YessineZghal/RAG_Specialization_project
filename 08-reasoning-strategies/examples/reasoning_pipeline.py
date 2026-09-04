#!/usr/bin/env python3
"""Answer one real StrategyQA-style question with a chosen reasoning
strategy, retrieving real evidence from the real pooled fact corpus
first.

Usage:
    cd 08-reasoning-strategies
    uv run python examples/reasoning_pipeline.py --strategy cot "Is a pizza box compostable?"
    uv run python examples/reasoning_pipeline.py --strategy tot "Is a pizza box compostable?"
    uv run python examples/reasoning_pipeline.py --strategy got "Is a pizza box compostable?"
    uv run python examples/reasoning_pipeline.py --strategy hgot "Is a pizza box compostable?"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_LEVEL_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_LEVEL_DIR))
sys.path.insert(0, str(_LEVEL_DIR / "chain-of-thought"))
sys.path.insert(0, str(_LEVEL_DIR / "tree-of-thought"))
sys.path.insert(0, str(_LEVEL_DIR / "graph-of-thought"))

from reasoning_common.dataset import prepare
from reasoning_common.embed import OllamaEmbedder
from reasoning_common.llm import OllamaLLM
from reasoning_common.retrieval import DenseRetriever
from cot_prompt import cot_answer
from tree_search import tree_of_thought_search
from graph_search import graph_of_thought_search
from hgot_retrieval import hgot_answer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?", default="Is a pizza box compostable?")
    parser.add_argument("--strategy", choices=["cot", "tot", "got", "hgot"], default="cot")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    print("Loading StrategyQA subset (cached after first run)...")
    data = prepare()
    print(f"Corpus: {len(data.corpus)} facts, {len(data.questions)} labeled questions available.\n")

    embedder = OllamaEmbedder()
    llm = OllamaLLM()
    retriever = DenseRetriever.from_corpus(data.corpus, embedder=embedder)

    print(f"Retrieving evidence for: {args.question!r}")
    results = retriever.search(args.question, top_k=args.top_k)
    context = "\n".join(data.corpus[doc_id] for doc_id, _score in results)
    print(f"Retrieved {len(results)} facts.\n")

    print(f"Running strategy: {args.strategy}\n")
    if args.strategy == "cot":
        result = cot_answer(args.question, context, llm=llm)
    elif args.strategy == "tot":
        result = tree_of_thought_search(args.question, context, llm=llm)
    elif args.strategy == "got":
        result = graph_of_thought_search(args.question, context, llm=llm)
    else:
        result = hgot_answer(args.question, retriever, data.corpus, llm=llm)

    print(f"Question: {args.question}")
    print(f"Answer:   {result['answer']}")
    print(f"LLM calls: {result['llm_calls']}")
    if "best_path" in result:
        print("\nReasoning path:")
        for step in result["best_path"]:
            print(f"  - {step}")
    if "sub_questions" in result:
        print("\nSub-questions investigated:")
        for sub_q in result["sub_questions"]:
            print(f"  - {sub_q}")


if __name__ == "__main__":
    main()
