#!/usr/bin/env python3
"""This level's mini project: an end-to-end example that picks a
reasoning strategy per question type, rather than the caller choosing
one up front the way `examples/reasoning_pipeline.py` requires.

Classifies the question (`reasoning_common/strategy_classifier.py`) into
simple / comparative / combinatorial / multi_hop, maps that to
cot / tot / got / hgot, and runs the real strategy end to end -- the
same retrieval-then-reason pipeline `reasoning_pipeline.py` uses, just
with the strategy chosen automatically instead of passed with `--strategy`.

Usage:
    cd 08-reasoning-strategies
    uv run python examples/strategy_selector.py "Is a pizza box compostable?"
    uv run python examples/strategy_selector.py "Would a python starve on a diet of one mouse a week?"

This level's own real evaluation found Chain-of-Thought the strongest,
cheapest strategy on real StrategyQA questions
(`notebooks/04_reasoning_vs_plain_rag_eval.ipynb`) -- so this selector is
not expected to *beat* always-CoT on accuracy. Its point is different:
demonstrating a real, tested per-question-type dispatch mechanism, the
same kind of routing decision Levels 3/4/9 all measure rather than
assume correct, not a claim that dynamic selection outperforms the
simplest strategy on this dataset.
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
from reasoning_common.strategy_classifier import classify_strategy
from cot_prompt import cot_answer
from tree_search import tree_of_thought_search
from graph_search import graph_of_thought_search
from hgot_retrieval import hgot_answer


def run_auto(
    question: str,
    corpus: dict[str, str],
    retriever: DenseRetriever,
    llm: OllamaLLM,
    top_k: int = 5,
) -> dict:
    """Classify `question`, dispatch to the matching strategy, and
    return its result dict with `"strategy"` added -- one extra LLM call
    (the classifier itself) on top of whichever strategy gets picked."""
    strategy = classify_strategy(question, llm=llm)

    if strategy == "hgot":
        # HGoT does its own per-sub-question retrieval internally --
        # unlike the other three, it never wants one shared context blob
        result = hgot_answer(question, retriever, corpus, llm=llm)
    else:
        hits = retriever.search(question, top_k=top_k)
        context = "\n".join(corpus[doc_id] for doc_id, _score in hits)
        if strategy == "cot":
            result = cot_answer(question, context, llm=llm)
        elif strategy == "tot":
            result = tree_of_thought_search(question, context, llm=llm)
        else:  # got
            result = graph_of_thought_search(question, context, llm=llm)

    result["strategy"] = strategy
    result["llm_calls"] = result["llm_calls"] + 1  # + the classification call itself
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question", nargs="?", default="Is a pizza box compostable?")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    print("Loading StrategyQA subset (cached after first run)...")
    data = prepare()
    print(f"Corpus: {len(data.corpus)} facts, {len(data.questions)} labeled questions available.\n")

    embedder = OllamaEmbedder()
    llm = OllamaLLM()
    retriever = DenseRetriever.from_corpus(data.corpus, embedder=embedder)

    result = run_auto(args.question, data.corpus, retriever, llm, top_k=args.top_k)

    print(f"Question: {args.question}")
    print(f"Selected strategy: {result['strategy']}")
    print(f"Answer:   {result['answer']}")
    print(f"LLM calls: {result['llm_calls']} (including the classification call)")
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
