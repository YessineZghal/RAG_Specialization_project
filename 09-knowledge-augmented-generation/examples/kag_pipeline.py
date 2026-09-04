#!/usr/bin/env python3
"""Answer one question against a schema-constrained knowledge graph built
from a real, cached sample of PubMed abstracts.

Usage:
    cd 09-knowledge-augmented-generation
    uv run python examples/kag_pipeline.py "Is anesthetic choice associated with cancer recurrence?"
    uv run python examples/kag_pipeline.py --rebuild-graph "..."

Real PubMedQA questions are all yes/no/maybe about a study's own
findings -- they never ask about population thresholds or "the largest
study", so this pipeline also accepts the two hand-authored example
questions from the README that exercise the numerical/KG-reasoning
operators directly, the same precedent 04-adaptive-rag set for question
types a real dataset doesn't naturally contain:

    uv run python examples/kag_pipeline.py \\
        "Was the intervention studied in a population larger than 500 patients?"
    uv run python examples/kag_pipeline.py \\
        "What outcome was reported for the largest study of this condition?"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_LEVEL_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_LEVEL_DIR))
sys.path.insert(0, str(_LEVEL_DIR / "reasoning-engine"))

from kag_common.config import settings
from kag_common.dataset import prepare
from kag_common.embed import OllamaEmbedder, embed_texts
from kag_common.llm import OllamaLLM
from indexing.graph_builder import build_graph, load_graph, save_graph
from operator_router import answer_question

GRAPH_CACHE = settings.cache_dir / "kag_graph.json"


def get_graph(data, llm, rebuild: bool = False):
    if GRAPH_CACHE.exists() and not rebuild:
        print(f"Loading cached graph from {GRAPH_CACHE} (pass --rebuild-graph to force a fresh extraction)...")
        return load_graph(GRAPH_CACHE)

    print(f"Extracting schema-constrained graph from {len(data.corpus)} real PubMed abstracts...")
    graph, validator, mutual_index = build_graph(data.corpus, llm)
    print(f"Schema validator: {validator.summary()}")
    save_graph(graph, mutual_index, GRAPH_CACHE)
    return graph, mutual_index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question")
    parser.add_argument("--rebuild-graph", action="store_true")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    data = prepare()
    llm = OllamaLLM()
    embedder = OllamaEmbedder()

    graph, mutual_index = get_graph(data, llm, rebuild=args.rebuild_graph)
    doc_ids, matrix = embed_texts(data.corpus, embedder, cache_name="kag_corpus")

    answer = answer_question(
        args.question, data.corpus, doc_ids, matrix, graph, mutual_index,
        embedder=embedder, llm=llm, top_k=args.top_k,
    )

    print(f"\nQuestion: {args.question}")
    print(f"Operators used: {answer.operators_used}")
    print(f"Verdict: {answer.verdict}")
    print(f"Citations (source doc/pubids): {sorted(answer.citations)}")
    if answer.numeric_result is not None:
        print(f"Numeric result: {answer.numeric_result}")
    print(f"\nFull reasoning:\n{answer.raw_response}")


if __name__ == "__main__":
    main()
