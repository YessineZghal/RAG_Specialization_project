#!/usr/bin/env python3
"""Real evaluation: schema-constrained KAG vs. an unconstrained
graph-rag baseline, both built fresh from the same real sample of
PubMedQA abstracts and asked the same real yes/no/maybe questions.

Not a re-use of the KAG paper's own published 19.6%/33.5% figures --
those are measured on HotpotQA/2WikiMultiHopQA with a different backbone
model entirely. This runs both systems, end to end, on real data, with
whatever local Ollama model is configured, and reports what actually
happened.

Usage:
    cd 09-knowledge-augmented-generation
    uv run python kag_eval/kag_vs_graphrag_eval.py
    uv run python kag_eval/kag_vs_graphrag_eval.py --n-documents 25 --rebuild
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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

from kag_eval.metrics import evaluate
from kag_eval.simple_graphrag_baseline import baseline_answer, build_unconstrained_graph

RESULTS_FILE = Path(__file__).resolve().parent / "comparison_results.json"
KAG_GRAPH_CACHE = settings.cache_dir / "kag_graph_eval.json"
BASELINE_GRAPH_CACHE = settings.cache_dir / "baseline_graph_eval.json"


def _get_or_build_kag_graph(corpus: dict[str, str], llm: OllamaLLM, rebuild: bool):
    if KAG_GRAPH_CACHE.exists() and not rebuild:
        print(f"[KAG] loading cached graph from {KAG_GRAPH_CACHE}")
        graph, mutual_index = load_graph(KAG_GRAPH_CACHE)
        return graph, mutual_index, None
    print(f"[KAG] extracting schema-constrained graph from {len(corpus)} documents...")
    graph, validator, mutual_index = build_graph(corpus, llm)
    save_graph(graph, mutual_index, KAG_GRAPH_CACHE)
    return graph, mutual_index, validator


def _get_or_build_baseline_graph(corpus: dict[str, str], llm: OllamaLLM, rebuild: bool):
    from indexing.mutual_index import MutualIndex

    if BASELINE_GRAPH_CACHE.exists() and not rebuild:
        print(f"[baseline] loading cached graph from {BASELINE_GRAPH_CACHE}")
        graph, _ = load_graph(BASELINE_GRAPH_CACHE)
        return graph
    print(f"[baseline] extracting unconstrained graph from {len(corpus)} documents...")
    graph = build_unconstrained_graph(corpus, llm)
    save_graph(graph, MutualIndex(), BASELINE_GRAPH_CACHE)  # no real mutual index needed for the baseline -- it never does KG-provenance lookups
    return graph


def run_comparison(n_documents: int, seed: int, rebuild: bool) -> dict:
    data = prepare(n_documents=n_documents, seed=seed)
    llm = OllamaLLM()
    embedder = OllamaEmbedder()

    doc_ids, matrix = embed_texts(data.corpus, embedder, cache_name="kag_eval_corpus", force=rebuild)

    kag_graph, mutual_index, validator = _get_or_build_kag_graph(data.corpus, llm, rebuild)
    baseline_graph = _get_or_build_baseline_graph(data.corpus, llm, rebuild)

    gold = {qid: q["answer"] for qid, q in data.questions.items()}
    kag_predictions: dict[str, str | None] = {}
    baseline_predictions: dict[str, str | None] = {}
    kag_operator_counts: dict[str, int] = {}
    disagreements: list[dict] = []

    start = time.time()
    for i, (qid, q) in enumerate(data.questions.items(), 1):
        question = q["question"]
        print(f"  [{i}/{len(data.questions)}] {question[:80]}")

        kag_answer = answer_question(
            question, data.corpus, doc_ids, matrix, kag_graph, mutual_index,
            embedder=embedder, llm=llm,
        )
        kag_predictions[qid] = kag_answer.verdict
        for op in kag_answer.operators_used:
            kag_operator_counts[op] = kag_operator_counts.get(op, 0) + 1

        baseline = baseline_answer(question, data.corpus, doc_ids, matrix, baseline_graph, embedder=embedder, llm=llm)
        baseline_predictions[qid] = baseline.verdict

        if kag_answer.verdict != baseline.verdict:
            disagreements.append({
                "qid": qid, "question": question, "gold": q["answer"],
                "kag": kag_answer.verdict, "baseline": baseline.verdict,
            })

    elapsed = time.time() - start

    kag_eval = evaluate(kag_predictions, gold)
    baseline_eval = evaluate(baseline_predictions, gold)

    result = {
        "n_documents": n_documents,
        "n_questions": len(data.questions),
        "seed": seed,
        "elapsed_seconds": round(elapsed, 1),
        "gold_distribution": {label: sum(1 for v in gold.values() if v == label) for label in ("yes", "no", "maybe")},
        "kag": {
            "accuracy": kag_eval.accuracy,
            "n_correct": kag_eval.n_correct,
            "n_unparseable": kag_eval.n_unparseable,
            "per_label": kag_eval.per_label,
            "predicted_distribution": kag_eval.predicted_distribution,
            "operator_usage": kag_operator_counts,
            "schema_validator": validator.summary() if validator else "loaded from cache, not recomputed",
        },
        "baseline_unconstrained_graphrag": {
            "accuracy": baseline_eval.accuracy,
            "n_correct": baseline_eval.n_correct,
            "n_unparseable": baseline_eval.n_unparseable,
            "per_label": baseline_eval.per_label,
            "predicted_distribution": baseline_eval.predicted_distribution,
        },
        "n_disagreements": len(disagreements),
        "sample_disagreements": disagreements[:10],
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-documents", type=int, default=settings.n_documents)
    parser.add_argument("--seed", type=int, default=settings.dataset_seed)
    parser.add_argument("--rebuild", action="store_true", help="Force fresh extraction instead of using cached graphs/embeddings.")
    args = parser.parse_args()

    result = run_comparison(args.n_documents, args.seed, args.rebuild)

    print("\n" + "=" * 70)
    print(json.dumps(result, indent=2))
    RESULTS_FILE.write_text(json.dumps(result, indent=2))
    print(f"\nSaved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
