"""Real, aggregate evaluation of this level's adaptive components against
HotpotQA's genuine ground truth — not single hand-picked examples.

Answers, with numbers, the questions this level exists to raise:
1. How accurate is each query classifier, really?
2. Does multi-hop decomposition actually retrieve more gold evidence than
   a single plain retrieval step?
3. How often does CRAG correctly flag weak evidence?
"""

from __future__ import annotations

import sys
from pathlib import Path

_LEVEL_DIR = Path(__file__).resolve().parent.parent
for sub in ["", "query-classification", "multi-hop-rag", "corrective-rag"]:
    sys.path.insert(0, str(_LEVEL_DIR / sub) if sub else str(_LEVEL_DIR))

from classifier import classify_ensemble, classify_llm, classify_rule  # noqa: E402
from crag import corrective_retrieve  # noqa: E402
from subquestion_retrieval import multi_hop_retrieve  # noqa: E402

EXPECTED_LABEL = {"bridge": "multi_hop", "comparison": "complex"}


def evaluate_classification(questions: dict[str, dict], n_per_type: int = 15) -> dict:
    """Sample `n_per_type` real bridge and comparison questions each, score
    all three classifiers against HotpotQA's own type labels.
    """
    by_type: dict[str, list[dict]] = {"bridge": [], "comparison": []}
    for q in questions.values():
        if q["type"] in by_type and len(by_type[q["type"]]) < n_per_type:
            by_type[q["type"]].append(q)

    results = {}
    for classifier_name, classifier_fn in [
        ("rule", classify_rule), ("llm", classify_llm), ("ensemble", classify_ensemble),
    ]:
        scores = {}
        for qtype, items in by_type.items():
            expected = EXPECTED_LABEL[qtype]
            correct = sum(1 for item in items if classifier_fn(item["question"]) == expected)
            scores[qtype] = (correct, len(items))
        results[classifier_name] = scores
    return results


def evaluate_multi_hop_retrieval(
    questions: dict[str, dict], corpus: dict[str, str], retriever, n: int = 30, top_k: int = 5
) -> dict:
    """Compare gold-supporting-doc recall: one plain retrieval call vs.
    multi-hop decomposition, on real bridge questions only (comparison
    questions don't have a "hop order" to exploit).
    """
    bridge_items = [q for q in questions.values() if q["type"] == "bridge"][:n]

    plain_hits, multihop_hits, total_gold = 0, 0, 0
    for item in bridge_items:
        gold = set(item["supporting_titles"])
        total_gold += len(gold)

        plain_ids = {d for d, _ in retriever.search(item["question"], top_k=top_k)}
        plain_hits += len(plain_ids & gold)

        multihop_ids = {
            d for d, _ in multi_hop_retrieve(item["question"], retriever, corpus=corpus, top_k_per_hop=top_k)
        }
        multihop_hits += len(multihop_ids & gold)

    return {
        "n_questions": len(bridge_items),
        "total_gold_docs": total_gold,
        "plain_recall": plain_hits / total_gold if total_gold else 0.0,
        "multi_hop_recall": multihop_hits / total_gold if total_gold else 0.0,
    }


def evaluate_crag(questions: dict[str, dict], corpus: dict[str, str], retriever, n: int = 20) -> dict:
    """How often does CRAG's confidence correctly track whether the
    retrieved evidence actually contains a real supporting document?
    """
    items = list(questions.values())[:n]
    agree = 0
    for item in items:
        gold = set(item["supporting_titles"])
        graded = corrective_retrieve(item["question"], retriever, corpus, top_k=5)
        retrieved_ids = {g["doc_id"] for g in graded["graded_results"]}
        has_gold = bool(retrieved_ids & gold)
        # "Agreement": trustworthy when gold evidence was actually retrieved,
        # not trustworthy when it wasn't.
        if graded["trustworthy"] == has_gold:
            agree += 1
    return {"n_questions": len(items), "agreement": agree / len(items) if items else 0.0}
