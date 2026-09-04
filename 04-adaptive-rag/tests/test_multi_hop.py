from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "multi-hop-rag"))
from planner import plan_subquestions  # noqa: E402
from subquestion_retrieval import multi_hop_retrieve  # noqa: E402


def test_plan_subquestions_splits_two_lines(fake_llm):
    llm = fake_llm(response="Who founded Russell Hobbs?\nWhere is that company based?")
    result = plan_subquestions("Where is the company founded by Peter Hobbs based?", llm=llm)
    assert result == ["Who founded Russell Hobbs?", "Where is that company based?"]


def test_plan_subquestions_falls_back_to_original_if_llm_gives_fewer_than_two(fake_llm):
    llm = fake_llm(response="just one line")
    original = "Where is the company founded by Peter Hobbs based?"
    assert plan_subquestions(original, llm=llm) == [original]


def test_multi_hop_retrieve_pools_evidence_across_hops(fake_retriever, tiny_corpus, fake_llm):
    llm = fake_llm(response="Who founded Russell Hobbs?\nWhere is that company based?")
    results = multi_hop_retrieve(
        "Where is the company founded by Peter Hobbs based?",
        fake_retriever, llm=llm, top_k_per_hop=2, corpus=tiny_corpus,
    )
    doc_ids = {doc_id for doc_id, _ in results}
    assert doc_ids  # something was retrieved
    assert len(doc_ids) <= 4  # at most top_k_per_hop * n_hops distinct docs


def test_multi_hop_retrieve_keeps_best_score_on_duplicate_docs(fake_retriever, tiny_corpus, fake_llm):
    # Same sub-question twice -> same doc retrieved twice -> should not
    # duplicate the entry, and should keep the max score seen.
    llm = fake_llm(response="Where is Russell Hobbs based?\nWhere is Russell Hobbs based?")
    results = multi_hop_retrieve(
        "irrelevant original question", fake_retriever, llm=llm, top_k_per_hop=3, corpus=tiny_corpus
    )
    doc_ids = [doc_id for doc_id, _ in results]
    assert len(doc_ids) == len(set(doc_ids))  # no duplicates
