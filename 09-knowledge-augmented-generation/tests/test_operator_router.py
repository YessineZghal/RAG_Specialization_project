"""Integration test for `answer_question` -- scripts the exact sequence of
LLM calls (logical-form parse, then the final language-reasoning verdict)
so the router's own orchestration (which operators ran, what evidence
each contributed, which documents got cited) is verified directly, the
same "script the fake and check the exact call sequence" approach
08-reasoning-strategies' `test_tree_search.py` already used.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "reasoning-engine"))
from logical_form_parser import LogicalForm
from operator_router import answer_question

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from indexing.mutual_index import MutualIndex


def _build_graph_and_index():
    graph = nx.MultiDiGraph()
    graph.add_node("study-1", name="Study-1", type="Study", attributes={})
    graph.add_node("diabetes", name="Diabetes", type="Condition", attributes={})
    graph.add_node("pop-1", name="Pop-1", type="Population", attributes={"size": 620})
    graph.add_edge("study-1", "diabetes", relation="STUDIES", doc_id="d1")
    graph.add_edge("study-1", "pop-1", relation="HAS_POPULATION", doc_id="d1")

    index = MutualIndex()
    index.add_relation("Study-1", "STUDIES", "Diabetes", "d1")
    index.add_relation("Study-1", "HAS_POPULATION", "Pop-1", "d1")
    return graph, index


def _corpus_and_embeddings():
    corpus = {"d1": "Study-1 studied diabetes.", "d2": "Unrelated abstract."}
    doc_ids = ["d1", "d2"]
    matrix = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    return corpus, doc_ids, matrix


def test_answer_question_runs_the_parsed_operators_and_merges_evidence(fake_llm, fake_embedder):
    graph, index = _build_graph_and_index()
    corpus, doc_ids, matrix = _corpus_and_embeddings()
    embedder = fake_embedder(vectors={"Does the study concern diabetes?": [1.0, 0.0]})

    logical_form_response = json.dumps(
        {"operators": ["retrieval", "kg_reasoning"], "focus_hint": "diabetes", "numeric_comparison": None}
    )
    llm = fake_llm(responses=[logical_form_response, "It does.\nAnswer: Yes"])

    answer = answer_question(
        "Does the study concern diabetes?", corpus, doc_ids, matrix, graph, index,
        embedder=embedder, llm=llm,
    )

    assert answer.verdict == "yes"
    assert set(answer.operators_used) == {"retrieval", "kg_reasoning", "language_reasoning"}
    assert "d1" in answer.citations  # both the retrieval hit and the KG fact point at d1
    assert "Knowledge graph facts" in answer.evidence_text
    assert len(llm.calls) == 2  # one logical-form parse, one final synthesis


def test_answer_question_runs_the_numerical_operator_when_parsed(fake_llm, fake_embedder):
    graph, index = _build_graph_and_index()
    corpus, doc_ids, matrix = _corpus_and_embeddings()
    embedder = fake_embedder()

    logical_form_response = json.dumps(
        {
            "operators": ["numerical_calculation"],
            "focus_hint": "diabetes",
            "numeric_comparison": {"attribute": "size", "op": ">", "value": 500},
        }
    )
    llm = fake_llm(responses=[logical_form_response, "Answer: Yes"])

    answer = answer_question(
        "Was the population larger than 500?", corpus, doc_ids, matrix, graph, index,
        embedder=embedder, llm=llm,
    )

    assert answer.numeric_result is not None
    assert answer.numeric_result.comparison_result is True
    assert "Numeric calculation" in answer.evidence_text
    assert "Numeric calculation" in llm.calls[1]  # the final synthesis call actually saw it


def test_answer_question_skips_parsing_when_a_logical_form_is_supplied(fake_llm, fake_embedder):
    graph, index = _build_graph_and_index()
    corpus, doc_ids, matrix = _corpus_and_embeddings()
    embedder = fake_embedder()
    llm = fake_llm(response="Answer: No")

    form = LogicalForm(operators=("language_reasoning",))
    answer = answer_question(
        "some question", corpus, doc_ids, matrix, graph, index,
        embedder=embedder, llm=llm, logical_form=form,
    )

    assert answer.verdict == "no"
    assert len(llm.calls) == 1  # no logical-form parse call was made


def test_answer_question_dispatches_to_the_global_operator_when_summaries_are_given(fake_llm, fake_embedder):
    graph, index = _build_graph_and_index()
    corpus, doc_ids, matrix = _corpus_and_embeddings()
    embedder = fake_embedder(vectors={
        "What are the overall research themes here?": [1.0, 0.0],
        "Diabetes-related intervention studies.": [1.0, 0.0],
    })
    community_summaries = {"community-0": {"nodes": ["study-1", "diabetes"], "summary": "Diabetes-related intervention studies."}}
    llm = fake_llm(response="Answer: Yes")

    form = LogicalForm(operators=("global", "language_reasoning"))
    answer = answer_question(
        "What are the overall research themes here?", corpus, doc_ids, matrix, graph, index,
        embedder=embedder, llm=llm, logical_form=form, community_summaries=community_summaries,
    )

    assert "global" in answer.operators_used
    assert "Community-level themes" in answer.evidence_text
    assert "Diabetes-related intervention studies." in answer.evidence_text


def test_answer_question_skips_the_global_operator_when_no_summaries_are_given(fake_llm, fake_embedder):
    graph, index = _build_graph_and_index()
    corpus, doc_ids, matrix = _corpus_and_embeddings()
    embedder = fake_embedder()
    llm = fake_llm(response="Answer: No")

    form = LogicalForm(operators=("global", "language_reasoning"))
    answer = answer_question(
        "some question", corpus, doc_ids, matrix, graph, index,
        embedder=embedder, llm=llm, logical_form=form, community_summaries=None,
    )

    assert "Community-level themes" not in answer.evidence_text
