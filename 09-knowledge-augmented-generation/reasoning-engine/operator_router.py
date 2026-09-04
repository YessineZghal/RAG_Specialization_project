"""The hybrid reasoning engine's router: dispatch a parsed `LogicalForm`
to whichever of the four operators it named, merge their evidence, and
run the final language-reasoning synthesis -- the
"Query -> Logical-form parser -> router -> {operators} -> Merge evidence
-> Answer + citations" pipeline from this level's README architecture
diagram, made concrete.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kg_reasoning_op import reason_over_graph  # noqa: E402
from language_reasoning_op import reason_in_language  # noqa: E402
from logical_form_parser import LogicalForm, parse_logical_form  # noqa: E402
from numerical_op import evaluate_numeric  # noqa: E402
from retrieval_op import retrieve  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from indexing.mutual_index import MutualIndex  # noqa: E402
from kag_common.embed import OllamaEmbedder  # noqa: E402
from kag_common.llm import OllamaLLM  # noqa: E402


@dataclass(frozen=True)
class KagAnswer:
    verdict: str | None
    operators_used: tuple[str, ...]
    citations: frozenset[str]
    evidence_text: str
    raw_response: str
    numeric_result: object | None = None


def answer_question(
    question: str,
    corpus: dict[str, str],
    doc_ids: list[str],
    matrix: np.ndarray,
    graph: nx.MultiDiGraph,
    mutual_index: MutualIndex,
    embedder: OllamaEmbedder | None = None,
    llm: OllamaLLM | None = None,
    logical_form: LogicalForm | None = None,
    top_k: int = 3,
) -> KagAnswer:
    llm = llm or OllamaLLM()
    embedder = embedder or OllamaEmbedder()
    logical_form = logical_form or parse_logical_form(question, llm)

    citations: set[str] = set()
    evidence_parts: list[str] = []
    numeric_result = None

    if "retrieval" in logical_form.operators:
        for hit in retrieve(question, corpus, doc_ids, matrix, embedder, top_k=top_k):
            citations.add(hit.doc_id)
            evidence_parts.append(f"[retrieved doc {hit.doc_id}] {hit.text}")

    if "kg_reasoning" in logical_form.operators:
        kg_evidence = reason_over_graph(graph, mutual_index, logical_form.focus_hint)
        citations |= kg_evidence.doc_ids
        if kg_evidence.facts:
            evidence_parts.append("Knowledge graph facts:\n" + "\n".join(kg_evidence.facts))

    if "numerical_calculation" in logical_form.operators and logical_form.numeric_comparison:
        numeric_result = evaluate_numeric(graph, logical_form.focus_hint, logical_form.numeric_comparison)
        evidence_parts.append(f"Numeric calculation: {numeric_result.explanation}")

    evidence_text = "\n\n".join(evidence_parts)
    language_result = reason_in_language(question, evidence_text, llm)

    return KagAnswer(
        verdict=language_result.verdict,
        operators_used=logical_form.operators,
        citations=frozenset(citations),
        evidence_text=evidence_text,
        raw_response=language_result.raw_response,
        numeric_result=numeric_result,
    )
