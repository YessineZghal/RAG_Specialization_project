"""Corrective RAG (CRAG) — grade retrieved evidence *before* trusting it,
and take a corrective action when it's weak, instead of handing the LLM
whatever came back from search and hoping for the best.

Grading uses the LLM itself (a lightweight, single-word relevance
judgment per retrieved chunk) rather than the raw cosine/BM25 score —
those scores are calibrated to "closest match found," not "actually
answers the question," which is exactly the gap CRAG exists to catch
(see Level 1's `theory/vector_search.md` for why brute-force search always
returns *something*, relevant or not).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from adaptive_common.llm import OllamaLLM

GRADE_PROMPT = """Does the following passage contain information that helps answer the question?
Respond with only one word: relevant, ambiguous, or irrelevant.

Question: {question}
Passage: {passage}
Judgment:"""

GRADES = ("relevant", "ambiguous", "irrelevant")


def grade_passage(question: str, passage: str, llm: OllamaLLM | None = None) -> str:
    llm = llm or OllamaLLM()
    response = llm.complete(GRADE_PROMPT.format(question=question, passage=passage)).strip().lower()
    # NB: word-boundary matching, not plain substring containment --
    # "relevant" is a literal substring of "irrelevant", so a naive
    # `"relevant" in response` check (with "relevant" tested first) would
    # misread every "irrelevant" verdict as "relevant", silently defeating
    # the entire point of CRAG's evidence grading.
    for grade in GRADES:
        if re.search(rf"\b{grade}\b", response):
            return grade
    return "ambiguous"


def corrective_retrieve(
    question: str,
    retriever,
    corpus: dict[str, str],
    llm: OllamaLLM | None = None,
    top_k: int = 5,
    min_relevant: int = 1,
) -> dict:
    """Retrieve, grade every result, and report whether the evidence as a
    whole is trustworthy enough to answer from — the caller (or
    `fallback-strategies/`) decides what to do if it isn't.

    `trustworthy` is "at least `min_relevant` graded-relevant passages,"
    NOT "most of the Top-K is relevant." That distinction matters more
    than it looks: measured on 20 real questions, a majority/ratio
    threshold (>=50% relevant) agreed with whether gold evidence was
    actually retrieved only 0/20 times, because a question needing two
    specific supporting documents out of a Top-5 will always have 3
    unavoidable "distractor" retrievals graded irrelevant even when both
    needed documents were found — capping precision at 40% no matter how
    good the retrieval is. The same "at least 1 relevant" check agreed
    17/20 times. See `../README.md#corrective-rag` for the full story.
    """
    llm = llm or OllamaLLM()
    results = retriever.search(question, top_k=top_k)

    graded = []
    for doc_id, score in results:
        grade = grade_passage(question, corpus.get(doc_id, ""), llm=llm)
        graded.append({"doc_id": doc_id, "score": score, "grade": grade})

    n_relevant = sum(1 for g in graded if g["grade"] == "relevant")
    confidence = n_relevant / len(graded) if graded else 0.0

    return {
        "question": question,
        "graded_results": graded,
        "confidence": confidence,
        "n_relevant": n_relevant,
        "trustworthy": n_relevant >= min_relevant,
    }
