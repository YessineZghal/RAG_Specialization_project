"""Hierarchical Graph-of-Thoughts (HGoT) -- the RAG-specific descendant of
Graph-of-Thoughts this level's README names as its actual target, closer
to what "reasoning + retrieval" looks like in practice than the generic
algorithmic-puzzle framing most ToT/GoT demos use.

Where `graph_search.py`'s Graph-of-Thoughts reasons over one fixed block
of context, HGoT decomposes the question into several sub-questions and
retrieves **separate, real evidence for each one independently** -- a
genuinely different retrieval call per sub-question, not the same context
reused. Each sub-question is answered from only its own retrieved
evidence, and the final verdict is a vote over all the sub-answers
together, carrying forward exactly which retrieved facts (`doc_id`s)
supported it -- the "citation-aware" part of the technique.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reasoning_common.answer_parsing import parse_yes_no
from reasoning_common.llm import OllamaLLM

DECOMPOSE_PROMPT = (
    "Break the following yes/no question down into {n} simpler sub-"
    "questions whose answers, once known, would let you answer the "
    "original question. Return ONLY the sub-questions, one per line, no "
    "numbering.\n\nQuestion: {question}"
)

SUBANSWER_PROMPT = (
    "Context:\n{context}\n\nSub-question: {sub_question}\n\n"
    "Answer this sub-question using only the context above. If the "
    "context does not address it, say so plainly. Keep your answer to "
    "one or two sentences."
)

VOTE_PROMPT = (
    "Original question: {question}\n\n"
    "The following sub-questions were each investigated separately, with "
    "an answer grounded in its own retrieved evidence:\n\n{sub_answers}\n\n"
    "Based on all of these sub-answers together, give your final answer "
    "to the ORIGINAL question, on its own line, in exactly this form: "
    "'Answer: Yes' or 'Answer: No'."
)


def decompose_question(question: str, n: int = 3, llm: OllamaLLM | None = None) -> list[str]:
    llm = llm or OllamaLLM()
    raw = llm.complete(DECOMPOSE_PROMPT.format(question=question, n=n))
    sub_questions = [line.strip("-*0123456789. \t") for line in raw.splitlines() if line.strip()]
    sub_questions = [s for s in sub_questions if s]
    return sub_questions[:n] if sub_questions else [question]


def answer_subquestion(
    sub_question: str,
    retriever,
    corpus: dict[str, str],
    top_k: int = 3,
    llm: OllamaLLM | None = None,
) -> dict:
    llm = llm or OllamaLLM()
    results = retriever.search(sub_question, top_k=top_k)
    evidence_ids = [doc_id for doc_id, _score in results]
    context = "\n".join(corpus.get(doc_id, "") for doc_id in evidence_ids)
    prompt = SUBANSWER_PROMPT.format(context=context or "(no evidence retrieved)", sub_question=sub_question)
    answer_text = llm.complete(prompt)
    return {"sub_question": sub_question, "answer_text": answer_text, "evidence_ids": evidence_ids}


def hgot_answer(
    question: str,
    retriever,
    corpus: dict[str, str],
    n_subquestions: int = 3,
    top_k: int = 3,
    llm: OllamaLLM | None = None,
) -> dict:
    llm = llm or OllamaLLM()
    llm_calls = 0

    sub_questions = decompose_question(question, n=n_subquestions, llm=llm)
    llm_calls += 1

    sub_results = []
    all_evidence_ids: list[str] = []
    for sub_question in sub_questions:
        result = answer_subquestion(sub_question, retriever, corpus, top_k=top_k, llm=llm)
        llm_calls += 1
        sub_results.append(result)
        all_evidence_ids.extend(result["evidence_ids"])

    sub_answers_text = "\n\n".join(
        f"Sub-question: {r['sub_question']}\nAnswer: {r['answer_text']}" for r in sub_results
    )
    raw = llm.complete(VOTE_PROMPT.format(question=question, sub_answers=sub_answers_text))
    llm_calls += 1

    return {
        "answer": parse_yes_no(raw),
        "sub_questions": sub_questions,
        "sub_results": sub_results,
        # dedup while preserving first-seen order -- the real, citable set
        # of evidence this answer actually rests on.
        "cited_evidence_ids": list(dict.fromkeys(all_evidence_ids)),
        "reasoning": raw,
        "llm_calls": llm_calls,
    }
