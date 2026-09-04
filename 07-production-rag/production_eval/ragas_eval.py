"""Faithfulness + answer relevance — Ragas-style RAG-quality metrics,
hand-implemented against Ollama rather than depending on the actual
`ragas` package (which defaults to OpenAI models for its LLM-based
metrics; wiring it to a local model is possible but adds a fragile extra
integration layer this repo has avoided at every prior level's
evaluation step — see Level 2's `evaluation/`, Level 4's CRAG/Self-RAG,
Level 5's answer verification).

- **Faithfulness**: extract the individual factual claims in an answer,
  check each is supported by the retrieved context, score = fraction
  supported. Mirrors Ragas' actual faithfulness algorithm.
- **Answer relevance**: generate several questions the answer *would*
  address, embed them, and average their similarity to the real question.
  An answer that's off-topic generates dissimilar reverse-engineered
  questions even if every claim in it happens to be true.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from production_common.embed import OllamaEmbedder  # noqa: E402
from production_common.llm import OllamaLLM  # noqa: E402

CLAIMS_PROMPT = """Break the following answer into a list of individual factual claims.

Respond with ONLY a JSON array of strings, like: ["claim one", "claim two"]

Answer: {answer}
JSON:"""

SUPPORT_PROMPT = """Context: {context}

Claim: {claim}

Is this claim supported by the context above? Respond with only one word: yes or no.
Judgment:"""

REVERSE_QUESTION_PROMPT = """Generate {n} different questions that the following answer would be a good response to.
Return ONLY the questions, one per line, no numbering.

Answer: {answer}
Questions:"""


def _extract_claims(answer: str, llm: OllamaLLM) -> list[str]:
    raw = llm.complete(CLAIMS_PROMPT.format(answer=answer), temperature=0.0)
    candidate = raw.strip().strip("`")
    if candidate.lower().startswith("json"):
        candidate = candidate[4:].strip()
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        try:
            parsed = json.loads(match.group(0)) if match else []
        except json.JSONDecodeError:
            # Genuinely malformed output even after extraction (missing
            # comma, unterminated string, ...) -- degrade to "no claims
            # found" rather than crashing the whole evaluation run. This
            # is the exact defensive pattern already used in Level 3/6's
            # entity_extraction._parse_triples; missed here the first
            # time and caught by actually running a real evaluation batch.
            parsed = []
    return [str(c) for c in parsed] if isinstance(parsed, list) else []


def faithfulness(answer: str, context: str, llm: OllamaLLM | None = None) -> dict:
    llm = llm or OllamaLLM()
    claims = _extract_claims(answer, llm)
    if not claims:
        return {"score": 0.0, "claims": [], "n_supported": 0}

    supported = 0
    graded = []
    for claim in claims:
        response = llm.complete(SUPPORT_PROMPT.format(context=context[:3000], claim=claim)).strip().lower()
        is_supported = bool(re.search(r"\byes\b", response)) and not re.search(r"\bno\b", response)
        supported += is_supported
        graded.append({"claim": claim, "supported": is_supported})

    return {"score": supported / len(claims), "claims": graded, "n_supported": supported}


def answer_relevance(
    question: str, answer: str, llm: OllamaLLM | None = None, embedder: OllamaEmbedder | None = None, n: int = 3
) -> dict:
    llm = llm or OllamaLLM()
    embedder = embedder or OllamaEmbedder()

    raw = llm.complete(REVERSE_QUESTION_PROMPT.format(n=n, answer=answer))
    reverse_questions = [line.strip("-* ") for line in raw.splitlines() if line.strip()][:n]
    if not reverse_questions:
        return {"score": 0.0, "reverse_questions": []}

    question_vector = np.array(embedder.embed_one(question))
    q_norm = np.linalg.norm(question_vector) + 1e-12

    similarities = []
    for rq in reverse_questions:
        rq_vector = np.array(embedder.embed_one(rq))
        sim = float(np.dot(question_vector, rq_vector) / (q_norm * np.linalg.norm(rq_vector) + 1e-12))
        similarities.append(sim)

    return {"score": sum(similarities) / len(similarities), "reverse_questions": reverse_questions}
