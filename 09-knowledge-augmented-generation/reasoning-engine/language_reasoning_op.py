"""The `language_reasoning` operator: the final synthesis step every
question in this level goes through, since PubMedQA's real ground truth
is always a spoken yes/no/maybe verdict -- reading whatever evidence the
other three operators gathered (retrieved text, KG facts, a numeric
result) and turning it into that verdict plus a short justification.

This is the one operator that is genuinely an LLM call; the other three
are either deterministic (`numerical_op`) or a plain lookup/search
(`retrieval_op`, `kg_reasoning_op`) with no free-text judgment involved.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kag_common.answer_parsing import parse_yes_no_maybe  # noqa: E402
from kag_common.llm import OllamaLLM  # noqa: E402

LANGUAGE_REASONING_PROMPT = """Answer the following biomedical research question using ONLY the evidence given. The evidence may include retrieved abstract text, structured knowledge-graph facts, and/or a numeric calculation result.

Evidence:
{evidence}

Question: {question}

Think briefly, then give your final verdict on its own last line, in exactly this form: "Answer: Yes", "Answer: No", or "Answer: Maybe"."""


@dataclass(frozen=True)
class LanguageReasoningResult:
    verdict: str | None  # "yes" / "no" / "maybe" / None if unparseable
    raw_response: str


def reason_in_language(
    question: str,
    evidence: str,
    llm: OllamaLLM | None = None,
) -> LanguageReasoningResult:
    llm = llm or OllamaLLM()
    prompt = LANGUAGE_REASONING_PROMPT.format(evidence=evidence or "(no evidence retrieved)", question=question)
    raw = llm.complete(prompt, temperature=0.0)
    return LanguageReasoningResult(verdict=parse_yes_no_maybe(raw), raw_response=raw)
