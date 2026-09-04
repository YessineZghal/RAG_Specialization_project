"""Parse a natural-language question into a logical form: which of the
hybrid-reasoning operators it needs, plus any structured hints those
operators require.

Five operators as of the RAG-Anything gap review (see
`../../missing_to_complite.md`): the original four (retrieval,
kg_reasoning, language_reasoning, numerical_calculation), plus `global` --
a community-level operator answering the "what are the overall themes in
this corpus" style of question this level's own README always disclosed
as unbuilt (`indexing/community_summary.py`, `global_op.py`), the same
gap Microsoft GraphRAG and RAG-Anything/LightRAG's own `global` query
mode both name independently.

This is the level's other real classification decision (the first is
`operator_router`'s dispatch itself) -- and, per Level 4's query
classifier and Level 3's router, a classification decision has its own
error rate. `parse_logical_form` fails *open* on an unparseable response:
`language_reasoning` and `retrieval` are always included as a safety net
so a parser failure still produces an answer, just without the extra
structured evidence a correct parse would have added -- the same
fail-open philosophy as 04-adaptive-rag's `fallback-strategies`.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kag_common.llm import OllamaLLM

ALL_OPERATORS = ("retrieval", "kg_reasoning", "language_reasoning", "numerical_calculation", "global")

PARSE_PROMPT = """Read the question and decide which reasoning operators are needed to answer it. Available operators:

- retrieval: pull the relevant source text passages
- kg_reasoning: follow structured relations in a knowledge graph (e.g. which intervention a study used, which outcome it reported)
- language_reasoning: read text and judge/summarize in natural language (needed for almost every question that ends in a yes/no/maybe verdict)
- numerical_calculation: compare a number in the text/graph against a threshold (e.g. "larger than 500 patients")
- global: the question asks about an overall theme or pattern spanning many documents (e.g. "what are the main research directions in this corpus?"), not one specific fact

Also extract:
- focus_hint: the key entity/condition/study the question is about, as a short phrase, or null
- numeric_comparison: if numerical_calculation is needed, {{"attribute": "size", "op": ">", "value": <number>}} (op is one of >, <, >=, <=, ==), else null

Respond with ONLY a JSON object of this exact shape:
{{"operators": ["..."], "focus_hint": "..." or null, "numeric_comparison": {{...}} or null}}

Question: {question}

JSON:"""


class _NumericComparison(BaseModel):
    attribute: str
    op: str
    value: float


class _LogicalFormShape(BaseModel):
    operators: list[str] = []
    focus_hint: str | None = None
    numeric_comparison: _NumericComparison | None = None


@dataclass(frozen=True)
class LogicalForm:
    operators: tuple[str, ...]
    focus_hint: str | None = None
    numeric_comparison: dict | None = None
    fell_back: bool = False


def _extract_json_object(raw: str) -> dict:
    candidate = raw.strip().strip("`")
    if candidate.lower().startswith("json"):
        candidate = candidate[4:].strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}


def _fallback() -> LogicalForm:
    return LogicalForm(operators=("retrieval", "language_reasoning"), fell_back=True)


def parse_logical_form(question: str, llm: OllamaLLM | None = None) -> LogicalForm:
    """Classify `question` into a `LogicalForm`. Always returns a usable
    result -- an unparseable LLM response degrades to the fail-open
    default rather than raising."""
    llm = llm or OllamaLLM()
    raw = llm.complete(PARSE_PROMPT.format(question=question), temperature=0.0)
    payload = _extract_json_object(raw)
    if not payload:
        return _fallback()

    try:
        shape = _LogicalFormShape.model_validate(payload)
    except ValidationError:
        return _fallback()

    operators = tuple(op for op in shape.operators if op in ALL_OPERATORS)
    if not operators:
        return _fallback()
    if "language_reasoning" not in operators:
        # every PubMedQA-style question ends in a spoken yes/no/maybe
        # verdict -- language_reasoning is never truly optional, only
        # the *other* operators are
        operators = operators + ("language_reasoning",)

    numeric_comparison = (
        shape.numeric_comparison.model_dump() if shape.numeric_comparison is not None else None
    )
    return LogicalForm(operators=operators, focus_hint=shape.focus_hint, numeric_comparison=numeric_comparison)
