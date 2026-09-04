"""Self-query retrieval — ask the LLM to split one natural-language
question into (a) the plain-text topic to search for and (b) the
structured filters implied by the wording, instead of making the caller
write both by hand.

Example: "AI papers from 2023 or later that are short" should become
roughly:

    semantic_query = "AI papers"
    min_year = 2023
    length_bucket = "short"

`self_query()` then hands the semantic part to a normal retriever and the
structured part to `filters.filter_by_metadata` — this module is the glue
between the two, not a new retrieval mechanism of its own.

This is the one module in this level that uses Pydantic. Every other query
transformation in this folder (`query_rewrite.py`, `multi_query.py`, ...)
only ever gets back a plain string from the LLM, so there is nothing to
validate beyond stripping whitespace. Here the LLM is asked to return
*structured* data (several typed fields at once), and a real production
system cannot trust that blindly: the model might return a year as the
string "2023" instead of the integer 2023, invent a `length_bucket` value
that was never one of the three allowed choices, or omit a field
entirely. Pydantic is the right tool for exactly this: it parses the raw
JSON into a strictly-typed model, coerces the obvious cases (the string
"2023" does become the integer 2023), and raises one clear error for
anything it cannot make sense of — so this module can catch that one
error and fall back safely, instead of the caller hitting a confusing
`KeyError` or `TypeError` several lines later.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.llm import OllamaLLM

from filters import filter_by_metadata
from temporal import DateRange, filter_by_date_range


class SelfQueryFilters(BaseModel):
    """The structured shape the LLM's answer must fit.

    `semantic_query` is always required. Everything else is optional,
    because most real questions only imply one or two filters, not all of
    them at once.
    """

    semantic_query: str
    min_year: int | None = None
    max_year: int | None = None
    length_bucket: Literal["short", "medium", "long"] | None = None


SELF_QUERY_PROMPT = (
    "Read the following question and split it into a plain search topic "
    "plus any structured filters it implies.\n\n"
    "Respond with ONLY a JSON object with these exact keys:\n"
    '  "semantic_query": the core topic to search for, as plain text\n'
    '  "min_year": the earliest year mentioned, as an integer, or null\n'
    '  "max_year": the latest year mentioned, as an integer, or null\n'
    '  "length_bucket": one of "short", "medium", "long" if the question '
    "asks for document length, otherwise null\n\n"
    "Question: {query}\n\nJSON:"
)


def _extract_json_object(raw: str) -> dict:
    """Pull the first `{...}` object out of `raw`, tolerating the model
    wrapping it in prose or a markdown code fence — the same defensive
    extraction pattern used for LLM JSON output throughout this repo (see
    07-production-rag/production_eval/ragas_eval.py's `_extract_claims`).
    """
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


def parse_self_query(query: str, llm: OllamaLLM | None = None) -> SelfQueryFilters:
    """Ask the LLM to split `query` into a semantic part and structured
    filters, and validate the result with `SelfQueryFilters`.

    Falls back to `SelfQueryFilters(semantic_query=query)` (no filters at
    all) if the LLM's output cannot be parsed or does not fit the expected
    shape — a self-query system that cannot understand a filter should
    still be able to search, just without narrowing the results.
    """
    llm = llm or OllamaLLM()
    raw = llm.complete(SELF_QUERY_PROMPT.format(query=query))
    payload = _extract_json_object(raw)
    if not payload:
        return SelfQueryFilters(semantic_query=query)
    try:
        return SelfQueryFilters(**payload)
    except ValidationError:
        # The JSON parsed, but a field had the wrong shape (e.g. a
        # length_bucket outside the three allowed values). Still trust the
        # semantic_query if it is at least present and is a string.
        semantic_query = payload.get("semantic_query")
        if isinstance(semantic_query, str) and semantic_query.strip():
            return SelfQueryFilters(semantic_query=semantic_query)
        return SelfQueryFilters(semantic_query=query)


def self_query_search(
    query: str,
    retriever,
    metadata: dict[str, dict],
    llm: OllamaLLM | None = None,
    top_k: int = 10,
    candidate_k: int = 50,
) -> dict:
    """Run the full self-query pipeline: parse filters, retrieve on the
    semantic part, apply whichever filters were actually extracted.

    Returns a dict with the parsed filters alongside the results, so a
    caller (or a notebook) can show *why* a result set looks the way it
    does, not just the final list.
    """
    parsed = parse_self_query(query, llm=llm)
    candidates = retriever.search(parsed.semantic_query, top_k=candidate_k)

    if parsed.min_year is not None or parsed.max_year is not None:
        date_range = DateRange(after=parsed.min_year, before=parsed.max_year)
        candidates = filter_by_date_range(candidates, metadata, date_range)

    if parsed.length_bucket is not None:
        candidates = filter_by_metadata(
            candidates, metadata, lambda m: m.get("length_bucket") == parsed.length_bucket
        )

    return {"parsed_filters": parsed, "results": candidates[:top_k]}
