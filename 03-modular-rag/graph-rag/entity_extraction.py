"""Extract (subject, relation, object) triples from text using a local
LLM — the raw material `graph_builder.py` turns into a queryable graph.

LLMs are inconsistent about JSON formatting even when asked nicely, so
parsing here is defensive: try strict JSON first, fall back to pulling out
the first `[...]` block if the model wrapped its answer in prose.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modular_common.llm import OllamaLLM

EXTRACTION_PROMPT = """Extract factual (subject, relation, object) triples from the text below.
Only extract facts that are explicitly stated. Use short, consistent entity names.

Respond with ONLY a JSON array, like:
[{{"subject": "A", "relation": "works at", "object": "B"}}]

Text:
{text}

JSON:"""


def extract_triples(text: str, llm: OllamaLLM | None = None) -> list[dict]:
    llm = llm or OllamaLLM()
    raw = llm.complete(EXTRACTION_PROMPT.format(text=text), temperature=0.0)
    return _parse_triples(raw)


def _parse_triples(raw: str) -> list[dict]:
    candidate = raw.strip().strip("`")
    if candidate.lower().startswith("json"):
        candidate = candidate[4:].strip()

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not match:
            return []
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []

    if not isinstance(parsed, list):
        return []

    triples = []
    for item in parsed:
        if isinstance(item, dict) and {"subject", "relation", "object"} <= item.keys():
            triples.append(
                {
                    "subject": str(item["subject"]).strip(),
                    "relation": str(item["relation"]).strip(),
                    "object": str(item["object"]).strip(),
                }
            )
    return triples
