"""Real numeric pattern recognition over the tables `table_retrieval.py`
already finds -- an honest, modest version of RAG-Anything's "statistical
pattern recognition on tabular data" (see `../../missing_to_complite.md`),
scoped to what this repo's own text-window extraction actually allows.

**Why decimal numbers only, not every digit**: verified directly against
the real PDF's own table-2 text (`"ByteNet [18] 23.75 ... GNMT + RL [38]
24.6 39.92 2.3· 1019 1.4· 1020"`) -- a bare-integer regex would also
capture citation brackets (`[18]`, `[38]`) and the mangled scientific
notation this paper's own PDF text extraction produces for `10^19`,
`10^20` (flattened into a plain `1019`/`1020` with no exponent marker,
the same kind of extraction damage `equation_retrieval.py`'s module
docstring already discloses for equations). Requiring an actual decimal
point (`23.75`, `39.2`, ...) is a cheap, real filter that keeps every
genuine BLEU/metric score in this table and excludes every citation
number and mangled exponent -- verified against this exact real text,
not assumed to generalize.

**Not** the structured, cell-level statistical analysis a properly
parsed table would allow -- `table_retrieval.py`'s own module docstring
already discloses why real cell-level parsing isn't available here
(`pdfplumber` badly mangles this paper's tables). This module works with
the same caption-plus-text-window this level already has, not a
structured table this repo has never actually been able to parse.
"""

from __future__ import annotations

import re

_NUMBER_RE = re.compile(r"\b\d+\.\d+\b")


def extract_numbers(text: str) -> list[float]:
    """Every decimal number found in `text`, in the order they appear."""
    return [float(match) for match in _NUMBER_RE.findall(text)]


def compute_statistics(numbers: list[float]) -> dict | None:
    """`None` for an empty list -- "no numbers found" is a real, distinct
    outcome from "the numbers found were all zero," and callers should be
    able to tell the two apart."""
    if not numbers:
        return None
    return {
        "count": len(numbers),
        "min": min(numbers),
        "max": max(numbers),
        "mean": sum(numbers) / len(numbers),
    }


def enrich_tables_with_statistics(tables: dict[str, dict]) -> dict[str, dict]:
    """Add a `"statistics"` key to every table dict in `tables` (from
    `table_retrieval.extract_tables()`), computed from its own extracted
    text. Mutates and returns `tables` for convenience; safe to call
    multiple times (recomputes each time, does not accumulate)."""
    for table in tables.values():
        numbers = extract_numbers(table["text"])
        table["statistics"] = compute_statistics(numbers)
    return tables
