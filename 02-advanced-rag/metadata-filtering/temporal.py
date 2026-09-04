"""Temporal filtering — narrow retrieval results to documents that were
valid as of a given date, or that fall inside a date range.

This is the same mechanism as `filters.py`'s `filter_by_metadata` — a
predicate applied to a candidate's metadata after a wide retrieval pass —
specialized to the one field that keeps coming up in real systems: "what
was true as of a given date," "only documents from the last two years,"
"only policies that were in effect during Q1."

scifact ships no publication dates in the version of the dataset this repo
uses, so `build_temporal_metadata` derives a synthetic `year` field (the
same disclosed-simplification pattern `filters.py`'s
`build_length_metadata` already uses for document length) purely so
temporal filtering is demonstrable end-to-end. Swap in a corpus with real
dates and the rest of this file needs no changes.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class DateRange:
    """A half-open predicate over a document's `year` metadata field.

    `after` and `before` are both inclusive when set; leaving one unset
    means "no lower/upper bound." `as_of(year)` is a convenience
    constructor for the common "what was true on this date" case: it keeps
    only documents published on or before that year.
    """

    after: int | None = None
    before: int | None = None

    @classmethod
    def as_of(cls, year: int) -> "DateRange":
        return cls(after=None, before=year)

    def contains(self, year: int) -> bool:
        if self.after is not None and year < self.after:
            return False
        if self.before is not None and year > self.before:
            return False
        return True


def build_temporal_metadata(
    corpus: dict[str, dict],
    min_year: int = 2015,
    max_year: int = 2024,
    seed: int = 42,
) -> dict[str, dict]:
    """Assign each document a synthetic `year` in `[min_year, max_year]`.

    Deterministic for a given `seed` and a given corpus (same doc_ids
    always get the same year), so re-running this against the same
    cached corpus produces the same filtered results every time.
    """
    metadata = {}
    for doc_id in sorted(corpus):
        rng = random.Random(f"{seed}:{doc_id}")
        metadata[doc_id] = {"year": rng.randint(min_year, max_year)}
    return metadata


def filter_by_date_range(
    candidates: list[tuple[str, float]],
    metadata: dict[str, dict],
    date_range: DateRange,
) -> list[tuple[str, float]]:
    return [
        (doc_id, score)
        for doc_id, score in candidates
        if "year" in metadata.get(doc_id, {}) and date_range.contains(metadata[doc_id]["year"])
    ]


def temporal_search(
    retriever,
    query: str,
    metadata: dict[str, dict],
    date_range: DateRange,
    top_k: int = 10,
    candidate_k: int = 50,
) -> list[tuple[str, float]]:
    """Retrieve wide (`candidate_k`), keep only documents inside
    `date_range`, then truncate to `top_k` — same retrieve-then-filter
    shape as `filters.filtered_search`.
    """
    candidates = retriever.search(query, top_k=candidate_k)
    return filter_by_date_range(candidates, metadata, date_range)[:top_k]
