"""Pure text helpers with zero third-party dependencies.

Kept deliberately tiny and dependency-free so every level — from the
naive pipeline to the production API — can import them without pulling in
anything heavier than the standard library.
"""

from __future__ import annotations

import re

_WORD_RE = re.compile(r"[a-z0-9]+")


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace and strip the ends."""
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    """Lowercase, alphanumeric-only word tokenization (no stemming/stopwords)."""
    return _WORD_RE.findall(text.lower())


def jaccard_similarity(a: str, b: str) -> float:
    """Word-overlap similarity in [0, 1]. 1.0 means identical vocabularies.

    Deliberately naive (no stemming, no stopword removal, no weighting) —
    good enough to approximate "does this passage plausibly contain the
    answer?" for building a best-effort evaluation set, not a substitute
    for the IR metrics in `02-advanced-rag/evaluation/`.
    """
    tokens_a, tokens_b = set(tokenize(a)), set(tokenize(b))
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)
