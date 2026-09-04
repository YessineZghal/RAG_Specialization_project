"""Rule-based query router — fast, transparent, and a useful baseline (and
fallback) for `llm_router.py`. Classifies a question into one of this
level's five backends by keyword heuristics.

None of these rules need to be clever: the goal is a cheap first pass that
handles the obvious cases, leaving only genuinely ambiguous questions to
the (slower, costlier) LLM router.
"""

from __future__ import annotations

import re

ROUTES = ("documents", "sql", "graph", "web", "api")

_SQL_PATTERNS = re.compile(
    r"\bhow many\b|\btotal\b|\bcount\b|\baverage\b|\brevenue\b|\binvoice(s)?\b|"
    r"\btrack(s)?\b|\balbum(s)?\b|\bcustomer(s)?\b|\bartist(s)?\b|\bplaylist\b|\bgenre\b",
    re.IGNORECASE,
)
_GRAPH_PATTERNS = re.compile(
    # \w* suffixes so inflected forms match too (affiliated, authored, working, ...)
    r"\bwho\b.{0,20}\b(work\w*|author\w*|affiliat\w*|wrote|created)\b|\brelated to\b|"
    r"\bworks? at\b|\bauthor(s)?\b of|\baffiliat\w*\b|\bconnect(s|ed)? to\b",
    re.IGNORECASE,
)
_WEB_PATTERNS = re.compile(
    r"\blatest\b|\brecent(ly)?\b|\btoday\b|\bcurrent(ly)?\b|\bnews\b|\b202[4-9]\b",
    re.IGNORECASE,
)
_API_PATTERNS = re.compile(
    r"\barxiv\b|\bdoi\b|\bcitation(s)?\b|"
    # bounded distance, either word order: "paper ... published" or "published ... paper"
    r"\bpaper\b.{0,40}\b(publish\w*|date)\b|\b(publish\w*|date)\b.{0,40}\bpaper\b|"
    r"\blook ?up\b.{0,40}\bpaper\b",
    re.IGNORECASE,
)


def rule_route(question: str) -> str:
    """Return one of `ROUTES`. Checked in a fixed priority order because a
    question can plausibly match more than one pattern (e.g. "who published
    this paper?" matches both graph and api heuristics) — order encodes
    which backend is the better default when that happens.
    """
    if _API_PATTERNS.search(question):
        return "api"
    if _WEB_PATTERNS.search(question):
        return "web"
    if _SQL_PATTERNS.search(question):
        return "sql"
    if _GRAPH_PATTERNS.search(question):
        return "graph"
    return "documents"
