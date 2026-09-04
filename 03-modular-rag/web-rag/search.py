"""Real, live web search via DuckDuckGo (`ddgs` — no API key required).

This is the one backend in Level 3 that can answer questions about
anything *after* the PDF's publication date, or anything not in any of
this level's static data sources at all.
"""

from __future__ import annotations


def web_search(query: str, max_results: int = 5) -> list[dict]:
    """Return `[{"title", "url", "snippet"}, ...]`."""
    from ddgs import DDGS

    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))
    return [
        {"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")}
        for r in results
    ]
