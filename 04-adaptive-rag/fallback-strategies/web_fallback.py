"""Fall back to live web search when the local corpus can't answer a
question with confidence — e.g. after `corrective-rag/crag.py` marks the
retrieved evidence as not trustworthy.

Same live search as `03-modular-rag/web-rag/search.py` (`ddgs`, no API
key), reimplemented here so this level stays self-contained.
"""

from __future__ import annotations


def web_fallback_search(query: str, max_results: int = 3) -> list[dict]:
    from ddgs import DDGS

    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))
    return [
        {"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")}
        for r in results
    ]


def web_fallback_context(query: str, max_results: int = 3) -> str:
    results = web_fallback_search(query, max_results=max_results)
    return "\n\n".join(f"[{r['title']}] {r['snippet']}" for r in results)
