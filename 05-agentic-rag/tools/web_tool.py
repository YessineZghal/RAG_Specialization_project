"""`web_search(query)` — live web search (`ddgs`, no API key), for
questions the local corpus can't answer at all (e.g. asking about a
topic outside the 50 sampled TriviaQA articles). Same mechanism as
Level 3's `web-rag/`, reimplemented here to keep this level self-contained.
"""

from __future__ import annotations


class WebTool:
    name = "web_search"
    description = "Search the live web for information not in the local corpus."

    def __call__(self, query: str, max_results: int = 3) -> list[dict]:
        from ddgs import DDGS

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        return [
            {"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")}
            for r in results
        ]
