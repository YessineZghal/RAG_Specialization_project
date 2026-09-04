"""Fetch a real URL and extract its main readable text.

Deliberately simple (strip script/style/nav/footer, concatenate the rest)
rather than a full readability algorithm — good enough to turn a search
result into usable context, not a general-purpose web scraper.
"""

from __future__ import annotations


def fetch_page_text(url: str, timeout: int = 10, max_chars: int = 4000) -> str:
    import requests
    from bs4 import BeautifulSoup

    response = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    text = " ".join(soup.get_text(separator=" ").split())
    return text[:max_chars]


def search_and_extract(query: str, max_results: int = 3) -> list[dict]:
    """Search, then fetch+extract each result's page text — the full
    web-RAG retrieval step in one call.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from search import web_search

    results = web_search(query, max_results=max_results)
    for result in results:
        try:
            result["text"] = fetch_page_text(result["url"])
        except Exception as exc:  # noqa: BLE001 - a dead/blocked link shouldn't crash retrieval
            result["text"] = ""
            result["error"] = str(exc)
    return results
