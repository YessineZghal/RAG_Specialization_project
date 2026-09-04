"""Call a real external API as a retrieval "tool" — arXiv's public search
API (no key required), used to look up paper metadata: authors, abstract,
publication date, direct link.

This is the API-RAG pattern in miniature: instead of retrieving from a
local index, the answer comes from calling a structured, external service
and formatting its response as context.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

ARXIV_API_URL = "http://export.arxiv.org/api/query"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


def search_arxiv(query: str, max_results: int = 3) -> list[dict]:
    """Search arXiv by keywords or an arXiv id (e.g. `id:1706.03762`)."""
    import requests

    params = {"search_query": query, "max_results": max_results}
    response = requests.get(ARXIV_API_URL, params=params, timeout=15)
    response.raise_for_status()

    root = ET.fromstring(response.text)
    papers = []
    for entry in root.findall("atom:entry", ATOM_NS):
        papers.append(
            {
                "title": _text(entry, "atom:title"),
                "summary": _text(entry, "atom:summary"),
                "published": _text(entry, "atom:published"),
                "authors": [
                    _text(author, "atom:name") for author in entry.findall("atom:author", ATOM_NS)
                ],
                "link": _text(entry, "atom:id"),
            }
        )
    return papers


def _text(element: ET.Element, path: str) -> str:
    found = element.find(path, ATOM_NS)
    return " ".join(found.text.split()) if found is not None and found.text else ""


def lookup_paper_by_id(arxiv_id: str) -> dict | None:
    results = search_arxiv(f"id:{arxiv_id}", max_results=1)
    return results[0] if results else None
