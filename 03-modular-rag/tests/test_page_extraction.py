"""page_extraction.py tests with a canned HTML response (no live network)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "web-rag"))
import page_extraction  # noqa: E402

SAMPLE_HTML = """
<html>
<head><title>Test</title><style>.x{color:red}</style></head>
<body>
<nav>Navigation link</nav>
<header>Site header</header>
<main>
  <h1>Real Title</h1>
  <p>This is the actual article content that should be extracted.</p>
  <script>console.log('should not appear')</script>
</main>
<footer>Copyright footer</footer>
</body>
</html>
"""


class FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text

    def raise_for_status(self) -> None:
        pass


def test_fetch_page_text_strips_boilerplate(monkeypatch):
    monkeypatch.setattr("requests.get", lambda *a, **k: FakeResponse(SAMPLE_HTML))

    text = page_extraction.fetch_page_text("https://example.com")

    assert "actual article content" in text
    assert "should not appear" not in text  # <script> stripped
    assert "Navigation link" not in text  # <nav> stripped
    assert "Copyright footer" not in text  # <footer> stripped


def test_fetch_page_text_truncates_to_max_chars(monkeypatch):
    monkeypatch.setattr("requests.get", lambda *a, **k: FakeResponse("<p>" + "word " * 5000 + "</p>"))

    text = page_extraction.fetch_page_text("https://example.com", max_chars=100)

    assert len(text) == 100


def test_search_and_extract_handles_fetch_failures_gracefully(monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "web-rag"))
    import search as search_module

    monkeypatch.setattr(
        search_module,
        "web_search",
        lambda query, max_results=5: [{"title": "T", "url": "https://bad.example", "snippet": "s"}],
    )

    def broken_fetch(url, timeout=10, max_chars=4000):
        raise ConnectionError("simulated network failure")

    monkeypatch.setattr(page_extraction, "fetch_page_text", broken_fetch)

    results = page_extraction.search_and_extract("anything")

    assert results[0]["text"] == ""
    assert "error" in results[0]
