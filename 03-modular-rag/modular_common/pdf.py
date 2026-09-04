"""Load the open-access PDF used as this level's "documents" backend.

Google explicitly grants reproduction rights for this paper's tables and
figures "for use in journalistic or scholarly works" (see page 1 of the
PDF itself) — which is exactly the use this repository makes of it.
Nothing is bundled: `ensure_pdf()` downloads it once, on first use.
"""

from __future__ import annotations

import logging

from .config import settings

logger = logging.getLogger(__name__)


def ensure_pdf() -> None:
    """Download the PDF to `data/pdfs/` if it isn't already there."""
    if settings.pdf_path.exists():
        return
    import requests

    logger.info("Downloading %s -> %s", settings.pdf_url, settings.pdf_path)
    settings.pdf_path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(settings.pdf_url, timeout=30)
    response.raise_for_status()
    settings.pdf_path.write_bytes(response.content)


def load_pages() -> list[str]:
    """Return one string of extracted text per PDF page."""
    ensure_pdf()
    from pypdf import PdfReader

    reader = PdfReader(str(settings.pdf_path))
    return [page.extract_text() or "" for page in reader.pages]


def chunk_text(text: str, chunk_size: int = 150, chunk_overlap: int = 20) -> list[str]:
    """Naive fixed-size word chunking — same strategy as Level 1, reused
    here because Level 3's focus is routing between backends, not chunking
    strategy (see Level 2 for that comparison).
    """
    words = text.split()
    if not words:
        return []
    stride = chunk_size - chunk_overlap
    chunks = []
    for start in range(0, len(words), stride):
        window = words[start : start + chunk_size]
        if not window:
            break
        chunks.append(" ".join(window))
        if start + chunk_size >= len(words):
            break
    return chunks


def load_chunks(chunk_size: int = 150, chunk_overlap: int = 20) -> dict[str, dict]:
    """Return {chunk_id: {"text": ..., "page": ...}} for the whole PDF."""
    pages = load_pages()
    chunks: dict[str, dict] = {}
    for page_num, page_text in enumerate(pages, start=1):
        for i, chunk in enumerate(chunk_text(page_text, chunk_size, chunk_overlap)):
            chunks[f"p{page_num}-c{i}"] = {"text": chunk, "page": page_num}
    return chunks
