"""Table-aware retrieval over the PDF.

pdfplumber's structured cell parser turns out to badly mangle this paper's
tables (merged cells, no visible rules — a genuinely common real-world PDF
table-extraction problem, not a bug in this code). The far more reliable
signal already at hand: every real table in an academic paper has a
"Table N: <caption>" line immediately followed by its data, and `pypdf`'s
plain text extraction (used everywhere else in this level) renders that
cleanly. So: find captions, take the text window right after each one —
real table content, without fighting a parser that can't handle this
PDF's layout.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modular_common.pdf import load_pages  # noqa: E402

CAPTION_RE = re.compile(r"\bTable\s+(\d+)\s*:")
WINDOW_CHARS = 500


def extract_tables(pages: list[str] | None = None) -> dict[str, dict]:
    """Return {"table-N": {"page": int, "text": "Table N: ... <data window>"}}.

    `pages` defaults to the real PDF's pages (`common.pdf.load_pages()`) but
    accepts a plain list of page-text strings so this logic can be unit
    tested without downloading anything.
    """
    tables: dict[str, dict] = {}
    for page_num, page_text in enumerate(pages if pages is not None else load_pages(), start=1):
        for match in CAPTION_RE.finditer(page_text):
            table_id = f"table-{match.group(1)}"
            if table_id in tables:
                continue  # keep the first (real) occurrence; later ones are in-text references
            window = page_text[match.start() : match.start() + WINDOW_CHARS]
            tables[table_id] = {"page": page_num, "text": " ".join(window.split())}
    return tables


def build_table_retriever(embedder=None):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "multi-retriever"))
    from vector_retriever import VectorRetriever

    tables = extract_tables()
    texts = {table_id: t["text"] for table_id, t in tables.items()}
    retriever = VectorRetriever.from_texts(texts, embedder=embedder, cache_name="tables")
    return retriever, tables
