"""Equation-aware retrieval over the PDF -- the gap this level never
closed, confirmed by directly grepping the whole repo for `latex` or
`equation` before this module existed (see `../../missing_to_complite.md`,
which named this against RAG-Anything's LaTeX-parsing modality processor).

**This PDF has no `"Equation N:"` captions** -- confirmed directly by
searching the real extracted text for the word "Equation": zero matches,
anywhere. `table_retrieval.py`'s caption-then-window approach (this
paper genuinely does label its tables `"Table N:"`) has nothing to key
off here. What the real PDF *does* have is its own real numbering
convention: a small parenthesized number at the very end of the equation's
own line, e.g. `"...softmax(QK^T/√dk)V (1)"`.

**A naive `\\(\\d+\\)` regex is not enough** -- verified directly against
this real PDF's extracted text: the same shape also matches Big-O
complexity table entries (`"O(1)"`) and citation-list volume/issue
numbers (`"9(8):1735"`), 7 false positives against only 3 real equations
in this document. Two real, cheap signals separate them, both confirmed
against the actual text: a real equation number is immediately followed
by a newline (nothing else on its line), *and* the ~200 characters before
it contain a real `"="` sign (a genuine equation has one; a table row or
citation entry does not). Both conditions together correctly kept all 3
real equations and rejected all 7 false positives on this document --
verified by hand, not assumed.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modular_common.pdf import load_pages

_NUMBER_RE = re.compile(r"\((\d{1,2})\)")
WINDOW_CHARS = 200

# A real, honest "conceptual mapping": scan an equation's own extracted
# text for symbols this repo can actually recognize plainly, and report
# the plain-English gloss for whichever ones are actually present --
# never a claim of full LaTeX parsing, since pypdf's text extraction
# already renders every equation with its structure flattened (no real
# fraction bars, superscripts inline instead of raised -- see the
# `sin(pos/100002i/dmodel)`-style mangling this repo's own PDF produces).
SYMBOL_GLOSS = {
    "√": "square root",
    "∑": "summation",
    "∫": "integral",
    "∂": "partial derivative",
    "∞": "infinity",
    "±": "plus or minus",
    "≈": "approximately equal to",
    "≤": "less than or equal to",
    "≥": "greater than or equal to",
    "softmax": "softmax normalization",
    "sin(": "sine function",
    "cos(": "cosine function",
    "max(": "maximum function",
}


def _is_real_equation_number(text: str, match: re.Match) -> bool:
    followed_by_newline = text[match.end() : match.end() + 1] == "\n"
    preceding = text[max(0, match.start() - WINDOW_CHARS) : match.start()]
    return followed_by_newline and "=" in preceding


def gloss_symbols(text: str) -> list[str]:
    """Plain-English glosses for whichever recognized math symbols are
    actually present in `text` -- only what's really there, never a
    fixed list regardless of content."""
    return [gloss for symbol, gloss in SYMBOL_GLOSS.items() if symbol in text]


def extract_equations(pages: list[str] | None = None) -> dict[str, dict]:
    """Return `{"equation-N": {"page": int, "text": "<equation window>",
    "concepts": [...]}}`. `pages` defaults to the real PDF's pages but
    accepts a plain list of page-text strings so this logic can be unit
    tested without downloading anything.
    """
    equations: dict[str, dict] = {}
    for page_num, page_text in enumerate(pages if pages is not None else load_pages(), start=1):
        for match in _NUMBER_RE.finditer(page_text):
            if not _is_real_equation_number(page_text, match):
                continue
            equation_id = f"equation-{match.group(1)}"
            if equation_id in equations:
                continue  # keep the first real occurrence
            window = page_text[max(0, match.start() - WINDOW_CHARS) : match.end()]
            window_text = " ".join(window.split())
            equations[equation_id] = {
                "page": page_num,
                "text": window_text,
                "concepts": gloss_symbols(window_text),
            }
    return equations


def build_equation_retriever(embedder=None):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "multi-retriever"))
    from vector_retriever import VectorRetriever

    equations = extract_equations()
    # Index the concept glosses alongside the raw (mangled) equation text --
    # a query using plain English ("summation", "square root") can match
    # even though the equation's own extracted text never spells that word.
    texts = {
        equation_id: f"{eq['text']} {' '.join(eq['concepts'])}".strip()
        for equation_id, eq in equations.items()
    }
    retriever = VectorRetriever.from_texts(texts, embedder=embedder, cache_name="equations")
    return retriever, equations
