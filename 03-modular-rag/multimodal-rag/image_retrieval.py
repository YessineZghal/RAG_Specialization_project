"""Image retrieval over the PDF.

Extracts the real embedded images (via `pymupdf`) and saves them to
`data/pdfs/images/`, then indexes each by its **figure caption** — the
same "find 'Figure N:' in the clean pypdf text" trick used in
`table_retrieval.py`. This is caption-based retrieval, not true visual
embedding: a real multimodal system would embed the image pixels
themselves with a vision-language model (e.g. CLIP) so retrieval works
even for images with no caption text at all — worth calling out as this
level's one genuine simplification, not a hidden one.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modular_common.config import settings
from modular_common.pdf import load_pages

CAPTION_RE = re.compile(r"\bFigure\s+(\d+)\s*:")
WINDOW_CHARS = 300


def extract_captions(pages: list[str] | None = None) -> dict[int, str]:
    """{figure_number: caption_text}, keyed by the FIRST (real) occurrence.

    `pages` defaults to the real PDF's pages but accepts a plain list of
    page-text strings for offline testing (see `extract_tables()`).
    """
    captions: dict[int, str] = {}
    for text in pages if pages is not None else load_pages():
        for match in CAPTION_RE.finditer(text):
            n = int(match.group(1))
            if n in captions:
                continue
            window = text[match.start() : match.start() + WINDOW_CHARS]
            captions[n] = " ".join(window.split())
    return captions


def extract_images() -> dict[str, dict]:
    """Save every embedded image to disk; return
    {"page{P}-img{I}": {"path": ..., "page": P, "caption": ...}}.

    Figures are matched to images by page number — the paper's figures
    each land on one page, and pypdf's caption search already told us
    which page each "Figure N:" caption is on.
    """
    import pymupdf

    settings.images_dir.mkdir(parents=True, exist_ok=True)
    captions = extract_captions()
    caption_by_page: dict[int, str] = {}
    for page_text_num, text in enumerate(load_pages(), start=1):
        for match in CAPTION_RE.finditer(text):
            caption_by_page.setdefault(page_text_num, captions[int(match.group(1))])

    images: dict[str, dict] = {}
    doc = pymupdf.open(str(settings.pdf_path))
    for page_index, page in enumerate(doc):
        page_num = page_index + 1
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            pixmap = pymupdf.Pixmap(doc, xref)
            if pixmap.n - pixmap.alpha > 3:  # CMYK etc. -> convert to RGB first
                pixmap = pymupdf.Pixmap(pymupdf.csRGB, pixmap)
            image_id = f"page{page_num}-img{img_index}"
            out_path = settings.images_dir / f"{image_id}.png"
            pixmap.save(str(out_path))
            images[image_id] = {
                "path": str(out_path),
                "page": page_num,
                "caption": caption_by_page.get(page_num, f"(uncaptioned image on page {page_num})"),
            }
    doc.close()
    return images


def build_image_retriever(embedder=None):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "multi-retriever"))
    from vector_retriever import VectorRetriever

    images = extract_images()
    texts = {image_id: img["caption"] for image_id, img in images.items()}
    retriever = VectorRetriever.from_texts(texts, embedder=embedder, cache_name="images")
    return retriever, images
