"""Load raw documents into the pipeline's common `Document` shape.

Two sources are supported:

1. `load_from_hf_dataset` — an **open-source** Hugging Face dataset,
   downloaded and cached by the `datasets` library the first time it runs.
   Default: `rag-datasets/rag-mini-wikipedia` (text-corpus/passages config),
   a small, public-domain-derived Wikipedia passage set built specifically
   for RAG tutorials. Nothing is bundled in this repo — see README.md#dataset.

2. `load_from_directory` — plain `.txt`/`.md` files and `.pdf` files from a
   local folder (e.g. `data/sample_docs/`), for the "local PDF chatbot"
   mini project.

Both return a flat `list[Document]` so the rest of the pipeline never has
to know where a document came from.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pypdf import PdfReader

from .config import settings
from .schema import Document

logger = logging.getLogger(__name__)

TEXT_SUFFIXES = {".txt", ".md"}


def load_from_hf_dataset(
    dataset_name: str | None = None,
    config: str | None = None,
    split: str | None = None,
    limit: int | None = None,
) -> list[Document]:
    """Download (and cache) an open-source HF dataset and adapt it to `Document`.

    This is the only place in Level 1 that touches the network. It is never
    called at import time — only when a script or notebook explicitly runs
    it, and results are cached locally by `datasets` (in `~/.cache/huggingface`)
    so subsequent runs are offline.
    """
    from datasets import load_dataset  # imported lazily: heavy, network-capable

    dataset_name = dataset_name or settings.hf_dataset_name
    config = config or settings.hf_dataset_config
    split = split or settings.hf_dataset_split

    logger.info("Loading %s (%s/%s) from Hugging Face...", dataset_name, config, split)
    ds = load_dataset(dataset_name, config, split=split)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    documents: list[Document] = []
    for row in ds:
        # rag-mini-wikipedia's text-corpus config exposes `id` and `passage`.
        # Fall back gracefully if a different dataset is swapped in.
        doc_id = str(row.get("id", len(documents)))
        text = row.get("passage") or row.get("text") or row.get("content") or ""
        if not text.strip():
            continue
        documents.append(
            Document(
                id=f"hf-{doc_id}",
                text=text.strip(),
                metadata={"source": dataset_name, "split": split},
            )
        )

    logger.info("Loaded %d documents from %s.", len(documents), dataset_name)
    return documents


def load_from_directory(directory: Path | str | None = None) -> list[Document]:
    """Load local `.txt`, `.md`, and `.pdf` files as `Document`s.

    Used by the "local PDF chatbot" mini project and by the offline test
    suite (against the small hand-written files in `data/sample_docs/`).
    """
    directory = Path(directory) if directory is not None else settings.sample_docs_dir
    documents: list[Document] = []

    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue

        if path.suffix.lower() in TEXT_SUFFIXES:
            text = path.read_text(encoding="utf-8")
        elif path.suffix.lower() == ".pdf":
            text = _read_pdf(path)
        else:
            continue

        if not text.strip():
            continue

        documents.append(
            Document(
                id=path.stem,
                text=text.strip(),
                metadata={"source": str(path.relative_to(directory.parent))},
            )
        )

    logger.info("Loaded %d local documents from %s.", len(documents), directory)
    return documents


def _read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)
