"""Naive fixed-size chunking.

Level 1 deliberately uses the simplest possible strategy — split on
whitespace into fixed-size, overlapping word windows — so its weaknesses
(splitting a fact across two chunks, ignoring sentence/paragraph
boundaries) are visible and motivate Level 2's recursive/semantic/
parent-child chunkers. See ../theory/chunking.md for the full discussion.
"""

from __future__ import annotations

from .config import settings
from .schema import Chunk, Document


def chunk_text(
    text: str,
    chunk_size: int = settings.chunk_size,
    chunk_overlap: int = settings.chunk_overlap,
) -> list[str]:
    """Split `text` into overlapping windows of `chunk_size` words.

    `chunk_overlap` words from the end of each window are repeated at the
    start of the next one, so a fact sitting on a window boundary still has
    a chance of appearing whole in at least one chunk.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and < chunk_size")

    words = text.split()
    if not words:
        return []

    stride = chunk_size - chunk_overlap
    chunks: list[str] = []
    for start in range(0, len(words), stride):
        window = words[start : start + chunk_size]
        if not window:
            break
        chunks.append(" ".join(window))
        if start + chunk_size >= len(words):
            break
    return chunks


def chunk_document(
    document: Document,
    chunk_size: int = settings.chunk_size,
    chunk_overlap: int = settings.chunk_overlap,
) -> list[Chunk]:
    """Chunk a single `Document`, preserving traceability back to it."""
    texts = chunk_text(document.text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return [
        Chunk(
            id=f"{document.id}::chunk-{position}",
            text=text,
            document_id=document.id,
            position=position,
            metadata=dict(document.metadata),
        )
        for position, text in enumerate(texts)
    ]


def chunk_documents(
    documents: list[Document],
    chunk_size: int = settings.chunk_size,
    chunk_overlap: int = settings.chunk_overlap,
) -> list[Chunk]:
    """Chunk a list of documents into a single flat list of chunks."""
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(
            chunk_document(document, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        )
    return chunks
