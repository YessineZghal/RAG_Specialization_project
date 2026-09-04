"""Fixed-size chunking — the Level 1 baseline, word-based.

Identical strategy to `01-naive-rag/src/chunk.py`, reimplemented here
(rather than imported) so each level stays self-contained. Included as the
control group for the chunking-strategy comparison in
`notebooks/01_chunking_strategies.ipynb` — everything else in this folder
exists to improve on it.
"""

from __future__ import annotations


def fixed_size_chunk(text: str, chunk_size: int = 200, chunk_overlap: int = 20) -> list[str]:
    """Split `text` into overlapping windows of `chunk_size` words."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and < chunk_size")

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
