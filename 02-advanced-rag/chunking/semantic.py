"""Semantic chunking — split where *meaning* shifts, not where a word/char
count runs out.

Method (a simplified version of the "percentile breakpoint" technique
popularized by Greg Kamradt): embed every sentence, measure the semantic
*distance* (1 - cosine similarity) between each consecutive pair, and cut
a new chunk boundary wherever that distance is unusually large relative to
the rest of the document (above the given percentile). Requires an
embedding function — pass `common.embed.OllamaEmbedder().embed_one`, or
any `Callable[[str], list[float]]` (a fake one is used in tests).
"""

from __future__ import annotations

import re
from collections.abc import Callable

import numpy as np

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text.strip()) if s.strip()]
    return sentences


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def semantic_chunk(
    text: str,
    embed_fn: Callable[[str], list[float]],
    breakpoint_percentile: float = 90.0,
    min_sentences_per_chunk: int = 1,
) -> list[str]:
    sentences = split_sentences(text)
    if len(sentences) <= max(1, min_sentences_per_chunk):
        return [text.strip()] if text.strip() else []

    embeddings = np.array([embed_fn(s) for s in sentences])
    distances = [
        1.0 - _cosine(embeddings[i], embeddings[i + 1]) for i in range(len(embeddings) - 1)
    ]
    threshold = float(np.percentile(distances, breakpoint_percentile))

    chunks: list[str] = []
    current = [sentences[0]]
    for i, distance in enumerate(distances):
        if distance > threshold and len(current) >= min_sentences_per_chunk:
            chunks.append(" ".join(current))
            current = []
        current.append(sentences[i + 1])
    if current:
        chunks.append(" ".join(current))
    return chunks
