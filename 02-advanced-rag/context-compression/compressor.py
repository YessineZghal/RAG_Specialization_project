"""Context compression — trim retrieved chunks down to just the sentences
that are actually relevant to the query before handing them to the LLM.

Retrieved chunks are relevant on average, not sentence-by-sentence — a
500-word chunk that matched a query might have only one sentence that
actually answers it. Passing the whole chunk wastes context budget and can
dilute the LLM's attention; compression re-scores at the sentence level
and keeps only the top fraction.
"""

from __future__ import annotations

import sys
from pathlib import Path
from collections.abc import Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chunking.semantic import split_sentences  # noqa: E402


def compress_context(
    query: str,
    texts: list[str],
    embed_fn: Callable[[str], list[float]],
    keep_ratio: float = 0.5,
    min_sentences: int = 1,
) -> str:
    """Return the most query-relevant sentences from `texts`, joined back
    into one string, in their original relative order.
    """
    sentences: list[str] = []
    for text in texts:
        sentences.extend(split_sentences(text))
    if not sentences:
        return ""

    query_vec = np.array(embed_fn(query))
    query_norm = np.linalg.norm(query_vec) + 1e-12

    scored = []
    for sentence in sentences:
        sentence_vec = np.array(embed_fn(sentence))
        similarity = float(
            np.dot(sentence_vec, query_vec) / (np.linalg.norm(sentence_vec) * query_norm + 1e-12)
        )
        scored.append((similarity, sentence))

    keep_n = max(min_sentences, round(len(scored) * keep_ratio))
    kept_sentences = {s for _, s in sorted(scored, key=lambda x: -x[0])[:keep_n]}

    # Preserve original reading order among the kept sentences.
    return " ".join(s for s in sentences if s in kept_sentences)
