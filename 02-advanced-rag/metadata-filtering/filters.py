"""Metadata filtering — narrow retrieval results using structured fields a
raw embedding/BM25 score has no way to express (e.g. "only recent docs",
"only from this department").

The brute-force in-memory retrievers in this repo can only **post-filter**:
retrieve a wide candidate set, then drop anything that fails the predicate,
then truncate to `top_k`. A real vector database (Qdrant, per this repo's
stack) supports **pre-filtering** at the index level, which is both faster
and doesn't risk under-filling `top_k` the way post-filtering can if too
many candidates get dropped — see `07-production-rag/retrieval-infrastructure/`.

scifact ships no rich metadata, so `build_length_metadata` derives a
synthetic field (document length bucket) purely to make filtering
demonstrable end-to-end — swap in real metadata (date, source, department)
in any other corpus.
"""

from __future__ import annotations

from collections.abc import Callable


def build_length_metadata(corpus: dict[str, dict]) -> dict[str, dict]:
    metadata = {}
    for doc_id, doc in corpus.items():
        word_count = len(doc["text"].split())
        bucket = "short" if word_count < 150 else "medium" if word_count < 250 else "long"
        metadata[doc_id] = {"word_count": word_count, "length_bucket": bucket}
    return metadata


def filter_by_metadata(
    candidates: list[tuple[str, float]],
    metadata: dict[str, dict],
    predicate: Callable[[dict], bool],
) -> list[tuple[str, float]]:
    return [(doc_id, score) for doc_id, score in candidates if predicate(metadata.get(doc_id, {}))]


def filtered_search(
    retriever,
    query: str,
    metadata: dict[str, dict],
    predicate: Callable[[dict], bool],
    top_k: int = 10,
    candidate_k: int = 50,
) -> list[tuple[str, float]]:
    """Retrieve wide (`candidate_k`), post-filter, then truncate to `top_k`."""
    candidates = retriever.search(query, top_k=candidate_k)
    return filter_by_metadata(candidates, metadata, predicate)[:top_k]
