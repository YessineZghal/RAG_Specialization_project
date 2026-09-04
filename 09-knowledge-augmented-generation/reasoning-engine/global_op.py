"""The `global` operator: answer from community-level summaries instead
of specific entity facts -- the retrieval counterpart to
`indexing/community_summary.py`'s generation step. Embeds the question
and every community summary, and returns the most relevant ones as
evidence, the same brute-force cosine approach every other retrieval in
this repo uses (no new retrieval mechanism, just a different, coarser
corpus to search over: summaries instead of documents or graph facts).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GlobalEvidence:
    summaries: list[str]
    community_ids: list[str]


def answer_from_communities(
    question: str,
    community_summaries: dict[str, dict],
    embedder,
    top_k: int = 2,
) -> GlobalEvidence:
    """`community_summaries` is `indexing.community_summary.build_community_summaries`'s
    output. Returns the `top_k` most relevant summaries for `question`,
    or empty evidence if there are no communities to search at all
    (a graph too small or too fragmented to have any community of size
    >= `MIN_COMMUNITY_SIZE` is a real, possible outcome, not an error)."""
    if not community_summaries:
        return GlobalEvidence(summaries=[], community_ids=[])

    community_ids = list(community_summaries.keys())
    summary_texts = [community_summaries[cid]["summary"] for cid in community_ids]

    query_vector = np.asarray(embedder.embed_one(question), dtype=np.float32)
    summary_vectors = np.asarray(embedder.embed_many(summary_texts), dtype=np.float32)

    norms = np.linalg.norm(summary_vectors, axis=1) + 1e-12
    query_norm = np.linalg.norm(query_vector) + 1e-12
    scores = (summary_vectors @ query_vector) / (norms * query_norm)

    k = min(top_k, len(community_ids))
    top_indices = np.argsort(-scores)[:k]

    return GlobalEvidence(
        summaries=[summary_texts[i] for i in top_indices],
        community_ids=[community_ids[i] for i in top_indices],
    )
