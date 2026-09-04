from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "reasoning-engine"))
from global_op import answer_from_communities  # noqa: E402


class FakeEmbedder:
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors

    def embed_one(self, text):
        return self.vectors[text]

    def embed_many(self, texts):
        return [self.vectors[t] for t in texts]


def test_answer_from_communities_returns_the_most_similar_summary_first():
    summaries = {
        "community-0": {"nodes": ["a", "b"], "summary": "diabetes research"},
        "community-1": {"nodes": ["c", "d"], "summary": "orthopedic surgery outcomes"},
    }
    embedder = FakeEmbedder({
        "what are the diabetes findings?": [1.0, 0.0],
        "diabetes research": [1.0, 0.0],
        "orthopedic surgery outcomes": [0.0, 1.0],
    })

    evidence = answer_from_communities("what are the diabetes findings?", summaries, embedder, top_k=1)

    assert evidence.summaries == ["diabetes research"]
    assert evidence.community_ids == ["community-0"]


def test_answer_from_communities_respects_top_k():
    summaries = {f"community-{i}": {"nodes": [], "summary": f"summary {i}"} for i in range(5)}
    embedder = FakeEmbedder({
        "question": [1.0, 0.0],
        **{f"summary {i}": [1.0, 0.0] for i in range(5)},
    })

    evidence = answer_from_communities("question", summaries, embedder, top_k=2)

    assert len(evidence.summaries) == 2
    assert len(evidence.community_ids) == 2


def test_answer_from_communities_on_no_communities_returns_empty_evidence():
    embedder = FakeEmbedder({})
    evidence = answer_from_communities("question", {}, embedder)
    assert evidence.summaries == []
    assert evidence.community_ids == []
