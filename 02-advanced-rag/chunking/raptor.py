"""RAPTOR — Recursive Abstractive Processing for Tree-Organized Retrieval.

`hierarchical.py` groups chunks by their *position* in a document (the
first 120 words, the next 120 words, ...). RAPTOR groups them by
*meaning* instead: chunks that are semantically similar are clustered
together, each cluster is summarized by the LLM into one higher-level
node, and the process repeats on those summaries — producing a genuine
tree of increasingly abstract summaries above the original chunks.

Retrieval then searches *every level of the tree at once* (leaf chunks
and every summary above them), so a broad question can match a top-level
summary while a narrow, specific question can still match an original
leaf chunk — the same corpus, searchable at whichever granularity the
question actually needs.

**Disclosed simplification**: the original RAPTOR paper clusters with a
Gaussian Mixture Model over UMAP-reduced embeddings. This module clusters
with a much simpler greedy similarity-threshold pass (see
`cluster_by_similarity`) — real clustering, not a stand-in, just a
simpler algorithm than the paper's. It produces the same kind of result
(similar chunks grouped together) without adding a new dependency this
repo doesn't already have (`numpy` is the only import here).

Requires an embedding function and a summarization function — pass
`common.embed.OllamaEmbedder().embed_one` and a small wrapper around
`common.llm.OllamaLLM().complete`, or fakes (as used in tests).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np


@dataclass
class RaptorNode:
    text: str  # original chunk text (leaf) or an LLM-written summary (internal node)
    level: int  # 0 = leaf; each level up is one round of clustering + summarization
    children: list["RaptorNode"] = field(default_factory=list)

    @property
    def is_leaf(self) -> bool:
        return not self.children


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def cluster_by_similarity(vectors: np.ndarray, similarity_threshold: float = 0.5) -> list[list[int]]:
    """Greedily group vectors into clusters: each vector joins the
    existing cluster whose running centroid it is most similar to, if
    that similarity clears `similarity_threshold`; otherwise it starts a
    new cluster of its own. Deterministic for a fixed input order.

    Returns a list of clusters, each a list of indices into `vectors`.
    """
    clusters: list[list[int]] = []
    centroids: list[np.ndarray] = []

    for i, vector in enumerate(vectors):
        best_cluster_idx, best_similarity = None, -1.0
        for cluster_idx, centroid in enumerate(centroids):
            similarity = _cosine(vector, centroid)
            if similarity > best_similarity:
                best_cluster_idx, best_similarity = cluster_idx, similarity

        if best_cluster_idx is not None and best_similarity >= similarity_threshold:
            clusters[best_cluster_idx].append(i)
            member_vectors = vectors[clusters[best_cluster_idx]]
            centroids[best_cluster_idx] = member_vectors.mean(axis=0)
        else:
            clusters.append([i])
            centroids.append(vector)

    return clusters


def build_raptor_tree(
    leaf_texts: list[str],
    embed_fn: Callable[[str], list[float]],
    summarize_fn: Callable[[list[str]], str],
    similarity_threshold: float = 0.5,
    max_levels: int = 3,
) -> RaptorNode:
    """Build the tree bottom-up: cluster the current level's nodes by
    embedding similarity, summarize each cluster of two or more into one
    parent node, carry any singleton cluster up unchanged, and repeat on
    the new (shorter) list of nodes until only one remains, a level limit
    is hit, or a round produces no merges at all (every node stayed its
    own cluster — clustering has nothing left to do, so stop instead of
    looping forever).
    """
    if not leaf_texts:
        raise ValueError("leaf_texts must not be empty")

    current_nodes = [RaptorNode(text=text, level=0) for text in leaf_texts]
    level = 0

    while len(current_nodes) > 1 and level < max_levels:
        vectors = np.array([embed_fn(node.text) for node in current_nodes])
        clusters = cluster_by_similarity(vectors, similarity_threshold=similarity_threshold)

        if len(clusters) == len(current_nodes):
            break  # nothing merged this round -- further rounds would not help either

        next_level_nodes: list[RaptorNode] = []
        for member_indices in clusters:
            members = [current_nodes[i] for i in member_indices]
            if len(members) == 1:
                next_level_nodes.append(members[0])
                continue
            summary = summarize_fn([member.text for member in members])
            next_level_nodes.append(RaptorNode(text=summary, level=level + 1, children=members))

        current_nodes = next_level_nodes
        level += 1

    if len(current_nodes) == 1:
        return current_nodes[0]

    # More than one node survived (hit max_levels, or clustering stalled)
    # -- wrap whatever is left under one synthetic root so the tree always
    # has a single entry point, without inventing a fake LLM summary for it.
    combined_text = " ".join(node.text for node in current_nodes)
    return RaptorNode(text=combined_text, level=level + 1, children=current_nodes)


def flatten_tree(root: RaptorNode) -> list[RaptorNode]:
    """Every node in the tree, root first, depth-first — what gets
    searched at query time (see `raptor_search`).
    """
    nodes = [root]
    for child in root.children:
        nodes.extend(flatten_tree(child))
    return nodes


def raptor_search(
    query: str,
    root: RaptorNode,
    embed_fn: Callable[[str], list[float]],
    top_k: int = 5,
) -> list[tuple[RaptorNode, float]]:
    """Search every node in the tree at once (leaves and summaries alike)
    and return the `top_k` best matches, highest similarity first.
    """
    nodes = flatten_tree(root)
    query_vector = np.array(embed_fn(query))
    scored = [(node, _cosine(query_vector, np.array(embed_fn(node.text)))) for node in nodes]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:top_k]
