from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chunking.raptor import (
    RaptorNode,
    build_raptor_tree,
    cluster_by_similarity,
    flatten_tree,
    raptor_search,
)


def test_cluster_by_similarity_groups_identical_vectors_and_separates_orthogonal_ones():
    vectors = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    clusters = cluster_by_similarity(vectors, similarity_threshold=0.9)
    assert clusters == [[0, 1], [2, 3]]


def test_cluster_by_similarity_with_a_strict_threshold_keeps_everything_separate():
    vectors = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    clusters = cluster_by_similarity(vectors, similarity_threshold=0.999)
    assert clusters == [[0], [1], [2]]


def test_flatten_tree_returns_root_first_then_children_depth_first():
    leaf_a = RaptorNode(text="a", level=0)
    leaf_b = RaptorNode(text="b", level=0)
    parent = RaptorNode(text="summary", level=1, children=[leaf_a, leaf_b])
    assert flatten_tree(parent) == [parent, leaf_a, leaf_b]


def test_build_raptor_tree_rejects_empty_input():
    with pytest.raises(ValueError):
        build_raptor_tree([], embed_fn=lambda t: [0.0], summarize_fn=lambda texts: "")


def test_build_raptor_tree_with_a_single_leaf_returns_that_leaf_unchanged():
    root = build_raptor_tree(["only leaf"], embed_fn=lambda t: [1.0], summarize_fn=lambda texts: "unused")
    assert root.level == 0
    assert root.text == "only leaf"
    assert root.is_leaf


def test_build_raptor_tree_converges_to_a_single_summarized_root():
    vectors = {"leaf-1": [1.0, 0.0], "leaf-2": [0.95, 0.05]}

    root = build_raptor_tree(
        ["leaf-1", "leaf-2"],
        embed_fn=lambda text: vectors[text],
        summarize_fn=lambda texts: "SUMMARY",
        similarity_threshold=0.5,
        max_levels=3,
    )

    assert root.level == 1
    assert root.text == "SUMMARY"
    assert {child.text for child in root.children} == {"leaf-1", "leaf-2"}


def test_build_raptor_tree_clusters_similar_leaves_and_leaves_unrelated_ones_apart():
    vectors = {
        "leaf-a1": [1.0, 0.0],
        "leaf-a2": [0.9, 0.1],
        "leaf-b1": [0.0, 1.0],
        "leaf-b2": [0.1, 0.9],
    }
    summaries_written: list[str] = []

    def summarize_fn(texts: list[str]) -> str:
        summary = f"summary of {sorted(texts)}"
        summaries_written.append(summary)
        return summary

    root = build_raptor_tree(
        ["leaf-a1", "leaf-a2", "leaf-b1", "leaf-b2"],
        embed_fn=lambda text: vectors[text],
        summarize_fn=summarize_fn,
        similarity_threshold=0.8,
        max_levels=1,  # stop after one clustering round -- see the module docstring
    )

    # Two clusters formed (the "a" pair and the "b" pair), each summarized
    # once, then wrapped under a synthetic root since max_levels stopped
    # the loop before a second round could merge them further.
    assert len(summaries_written) == 2
    assert len(root.children) == 2
    assert {child.level for child in root.children} == {1}

    grouped_leaf_texts = {frozenset(gc.text for gc in child.children) for child in root.children}
    assert grouped_leaf_texts == {
        frozenset({"leaf-a1", "leaf-a2"}),
        frozenset({"leaf-b1", "leaf-b2"}),
    }


def test_raptor_search_matches_the_right_leaf_across_a_tree_with_a_summary_above_it():
    leaf_alpha = RaptorNode(text="alpha content", level=0)
    leaf_beta = RaptorNode(text="beta content", level=0)
    root = RaptorNode(text="general summary", level=1, children=[leaf_alpha, leaf_beta])

    vectors = {
        "alpha content": [1.0, 0.0],
        "beta content": [0.0, 1.0],
        "general summary": [0.5, 0.5],
        "alpha query": [1.0, 0.0],
    }

    results = raptor_search("alpha query", root, embed_fn=lambda text: vectors[text], top_k=3)

    assert results[0][0].text == "alpha content"
    assert results[0][1] > results[1][1] > results[2][1]
