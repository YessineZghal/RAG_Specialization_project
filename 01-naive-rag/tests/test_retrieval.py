from __future__ import annotations

import numpy as np
import pytest

from src.retrieve import InMemoryVectorStore, _cosine_similarity
from src.schema import Chunk, EmbeddedChunk


def make_embedded_chunk(chunk_id: str, vector: list[float]) -> EmbeddedChunk:
    chunk = Chunk(id=chunk_id, text=f"text for {chunk_id}", document_id=chunk_id, position=0)
    return EmbeddedChunk(chunk=chunk, vector=vector)


def test_cosine_similarity_identical_vectors_is_one():
    matrix = np.array([[1.0, 0.0, 0.0]])
    query = np.array([1.0, 0.0, 0.0])
    assert _cosine_similarity(matrix, query)[0] == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors_is_zero():
    matrix = np.array([[0.0, 1.0]])
    query = np.array([1.0, 0.0])
    assert _cosine_similarity(matrix, query)[0] == pytest.approx(0.0)


def test_in_memory_store_starts_empty():
    store = InMemoryVectorStore()
    assert len(store) == 0
    assert store.search([1.0, 0.0], top_k=3) == []


def test_in_memory_store_returns_most_similar_first():
    store = InMemoryVectorStore()
    store.add(
        [
            make_embedded_chunk("far", [0.0, 1.0]),
            make_embedded_chunk("close", [1.0, 0.01]),
            make_embedded_chunk("exact", [1.0, 0.0]),
        ]
    )

    results = store.search([1.0, 0.0], top_k=2)

    assert len(results) == 2
    assert results[0].chunk.id == "exact"
    assert results[1].chunk.id == "close"
    assert results[0].score >= results[1].score


def test_in_memory_store_top_k_clamped_to_available_chunks():
    store = InMemoryVectorStore()
    store.add([make_embedded_chunk("only-one", [1.0, 0.0])])

    results = store.search([1.0, 0.0], top_k=10)

    assert len(results) == 1


def test_in_memory_store_save_and_load_round_trip(tmp_path):
    store = InMemoryVectorStore()
    store.add(
        [
            make_embedded_chunk("a", [1.0, 0.0, 0.0]),
            make_embedded_chunk("b", [0.0, 1.0, 0.0]),
        ]
    )
    store.save(tmp_path / "index")

    loaded = InMemoryVectorStore.load(tmp_path / "index")

    assert len(loaded) == len(store)
    results = loaded.search([1.0, 0.0, 0.0], top_k=1)
    assert results[0].chunk.id == "a"


def test_in_memory_store_load_missing_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        InMemoryVectorStore.load(tmp_path / "does-not-exist")
