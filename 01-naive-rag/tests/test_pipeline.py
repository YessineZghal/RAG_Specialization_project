"""End-to-end pipeline tests using fakes for the embedder and generator.

These exercise the real `chunk.py` and `retrieve.py` code — only the
network-touching pieces (Ollama) are replaced — so a green test suite is a
genuine signal that load -> chunk -> embed -> store -> retrieve -> generate
is wired correctly, without requiring Ollama or internet access.
"""

from __future__ import annotations

import pytest

from src.generate import ExtractiveGenerator, build_prompt
from src.pipeline import RAGPipeline
from src.retrieve import InMemoryVectorStore
from src.schema import Chunk, RetrievedChunk


@pytest.fixture
def pipeline(fake_embedder) -> RAGPipeline:
    return RAGPipeline(
        embedder=fake_embedder,
        generator=ExtractiveGenerator(),
        vector_store=InMemoryVectorStore(),
        chunk_size=50,
        chunk_overlap=0,
        top_k=2,
    )


def test_build_index_returns_chunk_count(pipeline, sample_documents):
    n_chunks = pipeline.build_index(sample_documents)
    assert n_chunks == len(sample_documents)  # each doc fits in a single 50-word chunk
    assert len(pipeline.vector_store) == n_chunks


def test_ask_before_indexing_raises(pipeline):
    with pytest.raises(RuntimeError, match="build_index"):
        pipeline.ask("anything")


def test_ask_retrieves_the_relevant_document(pipeline, sample_documents):
    pipeline.build_index(sample_documents)

    answer = pipeline.ask("How many days is the refund window and who approves refunds?")

    assert answer.sources, "expected at least one retrieved source"
    assert answer.sources[0].chunk.document_id == "doc-refunds"
    # ExtractiveGenerator embeds the top source's chunk text verbatim.
    assert answer.sources[0].chunk.text in answer.answer


def test_ask_respects_top_k(pipeline, sample_documents):
    pipeline.build_index(sample_documents)
    answer = pipeline.ask("probation period", top_k=1)
    assert len(answer.sources) == 1


def test_save_and_load_index_round_trip(pipeline, sample_documents, tmp_path):
    pipeline.build_index(sample_documents)
    pipeline.save_index(tmp_path / "index")

    reloaded = RAGPipeline.load_index(
        tmp_path / "index",
        embedder=pipeline.embedder,
        generator=ExtractiveGenerator(),
    )
    answer = reloaded.ask("remote work days per week")
    assert answer.sources[0].chunk.document_id == "doc-onboarding"


def test_save_index_rejects_non_memory_store(fake_embedder):
    class DummyStore:
        def __len__(self):
            return 0

    pipeline = RAGPipeline(
        embedder=fake_embedder,
        generator=ExtractiveGenerator(),
        vector_store=DummyStore(),
    )
    with pytest.raises(NotImplementedError):
        pipeline.save_index()


def test_build_prompt_numbers_sources_and_includes_question():
    sources = [
        RetrievedChunk(
            chunk=Chunk(id="c1", text="Refunds take 30 days.", document_id="doc-1", position=0),
            score=0.9,
        )
    ]
    prompt = build_prompt("What is the refund period?", sources)

    assert "[Source 1]" in prompt
    assert "Refunds take 30 days." in prompt
    assert "What is the refund period?" in prompt


def test_extractive_generator_returns_top_source():
    sources = [
        RetrievedChunk(
            chunk=Chunk(id="c1", text="Top answer text.", document_id="doc-1", position=0),
            score=0.9,
        ),
        RetrievedChunk(
            chunk=Chunk(id="c2", text="Second answer text.", document_id="doc-2", position=0),
            score=0.5,
        ),
    ]
    result = ExtractiveGenerator().generate("irrelevant", sources)
    assert "Top answer text." in result


def test_extractive_generator_handles_no_sources():
    result = ExtractiveGenerator().generate("irrelevant", [])
    assert "don't know" in result.lower()
