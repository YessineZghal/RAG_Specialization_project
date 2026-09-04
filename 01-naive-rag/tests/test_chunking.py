from __future__ import annotations

import pytest

from src.chunk import chunk_document, chunk_documents, chunk_text
from src.schema import Document


def test_chunk_text_respects_chunk_size():
    text = " ".join(f"word{i}" for i in range(100))
    chunks = chunk_text(text, chunk_size=20, chunk_overlap=0)

    assert len(chunks) == 5
    assert all(len(c.split()) == 20 for c in chunks)


def test_chunk_text_overlap_repeats_words_at_boundary():
    text = " ".join(f"word{i}" for i in range(30))
    chunks = chunk_text(text, chunk_size=10, chunk_overlap=3)

    # Last 3 words of chunk[0] should equal first 3 words of chunk[1].
    assert chunks[0].split()[-3:] == chunks[1].split()[:3]


def test_chunk_text_empty_string_returns_no_chunks():
    assert chunk_text("   ", chunk_size=10, chunk_overlap=0) == []


def test_chunk_text_shorter_than_chunk_size_returns_one_chunk():
    chunks = chunk_text("just a few words", chunk_size=50, chunk_overlap=5)
    assert chunks == ["just a few words"]


@pytest.mark.parametrize("chunk_size,chunk_overlap", [(0, 0), (-5, 0), (10, 10), (10, 20)])
def test_chunk_text_rejects_invalid_sizes(chunk_size, chunk_overlap):
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def test_chunk_document_preserves_traceability():
    doc = Document(id="doc-1", text=" ".join(f"w{i}" for i in range(25)), metadata={"k": "v"})
    chunks = chunk_document(doc, chunk_size=10, chunk_overlap=0)

    assert [c.document_id for c in chunks] == ["doc-1"] * len(chunks)
    assert [c.position for c in chunks] == list(range(len(chunks)))
    assert all(c.id == f"doc-1::chunk-{c.position}" for c in chunks)
    assert all(c.metadata == {"k": "v"} for c in chunks)


def test_chunk_documents_flattens_multiple_docs():
    docs = [
        Document(id="a", text=" ".join(f"w{i}" for i in range(15))),
        Document(id="b", text=" ".join(f"w{i}" for i in range(15))),
    ]
    chunks = chunk_documents(docs, chunk_size=10, chunk_overlap=0)

    assert {c.document_id for c in chunks} == {"a", "b"}
    assert len(chunks) == 4  # 2 chunks per doc
