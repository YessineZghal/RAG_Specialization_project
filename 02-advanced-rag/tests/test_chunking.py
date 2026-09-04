from __future__ import annotations

import pytest

from chunking.fixed_size import fixed_size_chunk
from chunking.parent_child import chunk_with_parents
from chunking.recursive import recursive_chunk
from chunking.semantic import semantic_chunk, split_sentences


def test_fixed_size_chunk_respects_size_and_overlap():
    text = " ".join(f"w{i}" for i in range(100))
    chunks = fixed_size_chunk(text, chunk_size=20, chunk_overlap=5)
    assert len(chunks) > 1
    assert chunks[0].split()[-5:] == chunks[1].split()[:5]


@pytest.mark.parametrize("chunk_size,chunk_overlap", [(0, 0), (10, 10), (10, 20)])
def test_fixed_size_chunk_rejects_invalid_sizes(chunk_size, chunk_overlap):
    with pytest.raises(ValueError):
        fixed_size_chunk("some text", chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def test_recursive_chunk_prefers_paragraph_boundaries():
    text = "First paragraph is short.\n\nSecond paragraph is also fairly short here."
    chunks = recursive_chunk(text, chunk_size=40, chunk_overlap=0)
    # Should split on the paragraph break rather than mid-sentence.
    assert any(c.startswith("First paragraph") for c in chunks)
    assert any(c.startswith("Second paragraph") for c in chunks)


def test_recursive_chunk_handles_empty_text():
    assert recursive_chunk("   ", chunk_size=100, chunk_overlap=10) == []


def test_recursive_chunk_never_exceeds_size_by_much_with_char_fallback():
    text = "a" * 500  # no separators at all -> falls back to character split
    chunks = recursive_chunk(text, chunk_size=100, chunk_overlap=0, separators=["\n\n", "\n", ""])
    assert all(len(c) <= 100 for c in chunks)
    assert "".join(chunks) == text


def test_split_sentences_basic():
    sentences = split_sentences("First sentence. Second sentence! Third one?")
    assert sentences == ["First sentence.", "Second sentence!", "Third one?"]


def test_semantic_chunk_returns_whole_text_when_too_short(fake_embed_fn):
    result = semantic_chunk("Just one sentence here.", embed_fn=fake_embed_fn)
    assert result == ["Just one sentence here."]


def test_semantic_chunk_produces_nonempty_chunks_covering_all_sentences(fake_embed_fn):
    text = (
        "Cats are small domesticated mammals. Cats often hunt small rodents. "
        "The stock market fell sharply today. Investors reacted to interest rate news."
    )
    chunks = semantic_chunk(text, embed_fn=fake_embed_fn, breakpoint_percentile=50)
    assert len(chunks) >= 1
    # Every original sentence should appear in exactly one output chunk.
    joined = " ".join(chunks)
    for sentence in split_sentences(text):
        assert sentence in joined


def test_chunk_with_parents_preserves_parent_text():
    text = " ".join(f"w{i}" for i in range(40))
    children = chunk_with_parents(text, parent_size=20, child_size=8, child_overlap=2)

    assert len(children) > 0
    parent_ids = {c.parent_id for c in children}
    assert parent_ids == {0, 1}
    for child in children:
        assert child.text.split()[0] in child.parent_text
