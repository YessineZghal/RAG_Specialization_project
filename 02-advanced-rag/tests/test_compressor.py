from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "context-compression"))
from compressor import compress_context


def test_compress_context_keeps_query_relevant_sentences(fake_embed_fn):
    texts = [
        "Refunds are processed within thirty days. The office is painted blue. "
        "Refund requests need an order number."
    ]

    compressed = compress_context("How long do refunds take?", texts, fake_embed_fn, keep_ratio=0.5)

    assert "refund" in compressed.lower()
    assert len(compressed) < len(texts[0])


def test_compress_context_handles_empty_input(fake_embed_fn):
    assert compress_context("query", [], fake_embed_fn) == ""


def test_compress_context_preserves_original_sentence_order(fake_embed_fn):
    texts = ["First sentence about refunds. Second sentence about refunds. Third about refunds."]
    compressed = compress_context("refunds", texts, fake_embed_fn, keep_ratio=1.0)
    assert compressed.index("First") < compressed.index("Second") < compressed.index("Third")
