"""Pure-logic tests for the caption-based table/image indexing in
multimodal-rag/ — synthetic page text in, no PDF download required.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "multimodal-rag"))
from image_retrieval import extract_captions  # noqa: E402
from table_retrieval import extract_tables  # noqa: E402

SAMPLE_PAGES = [
    "Some intro text with no captions on this page.",
    "Figure 1: The Transformer - model architecture. Follows an encoder-decoder design.",
    "Table 1: Maximum path lengths per layer type. n is the sequence length.",
    "As shown in Table 1. and Figure 1. the model performs well.",  # in-text refs, period not colon
    "Table 2: BLEU scores compared to prior work. Our model scores higher.",
]


def test_extract_tables_finds_real_captions_only():
    tables = extract_tables(SAMPLE_PAGES)
    assert set(tables.keys()) == {"table-1", "table-2"}
    assert tables["table-1"]["page"] == 3
    assert "Maximum path lengths" in tables["table-1"]["text"]


def test_extract_tables_ignores_in_text_period_references():
    tables = extract_tables(SAMPLE_PAGES)
    # page 4 has "Table 1." (period, in-text reference) -- must not create a second entry
    assert len(tables) == 2


def test_extract_tables_keeps_first_occurrence_only():
    pages = ["Table 1: Real caption here.", "Table 1: A different, wrong duplicate."]
    tables = extract_tables(pages)
    assert tables["table-1"]["page"] == 1
    assert "Real caption" in tables["table-1"]["text"]


def test_extract_captions_finds_figures_only_not_tables():
    captions = extract_captions(SAMPLE_PAGES)
    assert set(captions.keys()) == {1}
    assert "Transformer" in captions[1]


def test_extract_tables_empty_pages_returns_empty():
    assert extract_tables([]) == {}
    assert extract_captions([]) == {}
