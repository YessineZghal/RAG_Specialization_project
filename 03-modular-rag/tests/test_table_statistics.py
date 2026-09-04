"""Pure-logic tests for multimodal-rag/table_statistics.py -- synthetic
text modeled directly on the real PDF's own table-2 text (verified by
hand against the actual downloaded text), no PDF download required.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "multimodal-rag"))
from table_statistics import compute_statistics, enrich_tables_with_statistics, extract_numbers

# Modeled on the real table-2 text: real decimal BLEU scores, citation
# brackets that look like bare integers, and mangled scientific notation
# ("1019" for what was really "10^19") that also looks like a bare integer.
REAL_SHAPED_TABLE_TEXT = (
    "ByteNet [18] 23.75 Deep-Att + PosUnk [39] 39.2 1.0· 1020 "
    "GNMT + RL [38] 24.6 39.92 2.3· 1019 1.4· 1020"
)


def test_extract_numbers_finds_every_real_decimal_score():
    numbers = extract_numbers(REAL_SHAPED_TABLE_TEXT)
    assert numbers == [23.75, 39.2, 1.0, 24.6, 39.92, 2.3, 1.4]


def test_extract_numbers_excludes_citation_brackets():
    numbers = extract_numbers("ByteNet [18] achieved a score, see reference [39] for details.")
    assert numbers == []  # [18] and [39] are bare integers, not decimals -- correctly excluded


def test_extract_numbers_excludes_mangled_scientific_notation_exponents():
    numbers = extract_numbers("cost was 1020 FLOPs and separately 1019 FLOPs")
    assert numbers == []  # bare integers (the exponent, mangled), no decimal point


def test_extract_numbers_on_text_with_no_numbers_returns_empty_list():
    assert extract_numbers("Just a plain sentence with no numbers in it.") == []


def test_compute_statistics_on_real_bleu_scores():
    stats = compute_statistics([23.75, 39.2, 24.6, 39.92, 25.16, 40.46])
    assert stats["count"] == 6
    assert stats["min"] == 23.75
    assert stats["max"] == 40.46
    assert stats["mean"] == sum([23.75, 39.2, 24.6, 39.92, 25.16, 40.46]) / 6


def test_compute_statistics_on_empty_list_returns_none_not_zeros():
    assert compute_statistics([]) is None


def test_enrich_tables_with_statistics_adds_a_statistics_key_to_every_table():
    tables = {
        "table-1": {"page": 1, "text": "no numbers here at all"},
        "table-2": {"page": 2, "text": REAL_SHAPED_TABLE_TEXT},
    }
    enriched = enrich_tables_with_statistics(tables)
    assert enriched["table-1"]["statistics"] is None
    assert enriched["table-2"]["statistics"]["count"] == 7
    assert enriched["table-2"]["statistics"]["max"] == 39.92


def test_enrich_tables_with_statistics_mutates_and_returns_the_same_dict():
    tables = {"table-1": {"page": 1, "text": "5.5 and 6.6"}}
    result = enrich_tables_with_statistics(tables)
    assert result is tables
    assert tables["table-1"]["statistics"]["count"] == 2
