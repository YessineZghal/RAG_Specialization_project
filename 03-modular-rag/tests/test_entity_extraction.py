from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "graph-rag"))
from entity_extraction import _parse_triples, extract_triples  # noqa: E402


def test_parse_triples_clean_json():
    raw = '[{"subject": "A", "relation": "works at", "object": "B"}]'
    assert _parse_triples(raw) == [{"subject": "A", "relation": "works at", "object": "B"}]


def test_parse_triples_wrapped_in_markdown_fence():
    raw = '```json\n[{"subject": "A", "relation": "R", "object": "B"}]\n```'
    assert _parse_triples(raw) == [{"subject": "A", "relation": "R", "object": "B"}]


def test_parse_triples_extracts_json_from_surrounding_prose():
    raw = 'Here are the triples:\n[{"subject": "A", "relation": "R", "object": "B"}]\nHope that helps!'
    assert _parse_triples(raw) == [{"subject": "A", "relation": "R", "object": "B"}]


def test_parse_triples_ignores_malformed_entries():
    raw = '[{"subject": "A", "relation": "R"}, {"subject": "C", "relation": "R2", "object": "D"}]'
    # first entry is missing "object" and should be dropped, not crash
    assert _parse_triples(raw) == [{"subject": "C", "relation": "R2", "object": "D"}]


def test_parse_triples_returns_empty_for_unparseable_input():
    assert _parse_triples("I cannot extract any triples from this text.") == []


def test_parse_triples_returns_empty_for_non_list_json():
    assert _parse_triples('{"subject": "A"}') == []


def test_extract_triples_uses_injected_llm(fake_llm):
    llm = fake_llm(response='[{"subject": "X", "relation": "Y", "object": "Z"}]')
    result = extract_triples("some source text", llm=llm)
    assert result == [{"subject": "X", "relation": "Y", "object": "Z"}]
    assert "some source text" in llm.calls[0]["prompt"]
