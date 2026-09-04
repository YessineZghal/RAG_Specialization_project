from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chunking.hierarchical import HierarchicalNode, build_hierarchy, hierarchical_search  # noqa: E402


def test_build_hierarchy_creates_three_levels():
    text = "alpha alpha alpha beta beta gamma gamma delta delta epsilon"
    root = build_hierarchy(text, paragraph_size=5, chunk_size=2)

    assert root.level == "document"
    assert root.text == text
    assert len(root.children) == 2  # two 5-word paragraphs
    assert all(p.level == "paragraph" for p in root.children)

    first_paragraph = root.children[0]
    assert first_paragraph.text == "alpha alpha alpha beta beta"
    # 5 words, chunk_size=2, no overlap -> two full 2-word windows plus a
    # trailing 1-word remainder ("beta"), same windowing `fixed_size.py` uses.
    assert [c.text for c in first_paragraph.children] == ["alpha alpha", "alpha beta", "beta"]
    assert all(c.level == "chunk" for c in first_paragraph.children)


def test_build_hierarchy_on_short_text_is_a_single_paragraph_and_chunk():
    root = build_hierarchy("one two", paragraph_size=5, chunk_size=2)
    assert len(root.children) == 1
    assert len(root.children[0].children) == 1
    assert root.children[0].children[0].text == "one two"


def test_hierarchical_search_picks_the_document_sharing_the_querys_vocabulary(fake_embed_fn):
    doc_a = "alpha alpha alpha beta beta gamma gamma delta delta epsilon"
    doc_b = "zeta zeta zeta eta eta theta theta iota iota kappa"

    documents = {
        "doc-a": build_hierarchy(doc_a, paragraph_size=5, chunk_size=2),
        "doc-b": build_hierarchy(doc_b, paragraph_size=5, chunk_size=2),
    }

    result = hierarchical_search("alpha alpha alpha", documents, embed_fn=fake_embed_fn)

    assert result["doc_id"] == "doc-a"


def test_hierarchical_search_drills_down_to_the_right_paragraph_and_chunk(fake_embed_fn):
    # Paragraph 1 is all "alpha"; paragraph 2 is all "gamma"/"delta"/"epsilon".
    doc_a = "alpha alpha alpha beta beta gamma gamma delta delta epsilon"
    documents = {"doc-a": build_hierarchy(doc_a, paragraph_size=5, chunk_size=2)}

    result = hierarchical_search("alpha alpha alpha", documents, embed_fn=fake_embed_fn)

    assert result["doc_id"] == "doc-a"
    assert result["paragraph"] == "alpha alpha alpha beta beta"
    assert result["chunk"] == "alpha alpha"
    assert result["chunk_score"] > 0


def test_hierarchical_search_handles_a_document_with_only_one_chunk_total(fake_embed_fn):
    documents = {"doc-a": build_hierarchy("one two", paragraph_size=5, chunk_size=2)}
    result = hierarchical_search("one two", documents, embed_fn=fake_embed_fn)
    assert result["doc_id"] == "doc-a"
    assert result["chunk"] == "one two"
