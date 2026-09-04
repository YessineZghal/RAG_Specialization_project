"""Hierarchical chunking and progressive retrieval.

Every other chunking strategy in this folder produces one flat list of
chunks per document. Hierarchical chunking keeps three levels instead:

    document (the whole text)
      -> paragraph-level chunks (a few sentences each)
           -> sentence-level chunks (fine-grained, for precise matching)

Retrieval then drills down instead of searching everything at once: find
the best-matching *document* first, then the best-matching *paragraph*
inside it, then the best-matching *sentence-level chunk* inside that
paragraph. Each step only has to compare a handful of candidates, and the
final answer comes with its full path (which document, which paragraph,
which sentence) instead of a single anonymous chunk of text.

Requires an embedding function — pass `common.embed.OllamaEmbedder().embed_one`,
or any `Callable[[str], list[float]]` (a fake one is used in tests), the
same convention `semantic.py` already uses in this folder.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .fixed_size import fixed_size_chunk


@dataclass
class HierarchicalNode:
    text: str
    level: str  # "document", "paragraph", or "chunk"
    children: list["HierarchicalNode"] = field(default_factory=list)


def build_hierarchy(
    text: str,
    paragraph_size: int = 120,
    chunk_size: int = 30,
) -> HierarchicalNode:
    """Split `text` into paragraph-sized windows, then split each paragraph
    into smaller chunk-sized windows. Both splits reuse `fixed_size_chunk`
    (no overlap at either level — the point here is the level structure,
    not overlap handling, which `fixed_size.py` already covers).
    """
    root = HierarchicalNode(text=text, level="document")
    paragraphs = fixed_size_chunk(text, chunk_size=paragraph_size, chunk_overlap=0)
    for paragraph_text in paragraphs:
        paragraph_node = HierarchicalNode(text=paragraph_text, level="paragraph")
        chunks = fixed_size_chunk(paragraph_text, chunk_size=chunk_size, chunk_overlap=0)
        paragraph_node.children = [HierarchicalNode(text=c, level="chunk") for c in chunks]
        root.children.append(paragraph_node)
    return root


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _best_match(
    query_vector: np.ndarray,
    nodes: list[HierarchicalNode],
    embed_fn: Callable[[str], list[float]],
) -> tuple[HierarchicalNode, float]:
    scored = [(node, _cosine(query_vector, np.array(embed_fn(node.text)))) for node in nodes]
    return max(scored, key=lambda pair: pair[1])


def hierarchical_search(
    query: str,
    documents: dict[str, HierarchicalNode],
    embed_fn: Callable[[str], list[float]],
) -> dict:
    """Drill down: best document, then best paragraph inside it, then best
    chunk inside that paragraph. Returns every level's winner and score, so
    a caller can see the whole path, not just the final chunk.
    """
    query_vector = np.array(embed_fn(query))

    doc_ids = list(documents.keys())
    doc_nodes = list(documents.values())
    best_doc_id, best_doc_score = None, float("-inf")
    for doc_id, node in zip(doc_ids, doc_nodes, strict=True):
        score = _cosine(query_vector, np.array(embed_fn(node.text)))
        if score > best_doc_score:
            best_doc_id, best_doc_score = doc_id, score
    best_document = documents[best_doc_id]

    if not best_document.children:
        return {
            "doc_id": best_doc_id,
            "doc_score": best_doc_score,
            "paragraph": None,
            "chunk": None,
        }

    best_paragraph, paragraph_score = _best_match(query_vector, best_document.children, embed_fn)

    if not best_paragraph.children:
        return {
            "doc_id": best_doc_id,
            "doc_score": best_doc_score,
            "paragraph": best_paragraph.text,
            "paragraph_score": paragraph_score,
            "chunk": None,
        }

    best_chunk, chunk_score = _best_match(query_vector, best_paragraph.children, embed_fn)

    return {
        "doc_id": best_doc_id,
        "doc_score": best_doc_score,
        "paragraph": best_paragraph.text,
        "paragraph_score": paragraph_score,
        "chunk": best_chunk.text,
        "chunk_score": chunk_score,
    }
