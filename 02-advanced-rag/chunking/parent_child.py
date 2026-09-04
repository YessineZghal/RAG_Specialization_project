"""Parent-child ("small-to-big") chunking.

Retrieval matches best on small, focused chunks — but a small chunk often
lacks the surrounding context an LLM needs to answer well. Parent-child
chunking resolves this by indexing small **child** chunks for retrieval
while keeping a pointer to the larger **parent** chunk they came from; at
generation time you hand the LLM the parent, not just the matched child.

    Document
      -> Parent chunks (large, e.g. 2000 chars)
           -> Child chunks (small, e.g. 400 chars) <- these get embedded/indexed
"""

from __future__ import annotations

from dataclasses import dataclass

from .fixed_size import fixed_size_chunk


@dataclass
class ChildChunk:
    text: str
    parent_id: int
    parent_text: str


def chunk_with_parents(
    text: str,
    parent_size: int = 400,
    child_size: int = 80,
    child_overlap: int = 10,
) -> list[ChildChunk]:
    """Split `text` into non-overlapping parent windows (words), then split
    each parent into overlapping child windows (words) for indexing.
    """
    parents = fixed_size_chunk(text, chunk_size=parent_size, chunk_overlap=0)

    children: list[ChildChunk] = []
    for parent_id, parent_text in enumerate(parents):
        for child_text in fixed_size_chunk(parent_text, chunk_size=child_size, chunk_overlap=child_overlap):
            children.append(ChildChunk(text=child_text, parent_id=parent_id, parent_text=parent_text))
    return children
