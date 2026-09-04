"""Recursive character chunking.

Tries to split on the *most structural* separator first (paragraph breaks),
falling back to progressively finer-grained separators only where a piece
is still too big — so chunk boundaries land on paragraph/sentence edges
whenever possible, instead of at an arbitrary word count like
`fixed_size.py`. This is a from-scratch reimplementation of the same idea
behind LangChain's `RecursiveCharacterTextSplitter` (no dependency on
LangChain itself — sizes are in **characters**, not words).
"""

from __future__ import annotations

DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def _merge_splits(splits: list[str], separator: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    sep_len = len(separator)

    for piece in splits:
        piece_len = len(piece)
        added_len = piece_len + (sep_len if current else 0)
        if current and current_len + added_len > chunk_size:
            chunks.append(separator.join(current))
            while current and current_len > chunk_overlap:
                dropped = current.pop(0)
                current_len -= len(dropped) + sep_len
        current.append(piece)
        current_len += piece_len + (sep_len if len(current) > 1 else 0)

    if current:
        chunks.append(separator.join(current))
    return chunks


def recursive_chunk(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
    separators: list[str] | None = None,
) -> list[str]:
    separators = separators if separators is not None else DEFAULT_SEPARATORS
    if not text.strip():
        return []

    separator = separators[0]
    remaining = separators[1:]
    splits = list(text) if separator == "" else text.split(separator)

    good_splits: list[str] = []
    final_chunks: list[str] = []

    for piece in splits:
        if len(piece) < chunk_size:
            good_splits.append(piece)
            continue
        if good_splits:
            final_chunks.extend(_merge_splits(good_splits, separator, chunk_size, chunk_overlap))
            good_splits = []
        if remaining:
            final_chunks.extend(recursive_chunk(piece, chunk_size, chunk_overlap, remaining))
        else:
            final_chunks.append(piece)  # no separators left — keep as an oversized chunk

    if good_splits:
        final_chunks.extend(_merge_splits(good_splits, separator, chunk_size, chunk_overlap))

    return [c.strip() for c in final_chunks if c.strip()]
