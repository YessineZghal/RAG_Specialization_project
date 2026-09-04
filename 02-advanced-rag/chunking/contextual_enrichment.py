"""Contextual enrichment (Anthropic's "Contextual Retrieval" technique).

A chunk taken out of its document often reads as ambiguous on its own.
Given a document about a company's annual report, the chunk "Revenue
increased by 18%" says nothing about *which* revenue, *which* year, or
*which* part of the business — the surrounding document knows, but the
chunk alone does not.

`context-compression/compressor.py` (already in this level) fixes a
related but different problem at the *opposite* end of the pipeline: it
trims a chunk down to its most query-relevant sentences right before
generation. This module runs *before* embedding: it asks the LLM to write
a short note situating a chunk within its source document, and prepends
that note to the chunk so the enriched version — not the bare chunk — is
what actually gets embedded and indexed. The note travels with the chunk
from indexing time onward; nothing downstream needs to change.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.llm import OllamaLLM

CONTEXTUAL_ENRICHMENT_PROMPT = (
    "Document:\n{document}\n\n"
    "Chunk taken from that document:\n{chunk}\n\n"
    "Write a short context (one or two sentences) that situates this chunk "
    "within the overall document -- what the document is about, and what "
    "part of it this chunk covers. This context will be prepended to the "
    "chunk before it is indexed for search, so a search engine can match it "
    "correctly even without seeing the rest of the document. Respond with "
    "ONLY the context, nothing else."
)


def generate_chunk_context(document: str, chunk: str, llm: OllamaLLM | None = None) -> str:
    llm = llm or OllamaLLM()
    prompt = CONTEXTUAL_ENRICHMENT_PROMPT.format(document=document, chunk=chunk)
    return llm.complete(prompt).strip().strip('"')


def enrich_chunk(document: str, chunk: str, llm: OllamaLLM | None = None) -> str:
    """Return the chunk with its generated context prepended -- this is
    the text that should actually be embedded and indexed, not the bare
    `chunk` on its own.
    """
    context = generate_chunk_context(document, chunk, llm=llm)
    return f"{context}\n\n{chunk}"


def enrich_chunks(document: str, chunks: list[str], llm: OllamaLLM | None = None) -> list[str]:
    """Enrich every chunk from the same source `document` — one LLM call
    per chunk, since each chunk needs its own context note.
    """
    return [enrich_chunk(document, chunk, llm=llm) for chunk in chunks]
