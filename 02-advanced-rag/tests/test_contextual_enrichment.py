from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chunking.contextual_enrichment import (  # noqa: E402
    enrich_chunk,
    enrich_chunks,
    generate_chunk_context,
)


def test_generate_chunk_context_strips_quotes_and_whitespace(fake_llm):
    llm = fake_llm(response='"This chunk covers Q1 European sales."  ')
    context = generate_chunk_context("An annual report.", "Revenue increased by 18%.", llm=llm)
    assert context == "This chunk covers Q1 European sales."


def test_generate_chunk_context_sends_both_the_document_and_the_chunk(fake_llm):
    llm = fake_llm(response="context")
    generate_chunk_context("FULL DOCUMENT TEXT", "THE CHUNK", llm=llm)
    prompt = llm.calls[0]["prompt"]
    assert "FULL DOCUMENT TEXT" in prompt
    assert "THE CHUNK" in prompt


def test_enrich_chunk_prepends_the_generated_context_to_the_chunk(fake_llm):
    llm = fake_llm(response="This is the Q1 European sales section.")
    enriched = enrich_chunk("An annual report.", "Revenue increased by 18%.", llm=llm)
    assert enriched == "This is the Q1 European sales section.\n\nRevenue increased by 18%."


def test_enrich_chunks_calls_the_llm_once_per_chunk(fake_llm):
    llm = fake_llm(responses=["context one", "context two", "context three"])
    chunks = ["chunk one text", "chunk two text", "chunk three text"]

    enriched = enrich_chunks("A document.", chunks, llm=llm)

    assert len(llm.calls) == 3
    assert enriched[0] == "context one\n\nchunk one text"
    assert enriched[1] == "context two\n\nchunk two text"
    assert enriched[2] == "context three\n\nchunk three text"
