"""Level 1 — Naive RAG.

A from-scratch, end-to-end Retrieval-Augmented Generation pipeline:

    Document -> Load -> Chunk -> Embed -> Vector Store -> Retrieve Top-K
             -> Prompt -> LLM -> Answer

Every stage is a small, single-responsibility module so it can be read,
tested, and swapped independently. See ../README.md for the full write-up.
"""
