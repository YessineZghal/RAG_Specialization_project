"""`reindex_corpus` -- pure logic, no live Qdrant required. The real
question this covers: does reindexing embed and write every document
into the *new* store, and leave whatever the caller passes as the
currently-serving store completely untouched?
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "retrieval-infrastructure"))
from reindex import reindex_corpus  # noqa: E402


class FakeEmbedder:
    def __init__(self):
        self.calls = []

    def embed_many(self, texts):
        self.calls.append(list(texts))
        return [[float(len(t)), 0.0] for t in texts]  # deterministic, keyed off text length


class FakeStore:
    def __init__(self):
        self.collection = "fake-new-collection"
        self.upserted = None

    def upsert(self, doc_ids, vectors, payloads):
        self.upserted = (list(doc_ids), list(vectors), list(payloads))


def test_reindex_corpus_writes_every_document_into_the_new_store():
    corpus = {
        "d1": {"title": "Doc One", "text": "hello world"},
        "d2": {"title": "Doc Two", "text": "a longer piece of text"},
    }
    embedder = FakeEmbedder()
    new_store = FakeStore()

    n_written = reindex_corpus(corpus, embedder, new_store)

    assert n_written == 2
    doc_ids, vectors, payloads = new_store.upserted
    assert set(doc_ids) == {"d1", "d2"}
    assert len(vectors) == 2
    assert {p["title"] for p in payloads} == {"Doc One", "Doc Two"}
    assert all(p["text"] == corpus[d]["text"] for d, p in zip(doc_ids, payloads, strict=True))


def test_reindex_corpus_embeds_every_document_exactly_once():
    corpus = {"d1": {"title": "T", "text": "abc"}, "d2": {"title": "T2", "text": "defgh"}}
    embedder = FakeEmbedder()
    new_store = FakeStore()

    reindex_corpus(corpus, embedder, new_store)

    assert len(embedder.calls) == 1  # one batched embed_many call, not N individual ones
    assert set(embedder.calls[0]) == {"abc", "defgh"}


def test_reindex_corpus_on_an_empty_corpus_is_a_no_op():
    embedder = FakeEmbedder()
    new_store = FakeStore()

    n_written = reindex_corpus({}, embedder, new_store)

    assert n_written == 0
    assert new_store.upserted is None  # never called upsert() at all
    assert embedder.calls == []


def test_reindex_corpus_defaults_a_missing_title_to_empty_string():
    corpus = {"d1": {"text": "no title here"}}  # no "title" key at all
    embedder = FakeEmbedder()
    new_store = FakeStore()

    reindex_corpus(corpus, embedder, new_store)

    _doc_ids, _vectors, payloads = new_store.upserted
    assert payloads[0]["title"] == ""
