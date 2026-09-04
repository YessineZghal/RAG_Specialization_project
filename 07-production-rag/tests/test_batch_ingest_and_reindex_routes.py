"""`/admin/ingest/batch` and `/admin/reindex`'s actual route logic,
called as plain Python functions with a fake `request.app.state` --
bypassing FastAPI's dependency injection and any live server entirely
(the same offline-logic-only philosophy every other test file in this
level uses), while still exercising the real handler bodies, not just
the schemas around them.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from api.routes import ingest_batch, reindex  # noqa: E402
from api.schemas import BatchIngestRequest, IngestRequest, ReindexRequest  # noqa: E402


class FakeEmbedder:
    def __init__(self, model="fake-model"):
        self.model = model
        self.embed_many_calls = []

    def embed_one(self, text):
        return [float(len(text)), 0.0]

    def embed_many(self, texts):
        self.embed_many_calls.append(list(texts))
        return [self.embed_one(t) for t in texts]


class FakeQdrant:
    def __init__(self, collection="fake-collection"):
        self.collection = collection
        self.upsert_calls = []
        self._count = 0

    def upsert(self, doc_ids, vectors, payloads):
        self.upsert_calls.append((list(doc_ids), list(vectors), list(payloads)))
        self._count = len(doc_ids)

    def count(self):
        return self._count


def _fake_request(corpus, embedder=None, qdrant=None):
    state = SimpleNamespace(corpus=corpus, embedder=embedder or FakeEmbedder(), qdrant=qdrant or FakeQdrant())
    return SimpleNamespace(app=SimpleNamespace(state=state))


def test_ingest_batch_creates_new_documents_and_counts_them_correctly():
    request = _fake_request(corpus={})
    payload = BatchIngestRequest(documents=[
        IngestRequest(doc_id="d1", title="One", text="first document"),
        IngestRequest(doc_id="d2", title="Two", text="second document"),
    ])

    response = ingest_batch(payload, request, role="admin")

    assert response.n_created == 2
    assert response.n_updated == 0
    assert response.corpus_size == 2
    assert {r.doc_id for r in response.results} == {"d1", "d2"}
    assert all(r.status == "created" for r in response.results)


def test_ingest_batch_distinguishes_created_from_updated_within_one_call():
    request = _fake_request(corpus={"d1": {"title": "Old", "text": "old text"}})
    payload = BatchIngestRequest(documents=[
        IngestRequest(doc_id="d1", title="New", text="updated text"),  # already exists
        IngestRequest(doc_id="d2", title="Two", text="brand new"),  # does not
    ])

    response = ingest_batch(payload, request, role="admin")

    assert response.n_created == 1
    assert response.n_updated == 1
    assert response.corpus_size == 2
    by_id = {r.doc_id: r.status for r in response.results}
    assert by_id == {"d1": "updated", "d2": "created"}


def test_ingest_batch_embeds_and_upserts_in_exactly_one_batched_call_each():
    request = _fake_request(corpus={})
    embedder = FakeEmbedder()
    qdrant = FakeQdrant()
    request.app.state.embedder = embedder
    request.app.state.qdrant = qdrant
    payload = BatchIngestRequest(documents=[
        IngestRequest(doc_id=f"d{i}", title=f"T{i}", text=f"text {i}") for i in range(5)
    ])

    ingest_batch(payload, request, role="admin")

    assert len(embedder.embed_many_calls) == 1  # not 5 individual embed_one calls
    assert len(qdrant.upsert_calls) == 1  # not 5 individual upsert calls
    assert len(embedder.embed_many_calls[0]) == 5


def test_reindex_writes_the_whole_corpus_into_the_new_collection_without_activating():
    corpus = {"d1": {"title": "One", "text": "abc"}, "d2": {"title": "Two", "text": "defgh"}}
    live_qdrant = FakeQdrant(collection="live-collection")
    live_embedder = FakeEmbedder(model="original-model")
    request = _fake_request(corpus=corpus, embedder=live_embedder, qdrant=live_qdrant)

    payload = ReindexRequest(new_collection_name="v2-collection", activate=False)
    response = reindex(payload, request, role="admin")

    assert response.new_collection_name == "v2-collection"
    assert response.documents_written == 2
    assert response.activated is False
    # the live collection must never have been touched by a reindex call
    assert live_qdrant.upsert_calls == []
    assert request.app.state.qdrant is live_qdrant
    assert request.app.state.embedder is live_embedder


def test_reindex_with_activate_true_swaps_the_live_app_state():
    corpus = {"d1": {"title": "One", "text": "abc"}}
    live_qdrant = FakeQdrant(collection="live-collection")
    live_embedder = FakeEmbedder(model="original-model")
    request = _fake_request(corpus=corpus, embedder=live_embedder, qdrant=live_qdrant)

    payload = ReindexRequest(new_collection_name="v2-collection", embed_model="new-model", activate=True)
    response = reindex(payload, request, role="admin")

    assert response.activated is True
    assert response.embed_model == "new-model"
    # the app's live state now points at the new store/embedder, not the old ones
    assert request.app.state.qdrant is not live_qdrant
    assert request.app.state.qdrant.collection == "v2-collection"
    assert request.app.state.embedder.model == "new-model"


def test_reindex_defaults_to_the_currently_configured_embedding_model():
    corpus = {"d1": {"title": "One", "text": "abc"}}
    live_embedder = FakeEmbedder(model="already-configured-model")
    request = _fake_request(corpus=corpus, embedder=live_embedder)

    payload = ReindexRequest(new_collection_name="v2-collection")  # embed_model omitted
    response = reindex(payload, request, role="admin")

    assert response.embed_model == "already-configured-model"
