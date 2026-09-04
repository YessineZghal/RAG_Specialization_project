"""Reindexing into a brand-new collection -- the operational gap this
level's own Success Criteria already disclosed: *"How do you deploy a
new embedding model without corrupting the index? Not exercised in this
level."* `qdrant.py`'s `ensure_collection()` only ever creates a
collection if one doesn't already exist; there was no path to change the
embedding model (a different vector dimension or semantics entirely)
without either destroying the live collection first or leaving it in a
mixed, half-old-half-new state.

The fix is the same one RAG-Anything's "force multimodal reprocessing"
flag points at conceptually (see `../../missing_to_complite.md`): build
the *new* collection completely separately, under its own name, before
touching anything live. If reindexing fails partway, the collection
currently serving traffic was never written to at all -- the caller
swaps to the new collection only after `reindex_corpus` returns
successfully, an explicit, separate step, never automatic.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def reindex_corpus(corpus: dict[str, dict], embedder, new_store) -> int:
    """Embed every document in `corpus` with `embedder` and upsert it
    into `new_store`. `new_store` is any object providing `upsert(doc_ids,
    vectors, payloads)` -- normally a `QdrantStore` pointed at a
    collection name that does not yet exist, but duck-typed so this can
    be tested with a fake, no real Qdrant required.

    Returns the number of documents written. An empty corpus is a no-op,
    not an error -- reindexing "nothing" into a fresh collection is a
    legitimate (if unusual) starting state.
    """
    if not corpus:
        return 0

    doc_ids = list(corpus.keys())
    vectors = embedder.embed_many([corpus[d]["text"] for d in doc_ids])
    payloads = [{"title": corpus[d].get("title", ""), "text": corpus[d]["text"]} for d in doc_ids]
    new_store.upsert(doc_ids, vectors, payloads)

    logger.info("Reindexed %d documents into collection %r", len(doc_ids), getattr(new_store, "collection", "?"))
    return len(doc_ids)


def build_reindexed_store(
    corpus: dict[str, dict],
    embedder,
    new_collection_name: str,
    url: str | None = None,
):
    """Real-Qdrant convenience wrapper: construct a fresh `QdrantStore`
    under `new_collection_name` and reindex `corpus` into it. The
    currently-serving collection is never referenced here at all --
    swapping `app.state.qdrant` to the returned store is the caller's own
    explicit, separate step (see `api/routes.py`'s `/admin/reindex`)."""
    from qdrant import QdrantStore  # local import: only needed for the real-Qdrant path

    new_store = QdrantStore(url=url, collection=new_collection_name)
    reindex_corpus(corpus, embedder, new_store)
    return new_store
