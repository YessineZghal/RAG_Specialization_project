"""Route handlers — the RAG pipeline itself lives in `main.py`'s app
state (retriever, embedder, LLM, caches); this module wires HTTP
concerns (auth, request/response schemas, metrics) around calls into it.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from observability.telemetry import CACHE_HITS, CACHE_MISSES, GENERATION_LATENCY, REQUEST_COUNT, REQUEST_LATENCY, RETRIEVAL_LATENCY  # noqa: E402
from security.auth import AuthError, verify_admin_key, verify_api_key  # noqa: E402
from security.permissions import PermissionDeniedError, require_permission  # noqa: E402
from security.prompt_injection import is_suspicious  # noqa: E402

from .schemas import HealthResponse, IngestRequest, IngestResponse, QueryRequest, QueryResponse, Source  # noqa: E402

router = APIRouter()

# Retrieve this many times top_k from Qdrant before personalizing and
# truncating -- personalization can only ever promote a document that was
# actually retrieved, so it needs a wider candidate pool than top_k to have
# anything real to work with. See `security/personalization.py`.
PERSONALIZATION_CANDIDATE_MULTIPLIER = 4


def require_api_key(request: Request) -> str:
    provided = request.headers.get("x-api-key")
    try:
        return verify_api_key(provided)
    except AuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


def personalization_user_id(request: Request, user_id: str = Depends(require_api_key)) -> str:
    """The identity personalization and cache-namespacing use, which is
    deliberately *not* always the same as the API-key-derived `user_id`
    ACL decisions use. A real deployment sits behind one shared
    service-level API key while carrying the actual end user's identity
    separately (a JWT claim, a session header, ...); this repo has no
    real auth service in front of it, so an optional `x-user-id` header
    stands in for that -- present it to see a different user's
    personalized ranking through the same API key, or omit it to fall
    back to the API-key-derived id (the pre-personalization behavior).
    """
    return request.headers.get("x-user-id") or user_id


def _retrieve_and_personalize(state, question: str, top_k: int, acl_user_id: str, personalize_as: str) -> list[dict]:
    allowed_doc_ids = state.acl.allowed_doc_ids(acl_user_id, list(state.corpus.keys()))

    retrieval_start = time.time()
    query_vector = state.embedder.embed_one(question)
    candidate_k = top_k * PERSONALIZATION_CANDIDATE_MULTIPLIER
    results = state.qdrant.search(query_vector, top_k=candidate_k, allowed_doc_ids=allowed_doc_ids)
    RETRIEVAL_LATENCY.observe(time.time() - retrieval_start)

    reranked = state.personalization.rerank(personalize_as, results, state.corpus)
    return reranked[:top_k]


def _build_context(state, results: list[dict]) -> str:
    return "\n\n".join(state.corpus[r["doc_id"]]["text"] for r in results if r["doc_id"] in state.corpus)


def _build_sources(state, results: list[dict]) -> list[Source]:
    return [
        Source(doc_id=r["doc_id"], title=state.corpus.get(r["doc_id"], {}).get("title", ""), score=r["base_score"])
        for r in results
    ]


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    qdrant_docs = request.app.state.qdrant.count()
    return HealthResponse(status="ok", qdrant_docs=qdrant_docs)


@router.post("/query", response_model=QueryResponse)
def query(
    payload: QueryRequest,
    request: Request,
    user_id: str = Depends(require_api_key),
    personalize_as: str = Depends(personalization_user_id),
) -> QueryResponse:
    start = time.time()
    state = request.app.state

    if is_suspicious(payload.question, llm=state.llm, use_llm_check=False):
        REQUEST_COUNT.labels(endpoint="/query", status="rejected").inc()
        raise HTTPException(status_code=400, detail="Query rejected: possible prompt injection detected.")

    cached = state.response_cache.get(payload.question, namespace=personalize_as)
    if cached:
        CACHE_HITS.labels(cache_type="response").inc()
        REQUEST_COUNT.labels(endpoint="/query", status="200").inc()
        return QueryResponse(**cached, cache_hit="response", latency_ms=(time.time() - start) * 1000)
    CACHE_MISSES.labels(cache_type="response").inc()

    semantic_hit = state.semantic_cache.get(payload.question, namespace=personalize_as)
    if semantic_hit:
        CACHE_HITS.labels(cache_type="semantic").inc()
        REQUEST_COUNT.labels(endpoint="/query", status="200").inc()
        answer_data = semantic_hit["answer"]
        return QueryResponse(**answer_data, cache_hit="semantic", latency_ms=(time.time() - start) * 1000)
    CACHE_MISSES.labels(cache_type="semantic").inc()

    results = _retrieve_and_personalize(state, payload.question, payload.top_k, user_id, personalize_as)

    context = _build_context(state, results)
    generation_start = time.time()
    answer_text = state.llm.complete(
        f"Context:\n{context}\n\nQuestion: {payload.question}\nAnswer using only the context above:"
    )
    GENERATION_LATENCY.observe(time.time() - generation_start)

    sources = _build_sources(state, results)
    response_data = {"question": payload.question, "answer": answer_text, "sources": [s.model_dump() for s in sources]}
    state.response_cache.set(payload.question, response_data, namespace=personalize_as)
    state.semantic_cache.set(payload.question, response_data, namespace=personalize_as)

    REQUEST_COUNT.labels(endpoint="/query", status="200").inc()
    REQUEST_LATENCY.labels(endpoint="/query").observe(time.time() - start)
    return QueryResponse(**response_data, cache_hit=None, latency_ms=(time.time() - start) * 1000)


@router.post("/query/stream")
def query_stream(
    payload: QueryRequest,
    request: Request,
    user_id: str = Depends(require_api_key),
    personalize_as: str = Depends(personalization_user_id),
) -> StreamingResponse:
    """Same pipeline as `/query`, but the generation step streams tokens
    to the caller as they are produced instead of waiting for the full
    answer. This does not make generation itself faster -- see
    `../load-testing/scenarios.md` for the real, measured bottleneck --
    it changes what a caller experiences while waiting: a cache hit still
    returns instantly (streamed as a single chunk), and a cache miss now
    shows the answer arriving token by token instead of one long silent
    wait followed by everything at once.
    """
    state = request.app.state

    if is_suspicious(payload.question, llm=state.llm, use_llm_check=False):
        REQUEST_COUNT.labels(endpoint="/query/stream", status="rejected").inc()
        raise HTTPException(status_code=400, detail="Query rejected: possible prompt injection detected.")

    cached = state.response_cache.get(payload.question, namespace=personalize_as)
    if cached:
        CACHE_HITS.labels(cache_type="response").inc()
        REQUEST_COUNT.labels(endpoint="/query/stream", status="200").inc()
        return StreamingResponse(iter([cached["answer"]]), media_type="text/plain")
    CACHE_MISSES.labels(cache_type="response").inc()

    semantic_hit = state.semantic_cache.get(payload.question, namespace=personalize_as)
    if semantic_hit:
        CACHE_HITS.labels(cache_type="semantic").inc()
        REQUEST_COUNT.labels(endpoint="/query/stream", status="200").inc()
        return StreamingResponse(iter([semantic_hit["answer"]["answer"]]), media_type="text/plain")
    CACHE_MISSES.labels(cache_type="semantic").inc()

    results = _retrieve_and_personalize(state, payload.question, payload.top_k, user_id, personalize_as)
    context = _build_context(state, results)
    sources = _build_sources(state, results)
    prompt = f"Context:\n{context}\n\nQuestion: {payload.question}\nAnswer using only the context above:"

    def token_generator():
        generation_start = time.time()
        chunks: list[str] = []
        for token in state.llm.stream_complete(prompt):
            chunks.append(token)
            yield token
        GENERATION_LATENCY.observe(time.time() - generation_start)

        answer_text = "".join(chunks).strip()
        response_data = {
            "question": payload.question,
            "answer": answer_text,
            "sources": [s.model_dump() for s in sources],
        }
        state.response_cache.set(payload.question, response_data, namespace=personalize_as)
        state.semantic_cache.set(payload.question, response_data, namespace=personalize_as)
        REQUEST_COUNT.labels(endpoint="/query/stream", status="200").inc()

    return StreamingResponse(token_generator(), media_type="text/plain")


def require_admin(request: Request) -> str:
    """Gate for `/admin/ingest`: a *separate* admin key (never the regular
    `/query` API key -- see `security/auth.py`'s `verify_admin_key`), and
    then a real permission check via `security/permissions.py`. That
    permission check was previously only ever exercised in tests; this is
    the first route in this level that actually calls it for real.
    """
    provided = request.headers.get("x-admin-api-key")
    try:
        role = verify_admin_key(provided)
    except AuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    try:
        require_permission(role, "ingest")
    except PermissionDeniedError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    return role


@router.post("/admin/ingest", response_model=IngestResponse)
def ingest(payload: IngestRequest, request: Request, role: str = Depends(require_admin)) -> IngestResponse:
    """Add or update one document in the live corpus -- embedded and
    upserted into Qdrant immediately, and reflected in `app.state.corpus`
    for the very next request, with no API restart. This is the
    "push-based, live-updating" counterpart to Levels 3-6's web/API RAG,
    which only ever pull a fresh answer per query; this makes the
    corpus itself change while the process keeps running.
    """
    state = request.app.state
    is_update = payload.doc_id in state.corpus

    vector = state.embedder.embed_one(payload.text)
    state.qdrant.upsert([payload.doc_id], [vector], [{"title": payload.title, "text": payload.text}])
    state.corpus[payload.doc_id] = {"title": payload.title, "text": payload.text}

    return IngestResponse(
        doc_id=payload.doc_id,
        status="updated" if is_update else "created",
        corpus_size=len(state.corpus),
    )
