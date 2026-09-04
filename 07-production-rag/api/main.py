"""FastAPI app assembly: build the RAG pipeline once at startup (embed +
index the corpus into Qdrant if not already populated), store every
shared dependency on `app.state`, and expose `/query`, `/health`, and a
Prometheus `/metrics` endpoint.
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

LEVEL_DIR = Path(__file__).resolve().parent.parent
for sub in ["", "retrieval-infrastructure", "caching", "security"]:
    sys.path.insert(0, str(LEVEL_DIR / sub) if sub else str(LEVEL_DIR))

from production_common.dataset import prepare
from production_common.embed import OllamaEmbedder
from production_common.llm import OllamaLLM
from qdrant import QdrantStore
from response_cache import ResponseCache
from semantic_cache import SemanticCache
from document_acl import DocumentACL
from personalization import PersonalizationEngine, UserProfile
from observability.telemetry import setup_tracing

from .routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    data = prepare()
    embedder = OllamaEmbedder()
    llm = OllamaLLM()
    qdrant = QdrantStore()

    if qdrant.count() == 0:
        doc_ids = list(data.corpus.keys())
        vectors = embedder.embed_many([data.corpus[d]["text"] for d in doc_ids])
        payloads = [{"title": data.corpus[d]["title"], "text": data.corpus[d]["text"]} for d in doc_ids]
        qdrant.upsert(doc_ids, vectors, payloads)

    app.state.corpus = data.corpus
    app.state.embedder = embedder
    app.state.llm = llm
    app.state.qdrant = qdrant
    app.state.response_cache = ResponseCache()
    app.state.semantic_cache = SemanticCache(embedder=embedder)
    app.state.acl = DocumentACL(doc_owners={})  # empty -- every doc public by default

    # Demo personalization profiles -- in a real deployment these would come
    # from a user-profile store (built from real query history), not be
    # hardcoded. Two profiles with genuinely different interests, so the
    # same question can be shown ranking differently for each (see
    # tests/test_personalization.py and the README's worked example). A
    # user_id not listed here (including the default API-key-derived one)
    # gets the unpersonalized ranking, unchanged.
    app.state.personalization = PersonalizationEngine(
        profiles={
            "alice": UserProfile(user_id="alice", interests=frozenset({"history", "war", "government"})),
            "bob": UserProfile(user_id="bob", interests=frozenset({"science", "biology", "physics"})),
        }
    )

    yield


app = FastAPI(title="Production RAG API", version="0.1.0", lifespan=lifespan)
app.include_router(router)
setup_tracing(app)  # OpenTelemetry auto-instrumentation, spans -> console (see telemetry.py)


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
