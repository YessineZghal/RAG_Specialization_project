# Production RAG API image. Builds from the *repo root* build context
# (needs the root pyproject.toml + uv.lock, and only this level's code):
#
#   docker build -f 07-production-rag/docker/api.Dockerfile -t rag-l7-api .
#
# Verified to build successfully in this repo (see README.md#docker). Not
# pushed anywhere or run as a container here -- this level's own FastAPI
# process runs directly on the host against `ollama serve` and the
# docker-compose stack in ../deployment/, which is simpler for local
# development and is what every notebook and the load test actually hit.
# This Dockerfile is what a real deployment (e.g. the Kubernetes manifests
# in ../kubernetes/, reference-only) would build from.
FROM python:3.11-slim

RUN pip install --no-cache-dir uv

WORKDIR /app

# Root project files first (better layer caching -- only rebuilds the
# dependency layer when pyproject.toml/uv.lock change, not on every
# source edit). This is a monorepo with one root pyproject.toml and
# per-level optional-dependency extras -- no per-level pyproject.toml.
COPY pyproject.toml uv.lock ./

RUN uv sync --extra production --no-dev

COPY 07-production-rag/ 07-production-rag/

WORKDIR /app/07-production-rag

ENV OLLAMA_HOST=http://host.docker.internal:11434 \
    PROD_QDRANT_URL=http://host.docker.internal:16333 \
    PROD_POSTGRES_DSN=postgresql://postgres:postgres@host.docker.internal:15432/production_rag \
    PROD_REDIS_URL=redis://host.docker.internal:16379/0

EXPOSE 8001

CMD ["uv", "run", "--no-sync", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8001"]
