# Background worker image -- for jobs that shouldn't run in the request
# path of api.Dockerfile's container: batch re-embedding after a corpus
# update, the regression suite (production_eval/regression_suite.py) on a
# schedule, etc. Same base and dependency layer as api.Dockerfile,
# different entrypoint.
#
#   docker build -f 07-production-rag/docker/worker.Dockerfile -t rag-l7-worker .
#   docker run --rm rag-l7-worker uv run --no-sync python -m production_eval.regression_suite
#
# Reference-only in the same sense as ../kubernetes/*.yaml: the image
# builds (same verified base layer as api.Dockerfile), but no scheduler
# (cron, k8s CronJob) actually invokes it in this repo -- there is no
# cluster here to run one against.
FROM python:3.11-slim

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --extra production --no-dev

COPY 07-production-rag/ 07-production-rag/

WORKDIR /app/07-production-rag

ENV OLLAMA_HOST=http://host.docker.internal:11434 \
    PROD_QDRANT_URL=http://host.docker.internal:16333 \
    PROD_POSTGRES_DSN=postgresql://postgres:postgres@host.docker.internal:15432/production_rag \
    PROD_REDIS_URL=redis://host.docker.internal:16379/0

# No CMD -- invoked with an explicit command per job, e.g.:
#   docker run --rm rag-l7-worker uv run --no-sync python -m production_eval.regression_suite
ENTRYPOINT ["uv", "run", "--no-sync"]
CMD ["python", "-m", "production_eval.regression_suite"]
