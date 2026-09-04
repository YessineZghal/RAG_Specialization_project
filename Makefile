.PHONY: setup setup-st setup-qdrant ask ask-qdrant eval build-eval-set test lint qdrant-up qdrant-down notebook clean

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
setup: ## Install core dependencies for Level 1 (naive RAG)
	uv sync

setup-st: ## Also install the sentence-transformers embedding backend
	uv sync --extra sentence-transformers

setup-qdrant: ## Also install the Qdrant vector store client
	uv sync --extra qdrant

# ---------------------------------------------------------------------------
# Level 1 — Naive RAG
# (commands `cd` into 01-naive-rag so `python -m src.xxx` resolves the local
# `src/` package; they still run inside the single shared uv environment.)
# ---------------------------------------------------------------------------
ask: ## Ask a question against the in-memory pipeline: make ask Q="..."
	cd 01-naive-rag && uv run python -m src.cli ask "$(Q)"

ask-qdrant: ## Ask a question using the Qdrant-backed pipeline
	cd 01-naive-rag && uv run python examples/rag_with_qdrant.py "$(Q)"

build-eval-set: ## Derive shared/evaluation/*.jsonl from the open dataset
	cd 01-naive-rag && uv run python -m src.build_eval_set

eval: ## Run the Level 1 pipeline against shared/evaluation and report Recall@K
	cd 01-naive-rag && uv run python -m src.cli evaluate

test: ## Run the offline test suite (no network, no Ollama required)
	uv run pytest -v

lint: ## Static checks
	uv run ruff check .

qdrant-up: ## Start local Qdrant via Docker Compose
	docker compose up -d qdrant

qdrant-down: ## Stop Qdrant
	docker compose down

notebook: ## Launch Jupyter on the Level 1 notebooks
	uv run --with jupyter jupyter lab 01-naive-rag/notebooks

clean: ## Remove generated indexes and caches
	rm -rf 01-naive-rag/data/index .pytest_cache .ruff_cache
