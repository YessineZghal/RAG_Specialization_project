"""Load real SEC 10-K question/answer/context triples across 69 real
public companies. Unlike prior levels' chunked-article corpora, each row
here is already a short, focused passage — no chunking needed — paired
with its ticker, so a supervisor can route "research company X" tasks by
ticker directly.

Nothing downloads at import time — only when `prepare()` is called.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from .config import settings

logger = logging.getLogger(__name__)

CACHE_FILE = settings.cache_dir / "financial_qa_subset.json"


@dataclass
class FinancialData:
    corpus: dict[str, dict]  # doc_id -> {"text", "ticker", "filing"}
    questions: dict[str, dict]  # question_id -> {question, answer, ticker, filing}
    metadata: dict = field(default_factory=dict)

    def doc_ids(self) -> list[str]:
        return list(self.corpus.keys())

    def by_ticker(self, ticker: str) -> list[str]:
        return [doc_id for doc_id, doc in self.corpus.items() if doc["ticker"] == ticker]


def prepare(n_questions: int | None = None, seed: int | None = None, force: bool = False) -> FinancialData:
    n_questions = n_questions if n_questions is not None else settings.n_questions
    seed = seed if seed is not None else settings.dataset_seed

    if CACHE_FILE.exists() and not force:
        cached = json.loads(CACHE_FILE.read_text())
        meta = cached.get("metadata", {})
        if meta.get("n_questions") == n_questions and meta.get("seed") == seed:
            logger.info("Loaded cached financial-qa-10K subset from %s", CACHE_FILE)
            return FinancialData(corpus=cached["corpus"], questions=cached["questions"], metadata=meta)

    from datasets import load_dataset  # lazy: network + heavy

    logger.info("Downloading %s...", settings.hf_dataset_name)
    ds = load_dataset(settings.hf_dataset_name, split="train")
    ds = ds.shuffle(seed=seed).select(range(min(n_questions, len(ds))))

    corpus: dict[str, dict] = {}
    questions: dict[str, dict] = {}
    for i, row in enumerate(ds):
        doc_id = f"{row['ticker']}-{row['filing']}-{i}"
        corpus[doc_id] = {"text": row["context"], "ticker": row["ticker"], "filing": row["filing"]}
        questions[f"q-{i}"] = {
            "question": row["question"],
            "answer": row["answer"],
            "ticker": row["ticker"],
            "filing": row["filing"],
            "gold_doc_id": doc_id,
        }

    metadata = {
        "n_questions": n_questions,
        "seed": seed,
        "n_corpus_docs": len(corpus),
        "n_tickers": len({d["ticker"] for d in corpus.values()}),
    }
    logger.info("Built financial-qa-10K subset: %s", metadata)

    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps({"corpus": corpus, "questions": questions, "metadata": metadata}))

    return FinancialData(corpus=corpus, questions=questions, metadata=metadata)
