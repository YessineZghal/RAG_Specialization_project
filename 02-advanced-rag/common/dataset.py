"""Load the open-source BeIR/scifact IR benchmark and reduce it to a corpus
size that indexes quickly on a laptop, while keeping every document any
test query is actually judged relevant to.

Unlike Level 1's `build_eval_set.py` (which had to *approximate* relevance
with a word-overlap heuristic, because rag-mini-wikipedia ships no official
qrels), scifact ships real, human-annotated relevance judgments — so every
metric Level 2 reports is measured against genuine ground truth.

Nothing downloads at import time — only when `prepare()` is called.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field

from .config import settings

logger = logging.getLogger(__name__)

CACHE_FILE = settings.cache_dir / "scifact_subset.json"


@dataclass
class ScifactData:
    corpus: dict[str, dict]  # doc_id -> {"title": str, "text": str}
    queries: dict[str, str]  # query_id -> question text
    qrels: dict[str, dict[str, int]]  # query_id -> {doc_id: relevance}
    metadata: dict = field(default_factory=dict)

    def corpus_text(self, doc_id: str) -> str:
        doc = self.corpus[doc_id]
        return f"{doc['title']} {doc['text']}".strip()

    def doc_ids(self) -> list[str]:
        return list(self.corpus.keys())


def prepare(size: int | None = None, seed: int | None = None, force: bool = False) -> ScifactData:
    """Build (or load a cached) reduced scifact corpus + test queries + qrels."""
    size = size if size is not None else settings.corpus_size
    seed = seed if seed is not None else settings.corpus_seed

    if CACHE_FILE.exists() and not force:
        cached = json.loads(CACHE_FILE.read_text())
        if cached.get("metadata", {}).get("size") == size and cached.get("metadata", {}).get(
            "seed"
        ) == seed:
            logger.info("Loaded cached scifact subset from %s", CACHE_FILE)
            return ScifactData(
                corpus=cached["corpus"], queries=cached["queries"], qrels=cached["qrels"],
                metadata=cached["metadata"],
            )

    from datasets import load_dataset  # lazy: network + heavy

    logger.info("Downloading %s (corpus + queries)...", settings.hf_corpus_dataset)
    corpus_ds = load_dataset(settings.hf_corpus_dataset, "corpus")["corpus"]
    queries_ds = load_dataset(settings.hf_corpus_dataset, "queries")["queries"]
    logger.info("Downloading %s (qrels)...", settings.hf_qrels_dataset)
    qrels_ds = load_dataset(settings.hf_qrels_dataset)["test"]

    # qrels use int query-id / corpus-id; corpus/queries use string `_id`.
    # Normalize everything to str immediately to avoid subtle join bugs.
    qrels: dict[str, dict[str, int]] = {}
    for row in qrels_ds:
        qid, cid, score = str(row["query-id"]), str(row["corpus-id"]), int(row["score"])
        qrels.setdefault(qid, {})[cid] = score

    relevant_doc_ids = {cid for rels in qrels.values() for cid in rels}
    all_doc_ids = [str(row["_id"]) for row in corpus_ds]

    rng = random.Random(seed)
    remaining = [d for d in all_doc_ids if d not in relevant_doc_ids]
    n_distractors = max(0, size - len(relevant_doc_ids))
    distractor_ids = set(rng.sample(remaining, min(n_distractors, len(remaining))))
    keep_ids = relevant_doc_ids | distractor_ids

    corpus = {
        str(row["_id"]): {"title": row["title"], "text": row["text"]}
        for row in corpus_ds
        if str(row["_id"]) in keep_ids
    }

    query_lookup = {str(row["_id"]): row["text"] for row in queries_ds}
    queries = {qid: query_lookup[qid] for qid in qrels if qid in query_lookup}
    # Drop qrels for any query we couldn't resolve text for (shouldn't happen,
    # but keeps the invariant "every qrels key has a query" airtight).
    qrels = {qid: rels for qid, rels in qrels.items() if qid in queries}

    metadata = {
        "corpus_dataset": settings.hf_corpus_dataset,
        "qrels_dataset": settings.hf_qrels_dataset,
        "size": size,
        "seed": seed,
        "n_corpus": len(corpus),
        "n_relevant_docs": len(relevant_doc_ids),
        "n_distractors": len(distractor_ids),
        "n_queries": len(queries),
    }
    logger.info("Built scifact subset: %s", metadata)

    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(
        json.dumps({"corpus": corpus, "queries": queries, "qrels": qrels, "metadata": metadata})
    )

    return ScifactData(corpus=corpus, queries=queries, qrels=qrels, metadata=metadata)
