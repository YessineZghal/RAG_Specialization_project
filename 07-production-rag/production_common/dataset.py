"""Load SQuAD and build a deduplicated context corpus + labeled question
set with real ground-truth answers.

Nothing downloads at import time -- only when `prepare()` is called.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field

from .config import settings

logger = logging.getLogger(__name__)

CACHE_FILE = settings.cache_dir / "squad_subset.json"


@dataclass
class SquadData:
    corpus: dict[str, dict]  # doc_id -> {"title", "text"}
    questions: dict[str, dict]  # question_id -> {question, answers, gold_doc_id}
    metadata: dict = field(default_factory=dict)

    def doc_ids(self) -> list[str]:
        return list(self.corpus.keys())


def _doc_id(title: str, text: str) -> str:
    return f"{title}-{hashlib.sha1(text.encode()).hexdigest()[:10]}"


def prepare(n_contexts: int | None = None, seed: int | None = None, force: bool = False) -> SquadData:
    n_contexts = n_contexts if n_contexts is not None else settings.n_contexts
    seed = seed if seed is not None else settings.dataset_seed

    if CACHE_FILE.exists() and not force:
        cached = json.loads(CACHE_FILE.read_text())
        meta = cached.get("metadata", {})
        if meta.get("n_contexts") == n_contexts and meta.get("seed") == seed:
            logger.info("Loaded cached SQuAD subset from %s", CACHE_FILE)
            return SquadData(corpus=cached["corpus"], questions=cached["questions"], metadata=meta)

    from datasets import load_dataset  # lazy: network + heavy

    logger.info("Downloading %s...", settings.hf_dataset_name)
    ds = load_dataset(settings.hf_dataset_name, split="validation")
    ds = ds.shuffle(seed=seed)

    corpus: dict[str, dict] = {}
    questions: dict[str, dict] = {}
    for row in ds:
        if len(corpus) >= n_contexts and row["context"] not in {c["text"] for c in corpus.values()}:
            continue
        doc_id = _doc_id(row["title"], row["context"])
        corpus.setdefault(doc_id, {"title": row["title"], "text": row["context"]})
        questions[row["id"]] = {
            "question": row["question"],
            "answers": row["answers"]["text"],
            "gold_doc_id": doc_id,
        }
        if len(corpus) >= n_contexts and len(questions) >= n_contexts * 2:
            break

    metadata = {
        "n_contexts": n_contexts,
        "seed": seed,
        "n_corpus_docs": len(corpus),
        "n_questions": len(questions),
    }
    logger.info("Built SQuAD subset: %s", metadata)

    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps({"corpus": corpus, "questions": questions, "metadata": metadata}))

    return SquadData(corpus=corpus, questions=questions, metadata=metadata)
