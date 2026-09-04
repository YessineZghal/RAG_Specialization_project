"""Load HotpotQA and build a working corpus + question set with real
ground truth: each question ships the exact paragraph titles ("supporting
facts") required to answer it — genuine multi-hop evidence, not a
heuristic (contrast with Level 1's word-overlap approximation).

Nothing downloads at import time — only when `prepare()` is called, and
results are cached to `data/cache/hotpot_subset.json` afterward.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from .config import settings

logger = logging.getLogger(__name__)

CACHE_FILE = settings.cache_dir / "hotpot_subset.json"


@dataclass
class HotpotData:
    corpus: dict[str, str]  # paragraph title -> paragraph text
    questions: dict[str, dict]  # question id -> {question, answer, type, level, supporting_titles}
    metadata: dict = field(default_factory=dict)

    def doc_ids(self) -> list[str]:
        return list(self.corpus.keys())


def prepare(n_questions: int | None = None, seed: int | None = None, force: bool = False) -> HotpotData:
    n_questions = n_questions if n_questions is not None else settings.n_questions
    seed = seed if seed is not None else settings.dataset_seed

    if CACHE_FILE.exists() and not force:
        cached = json.loads(CACHE_FILE.read_text())
        meta = cached.get("metadata", {})
        if meta.get("n_questions") == n_questions and meta.get("seed") == seed:
            logger.info("Loaded cached HotpotQA subset from %s", CACHE_FILE)
            return HotpotData(corpus=cached["corpus"], questions=cached["questions"], metadata=meta)

    from datasets import load_dataset  # lazy: network + heavy

    logger.info("Downloading %s (%s)...", settings.hf_dataset_name, settings.hf_dataset_config)
    # `train` (not `validation`) has the full easy/medium/hard spread —
    # HotpotQA's validation split is entirely "hard" level.
    ds = load_dataset(settings.hf_dataset_name, settings.hf_dataset_config, split="train")
    ds = ds.shuffle(seed=seed).select(range(min(n_questions, len(ds))))

    corpus: dict[str, str] = {}
    questions: dict[str, dict] = {}
    for row in ds:
        qid = row["id"]
        titles = row["context"]["title"]
        sentences_per_doc = row["context"]["sentences"]
        for title, sentences in zip(titles, sentences_per_doc, strict=True):
            corpus[title] = " ".join(sentences)

        supporting_titles = sorted(set(row["supporting_facts"]["title"]))
        questions[qid] = {
            "question": row["question"],
            "answer": row["answer"],
            "type": row["type"],  # "bridge" or "comparison"
            "level": row["level"],  # "easy" | "medium" | "hard"
            "supporting_titles": supporting_titles,
        }

    metadata = {
        "n_questions": n_questions,
        "seed": seed,
        "n_corpus_docs": len(corpus),
        "n_questions_actual": len(questions),
    }
    logger.info("Built HotpotQA subset: %s", metadata)

    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps({"corpus": corpus, "questions": questions, "metadata": metadata}))

    return HotpotData(corpus=corpus, questions=questions, metadata=metadata)
