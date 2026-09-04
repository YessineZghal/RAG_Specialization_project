"""Load StrategyQA and build a real fact corpus + labeled yes/no question
set for multi-step reasoning.

StrategyQA ships each question with its own real supporting `facts` blob
(a few sentences of real grounding text) rather than a shared Wikipedia
corpus like Levels 4-5 used. This module splits every question's facts
into individual sentences and pools them into one shared corpus across
the whole sample -- a question's *own* facts become its gold sentences
inside a corpus that also contains every other sampled question's facts
as real distractors, the same "union of contexts" shape Level 4's
HotpotQA corpus already used, just built from a different real source.

Nothing downloads at import time -- only when `prepare()` is called.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import re
from dataclasses import dataclass, field

from .config import settings

logger = logging.getLogger(__name__)

CACHE_FILE = settings.cache_dir / "strategyqa_subset.json"

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class ReasoningData:
    corpus: dict[str, str]  # sentence_id -> sentence text
    questions: dict[str, dict]  # qid -> {question, answer, gold_sentence_ids, term}
    metadata: dict = field(default_factory=dict)

    def doc_ids(self) -> list[str]:
        return list(self.corpus.keys())


def split_sentences(text: str) -> list[str]:
    normalized = " ".join(text.split())
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(normalized) if s.strip()]


def _sentence_id(text: str) -> str:
    return f"fact-{hashlib.sha1(text.encode()).hexdigest()[:10]}"


def prepare(n_questions: int | None = None, seed: int | None = None, force: bool = False) -> ReasoningData:
    """Build (or load a cached) reduced StrategyQA sample + its pooled
    fact corpus."""
    n_questions = n_questions if n_questions is not None else settings.n_questions
    seed = seed if seed is not None else settings.dataset_seed

    if CACHE_FILE.exists() and not force:
        cached = json.loads(CACHE_FILE.read_text())
        meta = cached.get("metadata", {})
        if meta.get("n_questions") == n_questions and meta.get("seed") == seed:
            logger.info("Loaded cached StrategyQA subset from %s", CACHE_FILE)
            return ReasoningData(corpus=cached["corpus"], questions=cached["questions"], metadata=meta)

    from datasets import load_dataset  # lazy: network + heavy

    logger.info("Downloading %s...", settings.hf_dataset_name)
    ds = load_dataset(settings.hf_dataset_name, split="train")
    ds = ds.shuffle(seed=seed)

    corpus: dict[str, str] = {}
    questions: dict[str, dict] = {}
    for row in ds:
        if len(questions) >= n_questions:
            break
        sentences = split_sentences(row["facts"])
        if not sentences:
            continue  # a handful of rows have empty facts -- skip, nothing to ground on

        gold_ids = []
        for sentence in sentences:
            sid = _sentence_id(sentence)
            corpus.setdefault(sid, sentence)
            gold_ids.append(sid)

        questions[row["qid"]] = {
            "question": row["question"],
            "answer": bool(row["answer"]),
            "term": row["term"],
            "gold_sentence_ids": gold_ids,
        }

    metadata = {
        "dataset": settings.hf_dataset_name,
        "n_questions": n_questions,
        "seed": seed,
        "n_corpus_sentences": len(corpus),
        "n_questions_built": len(questions),
    }
    logger.info("Built StrategyQA subset: %s", metadata)

    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps({"corpus": corpus, "questions": questions, "metadata": metadata}))

    return ReasoningData(corpus=corpus, questions=questions, metadata=metadata)


def sample_gsm8k(n: int = 20, seed: int | None = None) -> list[dict]:
    """A small, no-retrieval sample of GSM8K for calibrating CoT/ToT/GoT's
    raw reasoning quality separately from retrieval quality -- optional,
    used only by `notebooks/01_chain_of_thought.ipynb`'s calibration cell.
    Returns `[{"question": ..., "answer": <final numeric answer as str>}]`
    -- GSM8K's own answer field is "<reasoning>\\n#### <final number>";
    only the final number is kept, since that is what a real evaluation
    checks against, not the worked solution.
    """
    seed = seed if seed is not None else settings.dataset_seed

    from datasets import load_dataset  # lazy: network + heavy

    ds = load_dataset(settings.hf_calibration_dataset, "main", split="test")
    ds = ds.shuffle(seed=seed)

    sampled = []
    for row in ds:
        if len(sampled) >= n:
            break
        final_answer = row["answer"].rsplit("####", 1)[-1].strip()
        sampled.append({"question": row["question"], "answer": final_answer})
    return sampled
