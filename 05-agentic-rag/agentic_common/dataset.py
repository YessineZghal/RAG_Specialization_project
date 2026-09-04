"""Load TriviaQA and build a chunked corpus + question set with real
ground truth: each question ships its source Wikipedia article title and
a list of acceptable answer *aliases* (multiple valid phrasings) — genuine
ground truth for automatic answer verification, not a heuristic.

Nothing downloads at import time — only when `prepare()` is called, and
results are cached to `data/cache/trivia_subset.json` afterward.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from .config import settings

logger = logging.getLogger(__name__)

CACHE_FILE = settings.cache_dir / "trivia_subset.json"


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Naive fixed-size word chunking -- same strategy as every prior level."""
    words = text.split()
    if not words:
        return []
    stride = chunk_size - chunk_overlap
    chunks = []
    for start in range(0, len(words), stride):
        window = words[start : start + chunk_size]
        if not window:
            break
        chunks.append(" ".join(window))
        if start + chunk_size >= len(words):
            break
    return chunks


@dataclass
class TriviaData:
    corpus: dict[str, dict]  # chunk_id -> {"text", "article_title", "question_id"}
    questions: dict[str, dict]  # question_id -> {question, answer, aliases, article_title}
    metadata: dict = field(default_factory=dict)

    def article_chunk_ids(self, article_title: str) -> list[str]:
        return [cid for cid, c in self.corpus.items() if c["article_title"] == article_title]

    def get_document(self, article_title: str) -> str:
        """Reassemble a full article's text from its chunks, in order."""
        ids = sorted(self.article_chunk_ids(article_title), key=lambda cid: int(cid.split("-c")[-1]))
        return " ".join(self.corpus[cid]["text"] for cid in ids)


def prepare(n_questions: int | None = None, seed: int | None = None, force: bool = False) -> TriviaData:
    n_questions = n_questions if n_questions is not None else settings.n_questions
    seed = seed if seed is not None else settings.dataset_seed

    if CACHE_FILE.exists() and not force:
        cached = json.loads(CACHE_FILE.read_text())
        meta = cached.get("metadata", {})
        if meta.get("n_questions") == n_questions and meta.get("seed") == seed:
            logger.info("Loaded cached TriviaQA subset from %s", CACHE_FILE)
            return TriviaData(corpus=cached["corpus"], questions=cached["questions"], metadata=meta)

    from datasets import load_dataset  # lazy: network + heavy

    logger.info("Downloading %s (%s)...", settings.hf_dataset_name, settings.hf_dataset_config)
    ds = load_dataset(settings.hf_dataset_name, settings.hf_dataset_config, split="validation")
    ds = ds.shuffle(seed=seed).select(range(min(n_questions, len(ds))))

    corpus: dict[str, dict] = {}
    questions: dict[str, dict] = {}
    for row in ds:
        titles = row["entity_pages"]["title"]
        contexts = row["entity_pages"]["wiki_context"]
        if not titles or not contexts:
            continue  # some rows have no linked Wikipedia page -- skip
        article_title = titles[0]
        article_text = contexts[0]

        chunks = chunk_text(article_text, settings.chunk_size, settings.chunk_overlap)
        for i, chunk in enumerate(chunks):
            chunk_id = f"{article_title}-c{i}"
            corpus[chunk_id] = {"text": chunk, "article_title": article_title, "question_id": row["question_id"]}

        answer = row["answer"]
        questions[row["question_id"]] = {
            "question": row["question"],
            "answer": answer["value"],
            "aliases": sorted(set(answer["normalized_aliases"])),
            "article_title": article_title,
        }

    metadata = {
        "n_questions": n_questions,
        "seed": seed,
        "n_corpus_chunks": len(corpus),
        "n_questions_actual": len(questions),
    }
    logger.info("Built TriviaQA subset: %s", metadata)

    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps({"corpus": corpus, "questions": questions, "metadata": metadata}))

    return TriviaData(corpus=corpus, questions=questions, metadata=metadata)
