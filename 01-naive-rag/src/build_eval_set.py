"""Derive `shared/evaluation/*.jsonl` from the open-source HF dataset.

This is the one script that both downloads data (the HF dataset, cached by
`datasets`) and writes into `shared/`, so every level can evaluate against
the exact same questions. Run it once, manually:

    cd 01-naive-rag
    uv run python -m src.build_eval_set

Produces (see ../../shared/README.md#evaluation-dataset):

- `questions.jsonl`         {"id", "question"}
- `expected_answers.jsonl`  {"id", "answer"}          (verbatim from the dataset)
- `expected_sources.jsonl`  {"id", "document_ids"}    (best-effort — see below)
- `difficult_queries.jsonl` hand-authored edge cases (not from the dataset)

`rag-mini-wikipedia`'s question-answer split does not ship an official
passage-id per question, so `expected_sources.jsonl` is a **best-effort**
approximation: for each question+answer pair we pick the passages whose
text has the highest word-overlap (Jaccard similarity, see
`shared/utils/text.py`) with the answer. This is intentionally naive —
fitting for Level 1 — and should be treated as a heuristic, not ground
truth. Level 2 onward can replace this with a better-labeled set once
retrieval itself is more advanced than "does this even work at all."
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import settings

REPO_ROOT = settings.level_dir.parent
sys.path.insert(0, str(REPO_ROOT))  # so `shared.utils.text` is importable

from shared.utils.text import jaccard_similarity  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Hand-authored edge cases — NOT derived from the dataset. These probe
# behavior the happy-path questions don't: refusal on out-of-scope
# questions, and questions phrased ambiguously.
DIFFICULT_QUERIES = [
    {
        "id": "diff-001",
        "question": "What is the CEO's personal phone number?",
        "note": "Not answerable from a Wikipedia-derived corpus — expects refusal.",
    },
    {
        "id": "diff-002",
        "question": "Summarize everything in the corpus in one sentence.",
        "note": "Under-specified / too broad for Top-K retrieval to serve well.",
    },
    {
        "id": "diff-003",
        "question": "According to the context, what will happen next year?",
        "note": "Asks the model to extrapolate beyond what any retrieved passage states.",
    },
]


def build(
    sample_size: int = 30,
    qa_config: str = "question-answer",
    qa_split: str = "test",
    corpus_limit: int | None = None,
    top_sources: int = 2,
    min_similarity: float = 0.15,
    seed: int = 42,
) -> None:
    from datasets import load_dataset  # lazy import: network + heavy

    from .ingest import load_from_hf_dataset

    logger.info("Loading question-answer pairs (%s/%s)...", qa_config, qa_split)
    qa_dataset = load_dataset(settings.hf_dataset_name, qa_config, split=qa_split)
    qa_dataset = qa_dataset.shuffle(seed=seed).select(
        range(min(sample_size, len(qa_dataset)))
    )

    logger.info("Loading text corpus to approximate expected sources...")
    passages = load_from_hf_dataset(config="text-corpus", split="passages", limit=corpus_limit)

    questions, expected_answers, expected_sources = [], [], []
    for row in qa_dataset:
        qid = f"q-{row.get('id', len(questions))}"
        question = row.get("question", "").strip()
        answer = row.get("answer", "").strip()
        if not question or not answer:
            continue

        scored = sorted(
            passages, key=lambda p: jaccard_similarity(p.text, answer), reverse=True
        )
        best = [p for p in scored[:top_sources] if jaccard_similarity(p.text, answer) >= min_similarity]

        questions.append({"id": qid, "question": question})
        expected_answers.append({"id": qid, "answer": answer})
        expected_sources.append({"id": qid, "document_ids": [p.id for p in best]})

    settings.shared_eval_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(settings.shared_eval_dir / "questions.jsonl", questions)
    _write_jsonl(settings.shared_eval_dir / "expected_answers.jsonl", expected_answers)
    _write_jsonl(settings.shared_eval_dir / "expected_sources.jsonl", expected_sources)
    _write_jsonl(settings.shared_eval_dir / "difficult_queries.jsonl", DIFFICULT_QUERIES)

    logger.info(
        "Wrote %d questions (+%d difficult queries) to %s",
        len(questions),
        len(DIFFICULT_QUERIES),
        settings.shared_eval_dir,
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=30)
    parser.add_argument("--corpus-limit", type=int, default=None)
    parser.add_argument("--top-sources", type=int, default=2)
    parser.add_argument("--min-similarity", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build(
        sample_size=args.sample_size,
        corpus_limit=args.corpus_limit,
        top_sources=args.top_sources,
        min_similarity=args.min_similarity,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
