"""Load PubMedQA and build a real per-question abstract corpus + labeled
yes/no/maybe question set for schema-constrained KG extraction.

Unlike StrategyQA (Level 8), PubMedQA does not share one pooled fact
corpus across questions -- every question already comes with its own
real PubMed abstract (`context.contexts`, 2-3 segments) identified by a
real `pubid`. That per-question abstract *is* the document used for
schema-constrained extraction (this level) and for the plain-embedding
baseline (`kag_eval`'s unconstrained graph-rag comparison) alike -- every
other sampled question's abstract becomes a real distractor document for
whichever one is being answered, the same "corpus of real documents,
answer only from your own gold one" shape every prior level's evaluation
has used, just built from PubMedQA's native per-question grounding
instead of a shared pool.

Nothing downloads at import time -- only when `prepare()` is called.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from .config import settings

logger = logging.getLogger(__name__)

CACHE_FILE = settings.cache_dir / "pubmedqa_subset.json"


@dataclass
class KagData:
    corpus: dict[str, str]  # doc_id (pubid) -> full abstract text
    questions: dict[str, dict]  # qid (pubid) -> {question, answer, gold_doc_ids, meshes}
    metadata: dict = field(default_factory=dict)

    def doc_ids(self) -> list[str]:
        return list(self.corpus.keys())


def _join_contexts(context: dict) -> str:
    """Join a PubMedQA row's labeled context segments into one abstract,
    e.g. "BACKGROUND: ... RESULTS: ..." -- keeps the real section labels
    since they are useful signal for extraction (a Study/Outcome split
    often falls cleanly along BACKGROUND/RESULTS/CONCLUSIONS)."""
    labels = context.get("labels") or [""] * len(context["contexts"])
    parts = []
    for label, segment in zip(labels, context["contexts"], strict=True):
        segment = " ".join(segment.split())
        if label:
            parts.append(f"{label}: {segment}")
        else:
            parts.append(segment)
    return " ".join(parts)


def prepare(n_documents: int | None = None, seed: int | None = None, force: bool = False) -> KagData:
    """Build (or load a cached) reduced PubMedQA sample: one real abstract
    per sampled question, plus its real yes/no/maybe ground truth."""
    n_documents = n_documents if n_documents is not None else settings.n_documents
    seed = seed if seed is not None else settings.dataset_seed

    if CACHE_FILE.exists() and not force:
        cached = json.loads(CACHE_FILE.read_text())
        meta = cached.get("metadata", {})
        if meta.get("n_documents") == n_documents and meta.get("seed") == seed:
            logger.info("Loaded cached PubMedQA subset from %s", CACHE_FILE)
            return KagData(corpus=cached["corpus"], questions=cached["questions"], metadata=meta)

    from datasets import load_dataset  # lazy: network + heavy

    logger.info("Downloading %s (%s)...", settings.hf_dataset_name, settings.hf_dataset_config)
    ds = load_dataset(settings.hf_dataset_name, settings.hf_dataset_config, split="train")
    ds = ds.shuffle(seed=seed)

    corpus: dict[str, str] = {}
    questions: dict[str, dict] = {}
    for row in ds:
        if len(questions) >= n_documents:
            break
        abstract = _join_contexts(row["context"])
        if not abstract.strip():
            continue  # defensive -- every real row has contexts, but don't trust blindly

        doc_id = str(row["pubid"])
        corpus[doc_id] = abstract
        questions[doc_id] = {
            "question": row["question"],
            "answer": row["final_decision"],  # real "yes" / "no" / "maybe"
            "gold_doc_ids": [doc_id],
            "meshes": row["context"].get("meshes", []),
            "long_answer": row["long_answer"],
        }

    metadata = {
        "dataset": settings.hf_dataset_name,
        "config": settings.hf_dataset_config,
        "n_documents": n_documents,
        "seed": seed,
        "n_corpus_docs": len(corpus),
        "n_questions_built": len(questions),
    }
    logger.info("Built PubMedQA subset: %s", metadata)

    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps({"corpus": corpus, "questions": questions, "metadata": metadata}))

    return KagData(corpus=corpus, questions=questions, metadata=metadata)
