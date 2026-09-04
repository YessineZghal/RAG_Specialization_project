"""Long-term memory — persisted across *multiple* questions in a session,
unlike short-term memory which resets every question. A JSON-backed cache
keyed by semantic similarity: ask something close enough to a question
already answered, and the agent reuses the answer instead of re-running
the whole tool loop.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agentic_common.config import settings
from agentic_common.embed import OllamaEmbedder

MEMORY_FILE = settings.cache_dir / "long_term_memory.json"


class LongTermMemory:
    def __init__(self, embedder: OllamaEmbedder | None = None, similarity_threshold: float = 0.92) -> None:
        self.embedder = embedder or OllamaEmbedder()
        self.similarity_threshold = similarity_threshold
        self._entries: list[dict] = self._load()

    def _load(self) -> list[dict]:
        if MEMORY_FILE.exists():
            return json.loads(MEMORY_FILE.read_text())
        return []

    def _save(self) -> None:
        settings.cache_dir.mkdir(parents=True, exist_ok=True)
        MEMORY_FILE.write_text(json.dumps(self._entries))

    def remember(self, question: str, answer: str) -> None:
        vector = self.embedder.embed_one(question)
        self._entries.append({"question": question, "answer": answer, "vector": vector})
        self._save()

    def recall_similar(self, question: str) -> dict | None:
        if not self._entries:
            return None
        import numpy as np

        query_vector = np.array(self.embedder.embed_one(question))
        best, best_score = None, -1.0
        for entry in self._entries:
            vector = np.array(entry["vector"])
            score = float(
                np.dot(vector, query_vector) / (np.linalg.norm(vector) * np.linalg.norm(query_vector) + 1e-12)
            )
            if score > best_score:
                best, best_score = entry, score
        if best is not None and best_score >= self.similarity_threshold:
            return {"question": best["question"], "answer": best["answer"], "similarity": best_score}
        return None
