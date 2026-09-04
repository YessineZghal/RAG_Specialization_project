"""BGE reranker — same cross-encoder mechanism as `cross_encoder.py`, using
BAAI's `bge-reranker-base` instead of `ms-marco-MiniLM`. Kept as a separate
class (rather than just a different argument) to make explicit that
rerankers, like embedders and generators elsewhere in this repo, are a
swappable component with real quality/latency trade-offs between vendors —
see `notebooks/04_reranking.ipynb` for a head-to-head comparison.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.config import settings
from reranking.cross_encoder import CrossEncoderReranker


class BGEReranker(CrossEncoderReranker):
    def __init__(self, model_name: str | None = None) -> None:
        super().__init__(model_name=model_name or settings.bge_reranker_model)
