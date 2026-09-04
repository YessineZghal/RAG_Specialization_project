"""The local-development inference backend — a thin wrapper matching this
level's `LLMBackend` protocol (see `vllm_client.py`), so `api/main.py` can
swap backends via one config value (`INFERENCE_BACKEND=ollama|vllm`)
without touching route logic.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from production_common.llm import OllamaLLM


class OllamaBackend:
    name = "ollama"

    def __init__(self, model: str | None = None) -> None:
        self._llm = OllamaLLM(model=model)

    def complete(self, prompt: str, system: str | None = None) -> str:
        return self._llm.complete(prompt, system=system)
