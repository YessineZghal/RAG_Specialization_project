"""The production inference backend — vLLM (or SGLang) serves an
OpenAI-compatible `/v1/chat/completions` endpoint, typically behind
LiteLLM (see `litellm_config.yaml`) for model routing across backends.

**Honesty note:** this repo runs entirely on CPU/Ollama — there is no GPU
here to actually start a vLLM server against, so this client is written
to the documented vLLM/OpenAI-compatible API contract and unit-tested
against a mocked HTTP response (see `tests/test_vllm_client.py`), but has
never been exercised against a real vLLM server. Everything else in this
level that claims to work has actually been run; this one file is the
one deliberate exception, and it's flagged here rather than hidden.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from production_common.config import settings  # noqa: E402


class VLLMBackend:
    name = "vllm"

    def __init__(self, base_url: str | None = None, model: str | None = None) -> None:
        self.base_url = (base_url or "http://localhost:8000").rstrip("/")
        self.model = model or settings.ollama_chat_model

    def complete(self, prompt: str, system: str | None = None) -> str:
        import requests

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={"model": self.model, "messages": messages, "temperature": 0.2},
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
