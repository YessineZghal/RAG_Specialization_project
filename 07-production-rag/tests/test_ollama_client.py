"""inference/ollama_client.py -- the local-dev inference backend.

Instantiation is safe offline (no network call happens until `.complete`
is invoked); replacing the wrapped OllamaLLM with a fake avoids needing a
real Ollama server for this file's own logic (matching the `LLMBackend`
protocol, delegating system/prompt through unchanged).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from inference.ollama_client import OllamaBackend  # noqa: E402


def test_ollama_backend_has_expected_name():
    backend = OllamaBackend()
    assert backend.name == "ollama"


def test_ollama_backend_delegates_complete_to_the_wrapped_llm(fake_llm):
    backend = OllamaBackend()
    backend._llm = fake_llm(responses=["a canned answer"])
    assert backend.complete("some prompt") == "a canned answer"
    assert backend._llm.calls == ["some prompt"]
