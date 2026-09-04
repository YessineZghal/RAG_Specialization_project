"""inference/vllm_client.py -- the production (GPU) inference backend.

Per the module's own docstring, this has never been exercised against a
real vLLM server (no GPU available in this environment) -- unit-tested
here against a mocked HTTP response to at least pin the OpenAI-compatible
request/response contract it's written to.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from inference.vllm_client import VLLMBackend  # noqa: E402


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self._payload


def test_vllm_backend_has_expected_name():
    assert VLLMBackend().name == "vllm"


def test_vllm_backend_posts_openai_compatible_chat_payload(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):  # noqa: A002
        captured["url"] = url
        captured["json"] = json
        return _FakeResponse({"choices": [{"message": {"content": "  a vllm answer  "}}]})

    monkeypatch.setattr("requests.post", fake_post)

    backend = VLLMBackend(base_url="http://localhost:8000", model="my-model")
    result = backend.complete("What is RAG?", system="Be concise.")

    assert result == "a vllm answer"
    assert captured["url"] == "http://localhost:8000/v1/chat/completions"
    assert captured["json"]["model"] == "my-model"
    assert captured["json"]["messages"] == [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "What is RAG?"},
    ]


def test_vllm_backend_omits_system_message_when_not_provided(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):  # noqa: A002
        captured["json"] = json
        return _FakeResponse({"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr("requests.post", fake_post)

    VLLMBackend().complete("prompt only, no system")
    assert captured["json"]["messages"] == [{"role": "user", "content": "prompt only, no system"}]
