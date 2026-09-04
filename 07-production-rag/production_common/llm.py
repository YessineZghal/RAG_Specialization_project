"""Thin local-LLM chat wrapper (Ollama) -- same interface as every prior level."""

from __future__ import annotations

from collections.abc import Iterator

from .config import settings


class OllamaLLM:
    def __init__(self, model: str | None = None, host: str | None = None) -> None:
        import ollama

        self.model = model or settings.ollama_chat_model
        self._client = ollama.Client(host=host or settings.ollama_host)

    def complete(self, prompt: str, system: str | None = None, temperature: float = 0.2) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        try:
            response = self._client.chat(
                model=self.model, messages=messages, options={"temperature": temperature}
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Could not reach Ollama at {settings.ollama_host} with model "
                f"'{self.model}'. Is `ollama serve` running and the model pulled? {exc}"
            ) from exc
        return response["message"]["content"].strip()

    def stream_complete(
        self, prompt: str, system: str | None = None, temperature: float = 0.2
    ) -> Iterator[str]:
        """Same request as `complete()`, but yields each token chunk as it
        arrives instead of waiting for and returning the full answer.
        Powers `api/routes.py`'s `/query/stream` -- see that module for why
        streaming exists (it changes what a caller experiences while
        waiting, not the total generation time itself).
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        try:
            stream = self._client.chat(
                model=self.model,
                messages=messages,
                options={"temperature": temperature},
                stream=True,
            )
            for chunk in stream:
                content = chunk["message"]["content"]
                if content:
                    yield content
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Could not reach Ollama at {settings.ollama_host} with model "
                f"'{self.model}'. Is `ollama serve` running and the model pulled? {exc}"
            ) from exc
