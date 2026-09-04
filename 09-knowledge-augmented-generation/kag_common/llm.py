"""Thin local-LLM chat wrapper (Ollama) -- same interface as every prior level."""

from __future__ import annotations

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
