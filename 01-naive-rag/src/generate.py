"""Prompt construction and answer generation.

`OllamaGenerator` (default) calls a local, open-source LLM served by
Ollama — no API key, nothing leaves your machine. `ExtractiveGenerator` is
a zero-dependency fallback that skips the LLM entirely and returns the
top retrieved chunk verbatim; it exists so the pipeline, and its tests,
can run end-to-end without Ollama installed (useful for CI, and for
seeing raw retrieval quality with generation held constant).
"""

from __future__ import annotations

from typing import Protocol

from .config import settings
from .schema import RetrievedChunk

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the "
    "provided context. If the context does not contain the answer, say "
    "'I don't know based on the provided context.' Do not use outside "
    "knowledge. Keep answers concise and cite which source number you used."
)


def build_prompt(question: str, sources: list[RetrievedChunk]) -> str:
    """Assemble the naive RAG prompt: numbered context blocks + question."""
    context_blocks = "\n\n".join(
        f"[Source {i}] (doc: {retrieved.chunk.document_id})\n{retrieved.chunk.text}"
        for i, retrieved in enumerate(sources, start=1)
    )
    return (
        f"Context:\n{context_blocks}\n\n"
        f"Question:\n{question}\n\n"
        "Answer using only the context above, citing source numbers like [Source 1]:"
    )


class Generator(Protocol):
    def generate(self, question: str, sources: list[RetrievedChunk]) -> str: ...


class OllamaGenerator:
    """Generation via a local Ollama chat model (default backend)."""

    def __init__(self, model: str | None = None, host: str | None = None) -> None:
        try:
            import ollama
        except ImportError as exc:  # pragma: no cover - defensive
            raise RuntimeError(
                "The 'ollama' package is required for OllamaGenerator. "
                "Run `uv sync` from the repo root."
            ) from exc

        self._client = ollama.Client(host=host or settings.ollama_host)
        self.model = model or settings.ollama_chat_model

    def generate(self, question: str, sources: list[RetrievedChunk]) -> str:
        prompt = build_prompt(question, sources)
        try:
            response = self._client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
        except Exception as exc:  # noqa: BLE001 - surface a friendlier message
            raise RuntimeError(
                f"Could not reach Ollama at {settings.ollama_host} to generate an answer "
                f"with model '{self.model}'. Is `ollama serve` running, and have you run "
                f"`ollama pull {self.model}`? Original error: {exc}"
            ) from exc
        return response["message"]["content"].strip()


class ExtractiveGenerator:
    """No LLM: returns the top retrieved chunk verbatim.

    Useful for isolating retrieval quality from generation quality, and as
    the default generator in the offline test suite.
    """

    def generate(self, question: str, sources: list[RetrievedChunk]) -> str:
        if not sources:
            return "I don't know based on the provided context."
        top = sources[0]
        return f"[Source 1] {top.chunk.text}"


def get_generator(backend: str | None = None, **kwargs) -> Generator:
    """Factory: `get_generator("ollama")` or `get_generator("extractive")`."""
    backend = (backend or settings.generation_backend).lower()
    if backend == "ollama":
        return OllamaGenerator(**kwargs)
    if backend == "extractive":
        return ExtractiveGenerator()
    raise ValueError(f"Unknown generation backend '{backend}'. Expected 'ollama' or 'extractive'.")
