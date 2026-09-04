"""Vision-based image retrieval — fixes this level's own disclosed
limitation. `image_retrieval.py` can only find an image if a "Figure N:"
caption happens to sit near it in the PDF text; an uncaptioned image, or
one whose caption shares no vocabulary with the query, is invisible to
it. This module makes an image findable from its actual visual content
instead, so a caption is no longer required.

**How it works**: a vision-language model looks at the image and writes
a plain-text description of what is actually in it (a diagram, a chart,
labeled parts, and so on). That description is then embedded with the
same text embedder every other retriever in this level already uses
(`modular_common.embed.OllamaEmbedder`), and indexed the normal way.

**Disclosed approximation, stated plainly**: a real CLIP-style system
embeds an image's pixels directly into the same vector space as text,
with no text step in between. Ollama's embeddings endpoint takes text
only — there is no local, pixel-level embedding model available through
it. What is available is a vision-*language* model: one that can look at
an image and describe it in words. This module chains two real, separate
model calls (describe, then embed) instead of one true visual embedding
call. It is a genuine two-step approximation, not visual embedding
itself, and disclosed here for exactly that reason — the same standard
this repo holds every other simplification to.

**Never exercised against a real vision model in this environment**: a
vision model (`moondream`) was pulled specifically to build and test this
module, but this machine's local Ollama installation could not load it
through the same `ollama.Client` interface every other module in this
repo uses (confirmed directly: `ollama.Client(...).generate(model="moondream", ...)`
returns "model not found" even though the model's files are genuinely
present on disk — a pre-existing local installation issue, not a bug in
this code, and not reproduced by any other model this repo already
depends on). Every function below is fully covered by offline tests
against a fake vision client and is written to Ollama's real, documented
vision-generation contract (`generate(model=..., prompt=..., images=[...])`),
but has never actually produced a real description from a real image —
the same disclosed gap as `07-production-rag/inference/vllm_client.py`'s
"no GPU available" note.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modular_common.config import settings  # noqa: E402

DESCRIBE_PROMPT = (
    "Describe exactly what is shown in this image in two or three plain "
    "sentences. Focus on concrete, specific details -- labeled parts, the "
    "kind of diagram or chart it is, any text visible within it -- rather "
    "than a general description. This will be used to make the image "
    "findable by a text search engine, so specific, distinctive wording "
    "matters more than a polished summary."
)


class OllamaVisionClient:
    """Thin wrapper around Ollama's vision-generation endpoint. Kept
    separate from `modular_common.llm.OllamaLLM` because a vision call
    takes an `images` argument the plain chat client has no use for.
    """

    def __init__(self, model: str | None = None, host: str | None = None) -> None:
        import ollama

        self.model = model or settings.ollama_vision_model
        self._client = ollama.Client(host=host or settings.ollama_host)

    def describe_image(self, image_path: str, prompt: str = DESCRIBE_PROMPT) -> str:
        try:
            response = self._client.generate(model=self.model, prompt=prompt, images=[image_path])
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Could not reach Ollama at {settings.ollama_host} with vision model "
                f"'{self.model}'. Is it pulled (`ollama pull {self.model}`)? {exc}"
            ) from exc
        return response["response"].strip()


def describe_images(
    image_paths: dict[str, str],
    vision_client: OllamaVisionClient | None = None,
) -> dict[str, str]:
    """`image_id -> generated description`, one vision call per image."""
    vision_client = vision_client or OllamaVisionClient()
    return {image_id: vision_client.describe_image(path) for image_id, path in image_paths.items()}


def build_enriched_index_text(images: dict[str, dict], descriptions: dict[str, str]) -> dict[str, str]:
    """Combine each image's real caption (when it has one) with its
    generated visual description into one piece of text to index.

    An image with a real caption gets both: the caption's own wording
    (which may use vocabulary a query matches directly) plus the visual
    description (which covers content the caption does not mention). An
    image with no real caption -- `image_retrieval.extract_images()`'s
    placeholder text for that case is "(uncaptioned image on page N)" --
    is indexed by its visual description alone, which is the entire point
    of this module: that image is now findable at all.
    """
    enriched: dict[str, str] = {}
    for image_id, image in images.items():
        caption = image.get("caption", "")
        description = descriptions.get(image_id, "")
        if caption.startswith("(uncaptioned image"):
            enriched[image_id] = description
        else:
            enriched[image_id] = f"{caption} {description}".strip()
    return enriched


def build_vision_retriever(
    images: dict[str, dict],
    embedder=None,
    vision_client: OllamaVisionClient | None = None,
    cache_name: str = "vision_images",
):
    """Full pipeline: describe every image, combine with its caption (if
    any), embed the combined text, and return a `VectorRetriever` over it
    -- the vision-aware counterpart to `image_retrieval.build_image_retriever`.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "multi-retriever"))
    from vector_retriever import VectorRetriever

    descriptions = describe_images({k: v["path"] for k, v in images.items()}, vision_client=vision_client)
    enriched_texts = build_enriched_index_text(images, descriptions)
    retriever = VectorRetriever.from_texts(enriched_texts, embedder=embedder, cache_name=cache_name)
    return retriever, descriptions
