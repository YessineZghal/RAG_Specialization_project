"""Query-time visual re-analysis -- the gap named against RAG-Anything's
`vlm_enhanced` query mode in `../../missing_to_complite.md`: this level's
own `vision_embedding.py` only ever describes an image **once, at
ingestion time**, with one fixed, generic prompt. Whatever that first
description happened to mention is all any future query can ever see --
a question asking about a visual detail the original caption didn't
happen to cover has no path to get it, no matter how many times it's
asked.

This module re-analyzes a retrieved image **at query time**, with a
prompt built from the actual question being asked, not the fixed
"describe what's in this image" prompt `vision_embedding.py` uses at
ingestion. The same image can now get a different, question-specific
answer depending on what's actually being asked about it -- the same
image asked "what architecture is shown?" versus "how many layers are
in this diagram?" gets two different, targeted analyses instead of one
fixed description neither question tailored itself to.

Reuses `vision_embedding.OllamaVisionClient` rather than a second vision
client -- same real, documented Ollama vision-generation contract
(`generate(model=..., prompt=..., images=[...])`), just called again
per-query with a different prompt. Inherits the exact same disclosed
environment limitation: this machine's local Ollama installation could
not load the pulled vision model (`moondream`) through this interface
(see `vision_embedding.py`'s own module docstring for the full
disclosure) -- every function below is fully covered by offline tests
against a fake vision client, written to the real contract, but has
never actually produced a real query-time analysis from a real image.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vision_embedding import OllamaVisionClient

QUERY_TIME_PROMPT_TEMPLATE = (
    "Look at this image and answer the following question as specifically "
    "as possible, citing only what you can actually see in the image -- "
    "if the image does not show anything relevant to the question, say so "
    "plainly rather than guessing.\n\nQuestion: {question}"
)


def analyze_image_for_query(
    image_path: str,
    question: str,
    vision_client: OllamaVisionClient | None = None,
) -> str:
    """Re-analyze one image with a prompt built from `question`, instead
    of `vision_embedding.py`'s fixed, generic description prompt."""
    vision_client = vision_client or OllamaVisionClient()
    prompt = QUERY_TIME_PROMPT_TEMPLATE.format(question=question)
    return vision_client.describe_image(image_path, prompt=prompt)


def vlm_enhanced_query(
    question: str,
    retrieved_image_ids: list[str],
    images: dict[str, dict],
    vision_client: OllamaVisionClient | None = None,
) -> dict[str, str]:
    """For every retrieved image candidate, re-analyze it specifically
    for `question` at query time. Returns `{image_id: analysis_text}`,
    one real vision call per retrieved image -- deliberately not cached
    or reused across different questions, since the whole point is that
    the *same* image can get a *different* analysis depending on what's
    actually being asked.
    """
    vision_client = vision_client or OllamaVisionClient()
    analyses: dict[str, str] = {}
    for image_id in retrieved_image_ids:
        if image_id not in images:
            continue  # a retrieved id this caller doesn't actually have image metadata for
        analyses[image_id] = analyze_image_for_query(images[image_id]["path"], question, vision_client=vision_client)
    return analyses
