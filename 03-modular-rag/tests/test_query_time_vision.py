"""Offline tests for multimodal-rag/query_time_vision.py -- a fake
vision client stands in for a real Ollama vision model (never exercised
against one in this environment, see vision_embedding.py's own
docstring), so these verify the *dispatch* logic (right image, right
question-specific prompt, one call per retrieved image) rather than a
real model's judgment about an image.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "multimodal-rag"))
from query_time_vision import analyze_image_for_query, vlm_enhanced_query


class FakeVisionClient:
    """Records every `(image_path, prompt)` pair it was called with, and
    returns a response that echoes the prompt back -- lets a test assert
    the *question* actually reached the model, not just that some call
    happened.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def describe_image(self, image_path: str, prompt: str = "") -> str:
        self.calls.append((image_path, prompt))
        return f"[analysis of {image_path} for prompt: {prompt}]"


def test_analyze_image_for_query_builds_a_question_specific_prompt_not_the_generic_one():
    client = FakeVisionClient()

    analyze_image_for_query("diagram.png", "How many layers does this architecture have?", vision_client=client)

    assert len(client.calls) == 1
    image_path, prompt = client.calls[0]
    assert image_path == "diagram.png"
    assert "How many layers does this architecture have?" in prompt


def test_the_same_image_gets_a_different_prompt_for_a_different_question():
    # The concrete improvement over vision_embedding.py: one fixed
    # ingestion-time description can never do this.
    client = FakeVisionClient()

    analyze_image_for_query("diagram.png", "What architecture is shown?", vision_client=client)
    analyze_image_for_query("diagram.png", "How many attention heads are there?", vision_client=client)

    prompts = [prompt for _path, prompt in client.calls]
    assert prompts[0] != prompts[1]
    assert "What architecture is shown?" in prompts[0]
    assert "How many attention heads are there?" in prompts[1]


def test_vlm_enhanced_query_calls_the_vision_client_once_per_retrieved_image():
    client = FakeVisionClient()
    images = {
        "img-a": {"path": "a.png", "page": 1, "caption": "Figure 1"},
        "img-b": {"path": "b.png", "page": 3, "caption": "Figure 2"},
        "img-c": {"path": "c.png", "page": 5, "caption": "Figure 3"},  # not retrieved -- must be skipped
    }

    analyses = vlm_enhanced_query("What is shown here?", ["img-a", "img-b"], images, vision_client=client)

    assert set(analyses.keys()) == {"img-a", "img-b"}
    called_paths = {path for path, _prompt in client.calls}
    assert called_paths == {"a.png", "b.png"}  # img-c's path was never called


def test_vlm_enhanced_query_skips_a_retrieved_id_with_no_known_image_metadata():
    client = FakeVisionClient()
    images = {"img-a": {"path": "a.png", "page": 1, "caption": "Figure 1"}}

    # "img-ghost" was somehow retrieved but this caller has no metadata for it
    analyses = vlm_enhanced_query("question", ["img-a", "img-ghost"], images, vision_client=client)

    assert set(analyses.keys()) == {"img-a"}
    assert len(client.calls) == 1


def test_vlm_enhanced_query_on_no_retrieved_images_makes_no_calls_at_all():
    client = FakeVisionClient()
    analyses = vlm_enhanced_query("question", [], {"img-a": {"path": "a.png", "page": 1, "caption": ""}}, vision_client=client)

    assert analyses == {}
    assert client.calls == []
