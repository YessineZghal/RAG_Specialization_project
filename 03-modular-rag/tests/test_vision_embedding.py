"""Offline tests for multimodal-rag/vision_embedding.py -- a fake vision
client stands in for a real Ollama vision model (never exercised against
one in this environment, see the module's own docstring), so these tests
verify the pipeline logic, not a real model's judgment about an image.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "multimodal-rag"))
from vision_embedding import (
    build_enriched_index_text,
    build_vision_retriever,
    describe_images,
)


class FakeVisionClient:
    """Duck-types `OllamaVisionClient.describe_image()` with a scripted
    description per image path, or one fixed description for every image
    if only a single string is given.
    """

    def __init__(self, description: str = "", by_path: dict[str, str] | None = None) -> None:
        self.description = description
        self.by_path = by_path or {}
        self.calls: list[str] = []

    def describe_image(self, image_path: str, prompt: str = "") -> str:
        self.calls.append(image_path)
        return self.by_path.get(image_path, self.description)


def test_describe_images_calls_the_vision_client_once_per_image():
    vision_client = FakeVisionClient(
        by_path={"img1.png": "A bar chart.", "img2.png": "A diagram of a neural network."}
    )
    image_paths = {"page1-img0": "img1.png", "page2-img0": "img2.png"}

    descriptions = describe_images(image_paths, vision_client=vision_client)

    assert descriptions == {"page1-img0": "A bar chart.", "page2-img0": "A diagram of a neural network."}
    assert vision_client.calls == ["img1.png", "img2.png"]


def test_build_enriched_index_text_combines_caption_and_description():
    images = {"page3-img0": {"path": "img.png", "page": 3, "caption": "Figure 1: The model architecture."}}
    descriptions = {"page3-img0": "A diagram showing an encoder and a decoder connected by arrows."}

    enriched = build_enriched_index_text(images, descriptions)

    assert enriched["page3-img0"] == (
        "Figure 1: The model architecture. "
        "A diagram showing an encoder and a decoder connected by arrows."
    )


def test_build_enriched_index_text_uses_description_alone_for_uncaptioned_images():
    # This is the exact real limitation this module exists to fix: an
    # uncaptioned image has nothing else to be indexed by.
    images = {"page7-img0": {"path": "img.png", "page": 7, "caption": "(uncaptioned image on page 7)"}}
    descriptions = {"page7-img0": "A scatter plot with two clusters of points, one red and one blue."}

    enriched = build_enriched_index_text(images, descriptions)

    assert enriched["page7-img0"] == "A scatter plot with two clusters of points, one red and one blue."
    assert "uncaptioned" not in enriched["page7-img0"]


def test_uncaptioned_image_becomes_retrievable_by_its_visual_content(fake_embedder):
    # The concrete improvement over image_retrieval.py: a query matching
    # only the image's real content -- never its (nonexistent) caption --
    # must now find it.
    images = {
        "captioned-img": {"path": "a.png", "page": 1, "caption": "Figure 1: Revenue growth by quarter."},
        "uncaptioned-img": {"path": "b.png", "page": 5, "caption": "(uncaptioned image on page 5)"},
    }
    vision_client = FakeVisionClient(
        by_path={
            "a.png": "A bar chart showing revenue growth by quarter.",
            "b.png": "A scatter plot comparing reaction time and temperature in degrees.",
        }
    )

    retriever, descriptions = build_vision_retriever(
        images, embedder=fake_embedder, vision_client=vision_client, cache_name="test-vision"
    )

    assert descriptions["uncaptioned-img"] == "A scatter plot comparing reaction time and temperature in degrees."

    results = retriever.search("scatter plot reaction time temperature", top_k=2)
    top_id = results[0][0]
    assert top_id == "uncaptioned-img"
