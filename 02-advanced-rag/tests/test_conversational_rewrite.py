from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "query-transformations"))
from conversational_rewrite import (  # noqa: E402
    conversational_search,
    format_history,
    rewrite_with_history,
)


class FakeRetriever:
    def __init__(self):
        self.calls: list[str] = []

    def search(self, query: str, top_k: int):
        self.calls.append(query)
        return [("a", 0.9)][:top_k]


def test_format_history_renders_speaker_and_text_per_line():
    history = [("user", "What is our refund policy?"), ("assistant", "30 days.")]
    assert format_history(history) == "user: What is our refund policy?\nassistant: 30 days."


def test_rewrite_with_history_returns_the_question_unchanged_with_no_history(fake_llm):
    llm = fake_llm(response="should never be used")
    result = rewrite_with_history("What is our refund policy?", history=[], llm=llm)
    assert result == "What is our refund policy?"
    assert llm.calls == []  # no history -- no LLM call needed at all


def test_rewrite_with_history_resolves_a_pronoun_using_prior_turns(fake_llm):
    history = [("user", "What is our refund policy?"), ("assistant", "30 days.")]
    llm = fake_llm(response='"What is the refund policy for enterprise customers?"')

    rewritten = rewrite_with_history("And for enterprise customers?", history, llm=llm)

    assert rewritten == "What is the refund policy for enterprise customers?"
    prompt = llm.calls[0]["prompt"]
    assert "refund policy" in prompt
    assert "And for enterprise customers?" in prompt


def test_conversational_search_retrieves_on_the_rewritten_question(fake_llm):
    history = [("user", "What is our refund policy?"), ("assistant", "30 days.")]
    llm = fake_llm(response="What is the refund policy for enterprise customers?")
    retriever = FakeRetriever()

    result = conversational_search("And for enterprise customers?", history, retriever, llm=llm)

    assert result["rewritten_question"] == "What is the refund policy for enterprise customers?"
    assert retriever.calls == ["What is the refund policy for enterprise customers?"]


def test_conversational_search_with_no_history_retrieves_on_the_original_question(fake_llm):
    llm = fake_llm(response="should never be used")
    retriever = FakeRetriever()

    result = conversational_search("What is our refund policy?", [], retriever, llm=llm)

    assert result["rewritten_question"] == "What is our refund policy?"
    assert retriever.calls == ["What is our refund policy?"]
