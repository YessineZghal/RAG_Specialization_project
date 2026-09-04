"""production_eval/ragas_eval.py -- hand-rolled faithfulness + answer relevance.

Includes a regression test for the real crash this level hit while
running a live 15-question evaluation batch: `_extract_claims`'s fallback
regex-extraction path called `json.loads` without its own try/except, so
genuinely malformed LLM JSON output (missing comma, unterminated string,
...) crashed the whole run instead of just that one question.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from production_eval.ragas_eval import _extract_claims, answer_relevance, faithfulness


def test_extract_claims_parses_clean_json_array(fake_llm):
    llm = fake_llm(responses=['["Paris is the capital of France.", "France is in Europe."]'])
    claims = _extract_claims("Paris is the capital of France, which is in Europe.", llm)
    assert claims == ["Paris is the capital of France.", "France is in Europe."]


def test_extract_claims_recovers_json_embedded_in_prose(fake_llm):
    llm = fake_llm(responses=['Sure, here you go:\n["Claim A", "Claim B"]\nHope that helps.'])
    claims = _extract_claims("some answer", llm)
    assert claims == ["Claim A", "Claim B"]


def test_extract_claims_degrades_to_empty_list_on_malformed_json_instead_of_crashing(fake_llm):
    # The real failure mode: an array-shaped substring that still isn't
    # valid JSON (unterminated string). Must not raise.
    llm = fake_llm(responses=['["Claim A, "Claim B]'])
    claims = _extract_claims("some answer", llm)
    assert claims == []


def test_extract_claims_degrades_to_empty_list_when_no_array_present_at_all(fake_llm):
    llm = fake_llm(responses=["I cannot break this into claims."])
    claims = _extract_claims("some answer", llm)
    assert claims == []


def test_faithfulness_scores_fraction_of_supported_claims(fake_llm):
    llm = fake_llm(
        responses=[
            '["Claim one.", "Claim two."]',  # claim extraction
            "yes",  # claim one supported
            "no",  # claim two not supported
        ]
    )
    result = faithfulness("some answer", "some context", llm=llm)
    assert result["score"] == 0.5
    assert result["n_supported"] == 1


def test_faithfulness_is_zero_when_no_claims_extracted(fake_llm):
    llm = fake_llm(responses=["not a json array"])
    result = faithfulness("some answer", "some context", llm=llm)
    assert result["score"] == 0.0
    assert result["claims"] == []


def test_answer_relevance_averages_similarity_of_reverse_questions(fake_llm, fake_embedder):
    llm = fake_llm(responses=["What is the capital of France?\nWhich city is France's capital?"])
    vectors = {
        "What is the capital of France?": [1.0, 0.0],
        "Which city is France's capital?": [1.0, 0.0],
    }
    embedder = fake_embedder(vectors)
    result = answer_relevance("What is the capital of France?", "Paris.", llm=llm, embedder=embedder, n=2)
    assert len(result["reverse_questions"]) == 2
    assert result["score"] > 0.99  # near-identical vectors here


def test_answer_relevance_is_zero_when_llm_returns_no_questions(fake_llm, fake_embedder):
    llm = fake_llm(responses=[""])
    embedder = fake_embedder({})
    result = answer_relevance("some question", "some answer", llm=llm, embedder=embedder)
    assert result["score"] == 0.0
    assert result["reverse_questions"] == []
