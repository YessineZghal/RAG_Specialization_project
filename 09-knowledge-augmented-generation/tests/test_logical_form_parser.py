from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "reasoning-engine"))
from logical_form_parser import parse_logical_form


def test_parse_logical_form_reads_a_well_formed_response(fake_llm):
    payload = {
        "operators": ["retrieval", "kg_reasoning"],
        "focus_hint": "diabetes",
        "numeric_comparison": None,
    }
    llm = fake_llm(response=json.dumps(payload))

    form = parse_logical_form("Does the study concern diabetes?", llm)

    assert set(form.operators) == {"retrieval", "kg_reasoning", "language_reasoning"}
    assert form.focus_hint == "diabetes"
    assert form.fell_back is False


def test_parse_logical_form_always_includes_language_reasoning(fake_llm):
    payload = {"operators": ["retrieval"], "focus_hint": None, "numeric_comparison": None}
    llm = fake_llm(response=json.dumps(payload))

    form = parse_logical_form("some question", llm)

    assert "language_reasoning" in form.operators


def test_parse_logical_form_keeps_a_valid_numeric_comparison(fake_llm):
    payload = {
        "operators": ["numerical_calculation"],
        "focus_hint": "the trial",
        "numeric_comparison": {"attribute": "size", "op": ">", "value": 500},
    }
    llm = fake_llm(response=json.dumps(payload))

    form = parse_logical_form("Was the population larger than 500?", llm)

    assert form.numeric_comparison == {"attribute": "size", "op": ">", "value": 500.0}


def test_parse_logical_form_falls_back_on_unparseable_response(fake_llm):
    llm = fake_llm(response="I'm not sure how to answer that in JSON.")

    form = parse_logical_form("some question", llm)

    assert form.fell_back is True
    assert set(form.operators) == {"retrieval", "language_reasoning"}


def test_parse_logical_form_falls_back_when_no_operator_survives_filtering(fake_llm):
    # every proposed operator is bogus -- must not silently produce an
    # empty operator tuple
    payload = {"operators": ["made_up_operator"], "focus_hint": None, "numeric_comparison": None}
    llm = fake_llm(response=json.dumps(payload))

    form = parse_logical_form("some question", llm)

    assert form.fell_back is True


def test_parse_logical_form_drops_an_invalid_numeric_comparison_shape(fake_llm):
    payload = {
        "operators": ["numerical_calculation"],
        "focus_hint": "x",
        "numeric_comparison": {"attribute": "size"},  # missing required "op"/"value"
    }
    llm = fake_llm(response=json.dumps(payload))

    form = parse_logical_form("some question", llm)

    # the whole response fails Pydantic validation -> fail-open default
    assert form.fell_back is True
