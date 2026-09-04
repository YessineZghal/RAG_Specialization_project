from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reasoning_common.strategy_classifier import classify_strategy


def test_classify_strategy_maps_simple_to_cot(fake_llm):
    llm = fake_llm(response="simple")
    assert classify_strategy("Is the sky blue?", llm) == "cot"


def test_classify_strategy_maps_comparative_to_tot(fake_llm):
    llm = fake_llm(response="comparative")
    assert classify_strategy("Is X more likely than Y?", llm) == "tot"


def test_classify_strategy_maps_combinatorial_to_got(fake_llm):
    llm = fake_llm(response="combinatorial")
    assert classify_strategy("Given both facts, does Z follow?", llm) == "got"


def test_classify_strategy_maps_multi_hop_to_hgot(fake_llm):
    llm = fake_llm(response="multi_hop")
    assert classify_strategy("Some multi-part question", llm) == "hgot"


def test_classify_strategy_accepts_the_hyphenated_spelling(fake_llm):
    llm = fake_llm(response="Category: multi-hop")
    assert classify_strategy("Some question", llm) == "hgot"


def test_classify_strategy_falls_open_to_cot_on_unparseable_response(fake_llm):
    llm = fake_llm(response="I'm not sure how to categorize this.")
    assert classify_strategy("Some question", llm) == "cot"


def test_classify_strategy_falls_open_to_cot_when_response_is_ambiguous(fake_llm):
    # mentions two categories at once -- genuinely ambiguous, not a guess
    llm = fake_llm(response="This could be simple or comparative.")
    assert classify_strategy("Some question", llm) == "cot"


def test_classify_strategy_is_case_insensitive(fake_llm):
    llm = fake_llm(response="COMBINATORIAL")
    assert classify_strategy("Some question", llm) == "got"
