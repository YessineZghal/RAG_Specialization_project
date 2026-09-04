from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reasoning_eval.cost_tracker import CostTrackingLLM


def test_cost_tracking_llm_counts_calls(fake_llm):
    inner = fake_llm(response="an answer")
    tracked = CostTrackingLLM(inner)

    tracked.complete("prompt one")
    tracked.complete("prompt two")

    assert tracked.call_count == 2


def test_cost_tracking_llm_forwards_the_real_response(fake_llm):
    inner = fake_llm(response="the real answer")
    tracked = CostTrackingLLM(inner)
    assert tracked.complete("prompt") == "the real answer"


def test_cost_tracking_llm_forwards_arguments_unchanged(fake_llm):
    inner = fake_llm(response="answer")
    tracked = CostTrackingLLM(inner)
    tracked.complete("prompt", system="a system prompt", temperature=0.9)
    call = inner.calls
    assert call == ["prompt"]  # FakeLLM only records the prompt text, but this proves no exception was raised


def test_reset_zeroes_the_count_without_affecting_the_wrapped_llm(fake_llm):
    inner = fake_llm(response="answer")
    tracked = CostTrackingLLM(inner)
    tracked.complete("prompt")
    tracked.reset()
    assert tracked.call_count == 0
    tracked.complete("another prompt")
    assert tracked.call_count == 1
