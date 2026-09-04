from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "workflows"))
from sequential import run_sequential
from parallel import run_parallel


def test_run_sequential_calls_every_agent(fake_agent):
    agents = {"a": fake_agent("a", output="out-a"), "b": fake_agent("b", output="out-b")}
    results = run_sequential("task", agents)
    assert results["a"].output == "out-a"
    assert results["b"].output == "out-b"


def test_run_sequential_carries_context_forward(fake_agent):
    a = fake_agent("a", output="finding from a")
    b = fake_agent("b", output="out-b")
    run_sequential("task", {"a": a, "b": b}, carry_context=True)
    # b's task should have been augmented with a's finding
    assert "finding from a" in b.calls[0]


def test_run_sequential_does_not_carry_context_when_disabled(fake_agent):
    a = fake_agent("a", output="finding from a")
    b = fake_agent("b", output="out-b")
    run_sequential("task", {"a": a, "b": b}, carry_context=False)
    assert b.calls[0] == "task"


class SlowAgent:
    def __init__(self, name: str, delay: float) -> None:
        self.name = name
        self.delay = delay

    def run(self, task: str):
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from multiagent_common.agent_result import AgentResult

        time.sleep(self.delay)
        return AgentResult(self.name, task, f"done-{self.name}")


def test_run_parallel_is_actually_concurrent_not_sequential():
    # Two agents that each sleep 0.3s: sequential would take ~0.6s,
    # real concurrency should take close to ~0.3s.
    agents = {"a": SlowAgent("a", 0.3), "b": SlowAgent("b", 0.3)}
    results, elapsed = run_parallel("task", agents)
    assert results["a"].output == "done-a"
    assert results["b"].output == "done-b"
    assert elapsed < 0.55  # well under the sequential sum, allowing overhead


def test_run_parallel_works_when_called_from_inside_a_running_event_loop():
    """The exact failure mode caught by actually executing this level's
    notebook 03 in Jupyter (which runs its own event loop) and NOT caught
    by a bare-script smoke test: `asyncio.run()` raises if a loop is
    already running. Simulate that here without needing a real kernel.
    """
    import asyncio

    async def _call_from_within_a_loop():
        return run_parallel("task", {"a": SlowAgent("a", 0.01)})

    results, elapsed = asyncio.run(_call_from_within_a_loop())
    assert results["a"].output == "done-a"
    assert elapsed >= 0
