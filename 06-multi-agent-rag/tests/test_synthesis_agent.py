from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from multiagent_common.loader import load_agent_class  # noqa: E402

SynthesisAgent = load_agent_class("synthesis-agent", "SynthesisAgent")


def test_synthesis_agent_combines_successful_results(fake_llm, fake_agent):
    llm = fake_llm(response="Combined answer citing both agents.")
    agent = SynthesisAgent(llm=llm)

    r1 = fake_agent("retrieval-agent", output="fact A", evidence=["ev1"]).run("task")
    r2 = fake_agent("sql-agent", output="fact B", evidence=["ev2"]).run("task")

    result = agent.run("task", [r1, r2])

    assert result.output == "Combined answer citing both agents."
    assert result.evidence == ["ev1", "ev2"]
    assert "retrieval-agent" in llm.calls[0]["prompt"]
    assert "sql-agent" in llm.calls[0]["prompt"]


def test_synthesis_agent_ignores_failed_results(fake_llm, fake_agent):
    llm = fake_llm(response="Answer from the one working agent.")
    agent = SynthesisAgent(llm=llm)

    good = fake_agent("retrieval-agent", output="fact A", success=True).run("task")
    bad = fake_agent("web-agent", output="nothing found", success=False).run("task")

    agent.run("task", [good, bad])

    assert "web-agent" not in llm.calls[0]["prompt"]
    assert "retrieval-agent" in llm.calls[0]["prompt"]


def test_synthesis_agent_returns_failure_when_all_results_failed(fake_llm, fake_agent):
    llm = fake_llm(response="should not be called")
    agent = SynthesisAgent(llm=llm)

    bad = fake_agent("web-agent", success=False).run("task")

    result = agent.run("task", [bad])

    assert result.success is False
    assert llm.calls == []
