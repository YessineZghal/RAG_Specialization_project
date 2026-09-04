from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from multiagent_common.loader import load_agent_class  # noqa: E402

VerificationAgent = load_agent_class("verification-agent", "VerificationAgent")


def test_verification_agent_supported(fake_llm):
    llm = fake_llm(response="supported")
    agent = VerificationAgent(llm=llm)
    result = agent.run("task", "claim", ["evidence"])
    assert result.success is True
    assert result.output == "supported"


def test_verification_agent_unsupported(fake_llm):
    llm = fake_llm(response="This claim is unsupported by the evidence.")
    agent = VerificationAgent(llm=llm)
    result = agent.run("task", "claim", ["evidence"])
    assert result.success is False


def test_verification_agent_handles_empty_evidence(fake_llm):
    llm = fake_llm(response="unsupported")
    agent = VerificationAgent(llm=llm)
    result = agent.run("task", "claim", [])
    assert result.success is False
    assert "(no evidence provided)" in llm.calls[0]["prompt"]
