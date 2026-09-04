"""Shared fixtures for Level 6's offline test suite — same fake-LLM
pattern as every prior level's conftest.py.
"""

from __future__ import annotations

import pytest


class FakeLLM:
    def __init__(self, response: str = "", responses: list[str] | None = None) -> None:
        self.response = response
        self.responses = list(responses) if responses else None
        self.calls: list[dict] = []

    def complete(self, prompt: str, system: str | None = None, temperature: float = 0.2) -> str:
        self.calls.append({"prompt": prompt, "system": system, "temperature": temperature})
        if self.responses:
            return self.responses.pop(0)
        return self.response


@pytest.fixture
def fake_llm():
    return FakeLLM


class FakeAgent:
    """Duck-types any specialized agent's `.run(task) -> AgentResult` interface."""

    def __init__(self, name: str, output: str = "ok", success: bool = True, evidence: list[str] | None = None) -> None:
        self.name = name
        self._output = output
        self._success = success
        self._evidence = evidence or []
        self.calls: list[str] = []

    def run(self, task: str):
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from multiagent_common.agent_result import AgentResult

        self.calls.append(task)
        return AgentResult(self.name, task, self._output, evidence=self._evidence, success=self._success)


@pytest.fixture
def fake_agent():
    return FakeAgent
