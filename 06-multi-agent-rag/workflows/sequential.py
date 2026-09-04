"""Run a set of agents one after another, each one optionally seeing the
task augmented with prior agents' findings — slower than parallel
execution, but lets a later agent build on an earlier one's answer.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from multiagent_common.agent_result import AgentResult  # noqa: E402


def run_sequential(task: str, agents: dict[str, object], carry_context: bool = False) -> dict[str, AgentResult]:
    results: dict[str, AgentResult] = {}
    context_so_far = ""

    for name, agent in agents.items():
        agent_task = f"{task}\n\nPrior findings:\n{context_so_far}" if carry_context and context_so_far else task
        result = agent.run(agent_task)
        results[name] = result
        if result.success:
            context_so_far += f"\n[{name}]: {result.output}"

    return results
