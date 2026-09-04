"""The full multi-agent workflow: supervisor routes -> selected agents run
in parallel -> each result gets verified -> synthesis agent combines only
the verified findings into one final answer.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from multiagent_common.agent_result import AgentResult

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parallel import run_parallel
from sequential import run_sequential


def run_supervisor_graph(
    task: str,
    supervisor,
    verification_agent,
    synthesis_agent,
    use_parallel: bool = True,
) -> dict:
    agent_names = supervisor.route(task)
    selected_agents = {name: supervisor.agents[name] for name in agent_names if name in supervisor.agents}

    if use_parallel:
        results, elapsed = run_parallel(task, selected_agents)
    else:
        import time

        start = time.time()
        results = run_sequential(task, selected_agents)
        elapsed = time.time() - start

    verified: dict[str, bool] = {}
    for name, result in results.items():
        if not result.success:
            verified[name] = False
            continue
        check = verification_agent.run(task, result.output, result.evidence)
        verified[name] = check.success

    verified_results = [r for name, r in results.items() if verified.get(name)]
    synthesis: AgentResult = synthesis_agent.run(task, verified_results or list(results.values()))

    return {
        "task": task,
        "routed_to": agent_names,
        "results": results,
        "verified": verified,
        "synthesis": synthesis,
        "elapsed_seconds": elapsed,
    }
