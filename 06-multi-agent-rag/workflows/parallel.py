"""Run a set of agents concurrently with `asyncio.gather` — the pattern
this repo's own plan document shows for multi-agent workflows. Each
agent's `.run()` is synchronous (plain HTTP calls to Ollama/SQLite/the
web), so it's wrapped with `asyncio.to_thread` to actually get wall-clock
concurrency on I/O-bound work without rewriting every agent as async.
"""

from __future__ import annotations

import asyncio
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from multiagent_common.agent_result import AgentResult  # noqa: E402


async def _run_one(name: str, agent, task: str) -> tuple[str, AgentResult]:
    result = await asyncio.to_thread(agent.run, task)
    return name, result


async def run_parallel_async(task: str, agents: dict[str, object]) -> dict[str, AgentResult]:
    pairs = await asyncio.gather(*(_run_one(name, agent, task) for name, agent in agents.items()))
    return dict(pairs)


def run_parallel(task: str, agents: dict[str, object]) -> tuple[dict[str, AgentResult], float]:
    """Sync wrapper + wall-clock timing, so the parallel-vs-sequential
    comparison in this level's notebook can report real numbers, not just
    "should be faster."

    Jupyter's kernel already runs its own asyncio event loop, so a plain
    `asyncio.run(...)` here raises "cannot be called from a running event
    loop" the moment this is called from a notebook cell -- a real
    failure caught by actually executing the notebook, not just testing
    this function from a bare script (where no loop is running yet, so
    the bug doesn't show up). Fix: if a loop is already running, execute
    the coroutine in a separate thread with its own fresh loop instead.
    """
    start = time.time()
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        results = asyncio.run(run_parallel_async(task, agents))  # no loop running -- the common script/test case
    else:
        with ThreadPoolExecutor(max_workers=1) as pool:
            results = pool.submit(asyncio.run, run_parallel_async(task, agents)).result()
    return results, time.time() - start
