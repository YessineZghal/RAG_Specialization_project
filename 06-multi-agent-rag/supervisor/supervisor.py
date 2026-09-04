"""Supervisor — decides which specialized agent(s) a task should go to,
and delegates. Unlike Level 3's single-choice router, a task here can
reasonably go to *more than one* agent at once (a company research
question might need both `retrieval-agent` and `graph-agent`).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from multiagent_common.agent_result import AgentResult  # noqa: E402
from multiagent_common.llm import OllamaLLM  # noqa: E402

AGENT_DESCRIPTIONS = """- retrieval-agent: searches real SEC 10-K filing excerpts for company facts
- sql-agent: queries the Sakila DVD rental business database
- web-agent: searches the live web for current information
- graph-agent: looks up company facts in a knowledge graph
- research-agent: combines document + graph research for broader company questions"""

ROUTE_PROMPT = """You are a supervisor delegating a business research task to specialized agents:
{agents}

Select ONLY the agent(s) needed for this task, comma-separated, from exactly these names:
retrieval-agent, sql-agent, web-agent, graph-agent, research-agent

Task: {task}
Agents:"""

VALID_AGENTS = ("retrieval-agent", "sql-agent", "web-agent", "graph-agent", "research-agent")


class Supervisor:
    def __init__(self, agents: dict[str, object], llm: OllamaLLM | None = None) -> None:
        self.agents = agents
        self.llm = llm or OllamaLLM()

    def route(self, task: str) -> list[str]:
        response = self.llm.complete(ROUTE_PROMPT.format(agents=AGENT_DESCRIPTIONS, task=task)).lower()
        selected = [name for name in VALID_AGENTS if re.search(rf"\b{re.escape(name)}\b", response)]
        return selected or ["research-agent"]  # safe default: the generalist

    def delegate(self, task: str) -> dict[str, AgentResult]:
        agent_names = [name for name in self.route(task) if name in self.agents]
        return {name: self.agents[name].run(task) for name in agent_names}
