"""Research Agent — a generalist that combines document retrieval *and*
graph lookup into one broader research pass, distinct from the narrow
Retrieval Agent (vector search only) and Graph Agent (graph only). This
is the agent a supervisor reaches for when a task needs synthesis across
more than one internal source, not just one.

Loads `RetrievalAgent`/`GraphAgent` by file path (see
`multiagent_common/loader.py`) rather than the usual "insert folder,
import agent" trick — every specialized agent module in this level is
named `agent.py`, so importing two of them into one process the normal
way would collide in `sys.modules`.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from multiagent_common.agent_result import AgentResult
from multiagent_common.llm import OllamaLLM
from multiagent_common.loader import load_agent_class

RetrievalAgent = load_agent_class("retrieval-agent", "RetrievalAgent")
GraphAgent = load_agent_class("graph-agent", "GraphAgent")

SYNTHESIS_PROMPT = """Document evidence:
{doc_evidence}

Graph facts:
{graph_evidence}

Task: {task}
Combine both sources into one concise research summary:"""


class ResearchAgent:
    name = "research-agent"

    def __init__(self, retrieval_agent, graph_agent, llm: OllamaLLM | None = None) -> None:
        self.retrieval_agent = retrieval_agent
        self.graph_agent = graph_agent
        self.llm = llm or OllamaLLM()

    def run(self, task: str) -> AgentResult:
        doc_result = self.retrieval_agent.run(task)
        graph_result = self.graph_agent.run(task)

        doc_evidence = "\n\n".join(doc_result.evidence) or "(none found)"
        graph_evidence = "\n".join(graph_result.evidence) or "(none found)"
        summary = self.llm.complete(
            SYNTHESIS_PROMPT.format(doc_evidence=doc_evidence, graph_evidence=graph_evidence, task=task)
        )
        return AgentResult(
            self.name, task, summary,
            evidence=doc_result.evidence + graph_result.evidence,
            success=doc_result.success or graph_result.success,
        )
