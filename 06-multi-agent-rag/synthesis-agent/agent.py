"""Synthesis Agent — combines multiple specialized agents' results into
one coherent, cited final answer. This is the one agent that reads
*other agents' outputs* as its input, not raw evidence.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from multiagent_common.agent_result import AgentResult
from multiagent_common.llm import OllamaLLM

SYNTHESIS_PROMPT = """Multiple specialized agents investigated this task. Combine their findings
into one coherent answer, noting which agent each piece of information came from.
Ignore any agent that failed or found nothing relevant.

Task: {task}

{agent_findings}

Combined answer:"""


class SynthesisAgent:
    name = "synthesis-agent"

    def __init__(self, llm: OllamaLLM | None = None) -> None:
        self.llm = llm or OllamaLLM()

    def run(self, task: str, results: list[AgentResult]) -> AgentResult:
        usable = [r for r in results if r.success]
        if not usable:
            return AgentResult(self.name, task, "No agent produced usable findings for this task.", success=False)

        findings = "\n\n".join(f"[{r.agent_name}]: {r.output}" for r in usable)
        combined = self.llm.complete(SYNTHESIS_PROMPT.format(task=task, agent_findings=findings))
        all_evidence = [e for r in usable for e in r.evidence]
        return AgentResult(self.name, task, combined, evidence=all_evidence)
