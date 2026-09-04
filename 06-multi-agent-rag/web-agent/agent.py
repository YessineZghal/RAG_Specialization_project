"""Web Agent — live web search (`ddgs`, no API key) for information not
in the local corpus (e.g. anything after the sampled 10-K filings' dates).
Same mechanism as Levels 3 and 5's web tools, reimplemented here to keep
this level self-contained.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from multiagent_common.agent_result import AgentResult  # noqa: E402
from multiagent_common.llm import OllamaLLM  # noqa: E402

ANSWER_PROMPT = """Web search results:
{context}

Task: {task}
Answer concisely using the results above:"""


class WebAgent:
    name = "web-agent"

    def __init__(self, llm: OllamaLLM | None = None, max_results: int = 3) -> None:
        self.llm = llm or OllamaLLM()
        self.max_results = max_results

    def run(self, task: str) -> AgentResult:
        from ddgs import DDGS

        with DDGS() as ddgs:
            results = list(ddgs.text(task, max_results=self.max_results))

        if not results:
            return AgentResult(self.name, task, "No web results found.", success=False)

        evidence = [f"[{r.get('title', '')}] {r.get('body', '')}" for r in results]
        context = "\n\n".join(evidence)
        answer = self.llm.complete(ANSWER_PROMPT.format(context=context, task=task))
        return AgentResult(self.name, task, answer, evidence=evidence)
