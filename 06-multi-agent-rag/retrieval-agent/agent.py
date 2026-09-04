"""Retrieval Agent — searches the financial-qa-10K corpus and answers
narrowly from what it finds. One job: vector search + grounded answer,
nothing else.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from multiagent_common.agent_result import AgentResult  # noqa: E402
from multiagent_common.llm import OllamaLLM  # noqa: E402

ANSWER_PROMPT = """Context from company filings:
{context}

Task: {task}
Answer concisely using only the context above:"""


class RetrievalAgent:
    name = "retrieval-agent"

    def __init__(self, retriever, corpus: dict[str, dict], llm: OllamaLLM | None = None, top_k: int = 5) -> None:
        self.retriever = retriever
        self.corpus = corpus
        self.llm = llm or OllamaLLM()
        self.top_k = top_k

    def run(self, task: str) -> AgentResult:
        results = self.retriever.search(task, top_k=self.top_k)
        if not results:
            return AgentResult(self.name, task, "No relevant documents found.", success=False)

        evidence = [self.corpus[doc_id]["text"] for doc_id, _ in results]
        context = "\n\n".join(evidence)
        answer = self.llm.complete(ANSWER_PROMPT.format(context=context, task=task))
        return AgentResult(self.name, task, answer, evidence=evidence)
