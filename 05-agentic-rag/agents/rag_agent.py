"""The agent loop: understand the task, decide whether to retrieve, pick a
tool, inspect the result, decide if evidence is enough, retrieve again if
not, then produce a cited answer — all as one small hand-rolled ReAct-style
loop (no LangGraph/agent framework), so the actual mechanism stays visible.
A line-based `ACTION: / INPUT:` format is used instead of JSON: local
models follow it far more reliably (see this level's README for the
measured comparison).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agentic_common.config import settings
from agentic_common.llm import OllamaLLM

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "planning"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "iterative-retrieval"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "verification"))
from agent_planner import make_plan
from loop import is_evidence_sufficient
from source_checker import check_sources

from .state import AgentState, ToolCall

TOOL_DESCRIPTIONS = """- vector_search(query): search the local document corpus for relevant passages
- get_document(title): fetch the full text of a specific article by its exact title
- sql_query(question): answer a question by querying the Northwind business database
- web_search(query): search the live web for information not in the local corpus
- graph_search(entity): look up facts connected to a named entity in the knowledge graph
- finish: you have enough information; give the final answer now"""

DECIDE_PROMPT = """You are a research agent with these tools:
{tools}

Question: {question}
Plan: {plan}

Actions taken so far:
{history}

Decide your NEXT single action. Respond in EXACTLY this format, nothing else:
ACTION: <one of vector_search, get_document, sql_query, web_search, graph_search, finish>
INPUT: <the input for that tool, or your final answer if ACTION is finish>"""

ANSWER_PROMPT = """Evidence gathered:
{evidence}

Question: {question}
Answer concisely and only from the evidence above:"""


class RAGAgent:
    def __init__(self, tools: dict[str, object], llm: OllamaLLM | None = None, max_steps: int | None = None) -> None:
        self.tools = tools
        self.llm = llm or OllamaLLM()
        self.max_steps = max_steps or settings.max_steps

    def run(self, question: str) -> AgentState:
        state = AgentState(question=question)
        state.plan = make_plan(question, llm=self.llm)

        for step in range(1, self.max_steps + 1):
            action, tool_input = self._decide(state)

            if action == "finish":
                state.answer = tool_input or self._generate_answer(state)
                state.stop_reason = "answered"
                break

            tool = self.tools.get(action)
            if tool is None:
                state.tool_history.append(ToolCall(step, "invalid_action", action, f"No such tool: {action!r}"))
                continue

            try:
                result = tool(tool_input)
            except Exception as exc:  # noqa: BLE001 - a broken tool call shouldn't crash the agent
                result = f"Tool error: {exc}"
            state.tool_history.append(ToolCall(step, action, tool_input, result))

            if is_evidence_sufficient(question, state.evidence_text(), llm=self.llm):
                state.answer = self._generate_answer(state)
                state.stop_reason = "sufficient_evidence"
                break
        else:
            state.stop_reason = "max_steps"
            state.answer = self._generate_answer(state) if state.tool_history else (
                "I could not gather enough information to answer this question."
            )

        if state.tool_history:
            state.verified = check_sources(state.evidence_text(), state.answer, llm=self.llm)
        return state

    def _decide(self, state: AgentState) -> tuple[str, str]:
        history = "\n".join(f"{i+1}. {c.tool}({c.tool_input!r}) -> {str(c.result)[:200]}" for i, c in enumerate(state.tool_history)) or "(none yet)"
        prompt = DECIDE_PROMPT.format(
            tools=TOOL_DESCRIPTIONS, question=state.question, plan="; ".join(state.plan), history=history
        )
        raw = self.llm.complete(prompt)
        return self._parse_decision(raw)

    @staticmethod
    def _parse_decision(raw: str) -> tuple[str, str]:
        action_match = re.search(r"ACTION:\s*(\w+)", raw, re.IGNORECASE)
        input_match = re.search(r"INPUT:\s*(.+)", raw, re.IGNORECASE | re.DOTALL)
        action = action_match.group(1).strip().lower() if action_match else "finish"
        tool_input = input_match.group(1).strip() if input_match else raw.strip()
        return action, tool_input

    def _generate_answer(self, state: AgentState) -> str:
        return self.llm.complete(ANSWER_PROMPT.format(evidence=state.evidence_text() or "(none)", question=state.question))
