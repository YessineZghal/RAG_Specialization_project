"""The agent's working state — a scratchpad, not a database: everything
needed to explain, after the fact, exactly what the agent did and why.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ToolCall:
    step: int
    tool: str
    tool_input: str
    result: object


@dataclass
class AgentState:
    question: str
    plan: list[str] = field(default_factory=list)
    tool_history: list[ToolCall] = field(default_factory=list)
    answer: str | None = None
    verified: bool = False
    stop_reason: str | None = None  # "answered" | "max_steps" | "sufficient_evidence"

    def evidence_text(self) -> str:
        """Flatten every tool result seen so far into one context block."""
        blocks = []
        for call in self.tool_history:
            blocks.append(f"[{call.tool}({call.tool_input!r})]\n{call.result}")
        return "\n\n".join(blocks)
