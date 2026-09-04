"""Short-term memory — the current question's own scratchpad. Just a thin,
explicit wrapper around `AgentState.tool_history` so the rest of the code
doesn't reach into agent internals directly.
"""

from __future__ import annotations


class ShortTermMemory:
    def __init__(self) -> None:
        self._entries: list[dict] = []

    def add(self, tool: str, tool_input: str, result: object) -> None:
        self._entries.append({"tool": tool, "input": tool_input, "result": result})

    def as_context(self) -> str:
        return "\n\n".join(f"[{e['tool']}({e['input']!r})]\n{e['result']}" for e in self._entries)

    def __len__(self) -> int:
        return len(self._entries)
