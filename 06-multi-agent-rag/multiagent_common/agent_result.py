"""The common return shape every specialized agent produces — so the
supervisor, workflows, and synthesis agent can treat results from
completely different backends (vector search, SQL, web, graph)
uniformly.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AgentResult:
    agent_name: str
    task: str
    output: str
    evidence: list[str] = field(default_factory=list)
    success: bool = True
    error: str | None = None
