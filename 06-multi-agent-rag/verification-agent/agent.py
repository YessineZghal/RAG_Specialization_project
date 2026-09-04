"""Verification Agent — checks whether a claim is actually supported by
the evidence pooled from the other agents, before synthesis treats it as
trustworthy. Same discipline as Level 5's source checker, applied here to
the *combined* output of multiple specialized agents rather than one.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from multiagent_common.agent_result import AgentResult
from multiagent_common.llm import OllamaLLM

CHECK_PROMPT = """Evidence:
{evidence}

Claim: {claim}

Is this claim supported by the evidence above, with nothing made up?
Respond with only one word: supported or unsupported.
Judgment:"""


class VerificationAgent:
    name = "verification-agent"

    def __init__(self, llm: OllamaLLM | None = None) -> None:
        self.llm = llm or OllamaLLM()

    def run(self, task: str, claim: str, evidence: list[str]) -> AgentResult:
        evidence_text = "\n\n".join(evidence)[:4000] or "(no evidence provided)"
        response = self.llm.complete(CHECK_PROMPT.format(evidence=evidence_text, claim=claim)).strip().lower()

        if re.search(r"\bunsupported\b", response):
            verified = False
        else:
            verified = bool(re.search(r"\bsupported\b", response))

        output = "supported" if verified else "unsupported"
        return AgentResult(self.name, task, output, evidence=evidence, success=verified)
