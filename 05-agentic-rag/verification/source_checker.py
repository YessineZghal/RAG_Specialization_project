"""Check that a generated answer's claims are actually traceable to the
retrieved evidence — citation checking, not fact-checking against the
outside world. An answer can be internally consistent with its evidence
and still be wrong if the evidence itself was bad; that's what
`answer_verifier.py` (checked against real ground truth) is for.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agentic_common.llm import OllamaLLM

CHECK_PROMPT = """Evidence:
{evidence}

Answer: {answer}

Is every claim in the answer supported by the evidence above, with nothing made up?
Respond with only one word: supported or unsupported.
Judgment:"""


def check_sources(evidence: str, answer: str, llm: OllamaLLM | None = None) -> bool:
    import re

    llm = llm or OllamaLLM()
    response = llm.complete(CHECK_PROMPT.format(evidence=evidence[:3000], answer=answer)).strip().lower()
    if re.search(r"\bunsupported\b", response):
        return False
    return bool(re.search(r"\bsupported\b", response))
