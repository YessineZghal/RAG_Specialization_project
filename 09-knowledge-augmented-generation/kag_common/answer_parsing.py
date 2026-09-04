"""Three-way yes/no/maybe verdict parsing for PubMedQA -- the same
word-boundary-safe approach as 08-reasoning-strategies'
`reasoning_common/answer_parsing.py`, extended from two labels to three
since PubMedQA's real `final_decision` field is yes/no/maybe, not a
binary. "maybe" is checked first: a text that says "the evidence is
mixed, so maybe" contains no bare "yes" or "no" token, but a text that
hedges with "not clearly yes or no, likely maybe" could otherwise trip
both the yes and no patterns -- checking maybe first and returning
immediately keeps that case unambiguous rather than falling through to
the "both matched -> None" rule below.
"""

from __future__ import annotations

import re

_MAYBE_RE = re.compile(r"\bmaybe\b|\bunclear\b|\binconclusive\b", re.IGNORECASE)
_YES_RE = re.compile(r"\byes\b", re.IGNORECASE)
_NO_RE = re.compile(r"\bno\b", re.IGNORECASE)


def parse_yes_no_maybe(text: str) -> str | None:
    """Return `"yes"`, `"no"`, `"maybe"`, or `None` if the verdict is
    genuinely ambiguous. Checked on the **last** non-empty line first,
    since every prompt in this level asks for the final verdict there."""
    lines = [line for line in text.splitlines() if line.strip()]
    if lines:
        last_line_verdict = _parse_single_line(lines[-1])
        if last_line_verdict is not None:
            return last_line_verdict
    return _parse_single_line(text)


def _parse_single_line(text: str) -> str | None:
    if _MAYBE_RE.search(text):
        return "maybe"
    has_yes = bool(_YES_RE.search(text))
    has_no = bool(_NO_RE.search(text))
    if has_yes and not has_no:
        return "yes"
    if has_no and not has_yes:
        return "no"
    return None  # neither, or both -- genuinely ambiguous, not a guess
