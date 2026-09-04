"""Shared answer-parsing helpers -- every reasoning strategy in this
level (CoT, ToT, GoT) ends in the same place: a StrategyQA question needs
a yes/no verdict extracted from the model's free-text output. Written
once here, word-boundary-safe from the start, instead of three separate
ad-hoc parsers risking three separate versions of the substring bug this
repo has already hit more than once (Level 4's CRAG "relevant" inside
"irrelevant", Level 7's prompt-injection regex) -- `\\byes\\b` and
`\\bno\\b`, never a bare `"yes" in text`.
"""

from __future__ import annotations

import re

_YES_RE = re.compile(r"\byes\b|\btrue\b", re.IGNORECASE)
_NO_RE = re.compile(r"\bno\b|\bfalse\b", re.IGNORECASE)


def parse_yes_no(text: str) -> bool | None:
    """Return `True`/`False` for a clear yes/no verdict anywhere in
    `text`, or `None` if neither (or both, ambiguously) appear. Checked
    on the **last** line first, since that is where this level's prompts
    ask the model to put its final answer -- a reasoning trace that
    mentions "no" earlier while concluding "yes" must not be misread.
    """
    lines = [line for line in text.splitlines() if line.strip()]
    if lines:
        last_line_verdict = _parse_single_line(lines[-1])
        if last_line_verdict is not None:
            return last_line_verdict
    return _parse_single_line(text)


def _parse_single_line(text: str) -> bool | None:
    has_yes = bool(_YES_RE.search(text))
    has_no = bool(_NO_RE.search(text))
    if has_yes and not has_no:
        return True
    if has_no and not has_yes:
        return False
    return None  # neither, or both -- genuinely ambiguous, not a guess
