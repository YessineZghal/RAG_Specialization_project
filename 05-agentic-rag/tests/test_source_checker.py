from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "verification"))
from source_checker import check_sources


def test_check_sources_true_when_supported(fake_llm):
    llm = fake_llm(response="supported")
    assert check_sources("evidence text", "an answer", llm=llm) is True


def test_check_sources_false_when_unsupported(fake_llm):
    llm = fake_llm(response="This claim is unsupported by the evidence.")
    assert check_sources("evidence text", "an answer", llm=llm) is False
