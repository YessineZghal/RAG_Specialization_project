"""The loader is what prevents Level 6's real `agent.py`-named-everywhere
collision (see multiagent_common/loader.py's docstring) -- verify it
actually loads two DIFFERENT `agent.py` files' classes correctly in the
same process, which a naive `sys.path.insert + import agent` cannot do.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from multiagent_common.loader import load_agent_class


def test_loader_loads_distinct_agent_py_files_without_collision():
    RetrievalAgent = load_agent_class("retrieval-agent", "RetrievalAgent")
    SqlAgent = load_agent_class("sql-agent", "SqlAgent")

    assert RetrievalAgent.name == "retrieval-agent"
    assert SqlAgent.name == "sql-agent"
    assert RetrievalAgent is not SqlAgent


def test_loader_returns_usable_classes_with_correct_run_signature():
    VerificationAgent = load_agent_class("verification-agent", "VerificationAgent")
    import inspect

    params = list(inspect.signature(VerificationAgent.run).parameters)
    assert params == ["self", "task", "claim", "evidence"]
