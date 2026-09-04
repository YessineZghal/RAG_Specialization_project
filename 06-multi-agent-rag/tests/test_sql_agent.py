from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from multiagent_common.loader import load_agent_class  # noqa: E402

# sql-agent/agent.py's module-level UnsafeQueryError/validate_sql aren't
# part of the SqlAgent class, so load the module directly (not just the
# class) via the same file-path mechanism the loader uses internally.
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("_test_sql_agent_module", Path(__file__).resolve().parent.parent / "sql-agent" / "agent.py")
_sql_agent_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sql_agent_module)
UnsafeQueryError = _sql_agent_module.UnsafeQueryError
validate_sql = _sql_agent_module.validate_sql


def test_validate_sql_accepts_select():
    assert validate_sql("SELECT * FROM film LIMIT 10") == "SELECT * FROM film LIMIT 10"


def test_validate_sql_adds_missing_limit():
    assert "LIMIT" in validate_sql("SELECT * FROM film")


@pytest.mark.parametrize(
    "sql", ["DROP TABLE film", "DELETE FROM film", "UPDATE film SET x=1",
            "SELECT * FROM film; DROP TABLE film"],
)
def test_validate_sql_rejects_dangerous_statements(sql):
    with pytest.raises(UnsafeQueryError):
        validate_sql(sql)
