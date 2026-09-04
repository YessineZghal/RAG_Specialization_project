from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from sql_tool import UnsafeQueryError, validate_sql


def test_validate_sql_accepts_select():
    assert validate_sql("SELECT * FROM Products LIMIT 10") == "SELECT * FROM Products LIMIT 10"


def test_validate_sql_adds_missing_limit():
    assert "LIMIT" in validate_sql("SELECT * FROM Products")


@pytest.mark.parametrize(
    "sql", ["DROP TABLE Products", "DELETE FROM Products", "UPDATE Products SET x=1",
            "SELECT * FROM Products; DROP TABLE Products"],
)
def test_validate_sql_rejects_dangerous_statements(sql):
    with pytest.raises(UnsafeQueryError):
        validate_sql(sql)
