from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sql-rag"))
from sql_guardrails import UnsafeQueryError, validate_sql


def test_validate_sql_accepts_select():
    assert validate_sql("SELECT * FROM Track LIMIT 10") == "SELECT * FROM Track LIMIT 10"


def test_validate_sql_adds_missing_limit():
    result = validate_sql("SELECT * FROM Track")
    assert "LIMIT 100" in result


def test_validate_sql_clamps_oversized_limit():
    result = validate_sql("SELECT * FROM Track LIMIT 999999")
    assert "LIMIT 100" in result
    assert "999999" not in result


def test_validate_sql_rejects_non_select():
    with pytest.raises(UnsafeQueryError):
        validate_sql("UPDATE Track SET Name = 'x'")


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE Track",
        "DELETE FROM Track",
        "INSERT INTO Track VALUES (1)",
        "ALTER TABLE Track ADD COLUMN x",
        "SELECT * FROM Track; DROP TABLE Track",
        "ATTACH DATABASE 'x' AS y",
        "PRAGMA table_info(Track)",
    ],
)
def test_validate_sql_rejects_dangerous_statements(sql):
    with pytest.raises(UnsafeQueryError):
        validate_sql(sql)


def test_validate_sql_rejects_multi_statement():
    with pytest.raises(UnsafeQueryError):
        validate_sql("SELECT * FROM Track; SELECT * FROM Album")


def test_validate_sql_case_insensitive_keyword_detection():
    with pytest.raises(UnsafeQueryError):
        validate_sql("select * from Track; drop table Track")
