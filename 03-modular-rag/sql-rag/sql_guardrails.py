"""Guardrails for LLM-generated SQL before it ever touches the database.

An LLM asked to "write SQL for this question" will, occasionally, write
something destructive, multi-statement, or simply too broad (no LIMIT on
a 3,500-row table). None of this is malicious — it's just what happens
when free-form text generation is allowed to produce something that gets
executed. Validate first, always.
"""

from __future__ import annotations

import re

FORBIDDEN_KEYWORDS = (
    "DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "ATTACH", "DETACH",
    "PRAGMA", "VACUUM", "REPLACE", "CREATE", "TRUNCATE", "GRANT",
)
MAX_ROWS = 100


class UnsafeQueryError(ValueError):
    pass


def validate_sql(sql: str) -> str:
    """Raise `UnsafeQueryError` if `sql` fails any guardrail; otherwise
    return a (possibly LIMIT-clamped) safe version of it.
    """
    cleaned = sql.strip().rstrip(";")

    if ";" in cleaned:
        raise UnsafeQueryError("Multi-statement SQL is not allowed.")

    if not re.match(r"^\s*SELECT\b", cleaned, re.IGNORECASE):
        raise UnsafeQueryError("Only SELECT statements are allowed.")

    for keyword in FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{keyword}\b", cleaned, re.IGNORECASE):
            raise UnsafeQueryError(f"Forbidden keyword detected: {keyword}")

    return _ensure_limit(cleaned)


def _ensure_limit(sql: str, max_rows: int = MAX_ROWS) -> str:
    match = re.search(r"\bLIMIT\s+(\d+)\b", sql, re.IGNORECASE)
    if match:
        if int(match.group(1)) > max_rows:
            return re.sub(r"\bLIMIT\s+\d+\b", f"LIMIT {max_rows}", sql, flags=re.IGNORECASE)
        return sql
    return f"{sql} LIMIT {max_rows}"


def run_safely(sql: str) -> list[dict]:
    """Validate, then execute against the real Chinook database."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from modular_common.db import get_connection

    safe_sql = validate_sql(sql)
    conn = get_connection()
    try:
        rows = conn.execute(safe_sql).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()
