"""Describe the Chinook schema for a text-to-SQL prompt.

Chinook has 11 tables; most questions only need a handful of them
(Artist/Album/Track/Customer/Invoice/InvoiceLine). Handing the LLM a
smaller, curated schema — rather than the full `CREATE TABLE` dump —
keeps the prompt focused and the generated SQL more reliable.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modular_common.db import get_connection, schema_description

CORE_TABLES = ["Artist", "Album", "Track", "Customer", "Invoice", "InvoiceLine", "Genre"]


def core_schema() -> str:
    return schema_description(tables=CORE_TABLES)


def sample_rows(table: str, n: int = 2) -> list[dict]:
    """A couple of example rows — helps the LLM infer data formats/casing."""
    conn = get_connection()
    try:
        rows = conn.execute(f"SELECT * FROM {table} LIMIT {n}").fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def schema_prompt_block() -> str:
    """Schema + a couple of sample rows per core table, ready to paste into
    a text-to-SQL prompt.
    """
    parts = [core_schema()]
    for table in ("Artist", "Track"):
        parts.append(f"-- Sample rows from {table}: {sample_rows(table)}")
    return "\n\n".join(parts)
