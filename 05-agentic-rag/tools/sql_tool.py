"""`sql_query(question)` — text-to-SQL against the open Northwind database.

Same guardrail discipline as Level 3's `sql-rag/sql_guardrails.py`
(validate before executing, SELECT-only, no multi-statement, forced
LIMIT), reimplemented here to keep this level self-contained.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agentic_common.db import get_connection, schema_description
from agentic_common.llm import OllamaLLM

FORBIDDEN_KEYWORDS = ("DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "ATTACH", "PRAGMA", "CREATE")
MAX_ROWS = 50

TEXT_TO_SQL_PROMPT = """You write SQLite queries against the schema below.

{schema}

Rules:
- Output ONLY the SQL query, no explanation, no markdown fences.
- Use only SELECT statements.
- Always include a LIMIT clause.
- Table/column names with spaces must be double-quoted, e.g. "Order Details".

Question: {question}
SQL:"""


class UnsafeQueryError(ValueError):
    pass


def validate_sql(sql: str) -> str:
    cleaned = sql.strip().rstrip(";")
    if ";" in cleaned:
        raise UnsafeQueryError("Multi-statement SQL is not allowed.")
    if not re.match(r"^\s*SELECT\b", cleaned, re.IGNORECASE):
        raise UnsafeQueryError("Only SELECT statements are allowed.")
    for keyword in FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{keyword}\b", cleaned, re.IGNORECASE):
            raise UnsafeQueryError(f"Forbidden keyword detected: {keyword}")
    if not re.search(r"\bLIMIT\s+\d+\b", cleaned, re.IGNORECASE):
        cleaned = f"{cleaned} LIMIT {MAX_ROWS}"
    return cleaned


class SqlTool:
    name = "sql_query"
    description = "Answer a question by querying the Northwind business database (products, orders, customers, suppliers)."

    def __init__(self, llm: OllamaLLM | None = None) -> None:
        self.llm = llm or OllamaLLM()

    def __call__(self, question: str) -> dict:
        schema = schema_description()
        raw_sql = self.llm.complete(TEXT_TO_SQL_PROMPT.format(schema=schema, question=question))
        sql = raw_sql.strip().strip("`").removeprefix("sql").strip()
        safe_sql = validate_sql(sql)  # raises before touching the database

        conn = get_connection()
        try:
            rows = [dict(row) for row in conn.execute(safe_sql).fetchall()]
        finally:
            conn.close()
        return {"sql": safe_sql, "rows": rows}
