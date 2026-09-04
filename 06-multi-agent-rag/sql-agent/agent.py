"""SQL Agent — text-to-SQL against the open Sakila database. Same
guardrail discipline as Levels 3 and 5's SQL tools (validate before
executing, SELECT-only, no multi-statement, forced LIMIT), reimplemented
here to keep this level self-contained.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from multiagent_common.agent_result import AgentResult
from multiagent_common.db import get_connection, schema_description
from multiagent_common.llm import OllamaLLM

FORBIDDEN_KEYWORDS = ("DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "ATTACH", "PRAGMA", "CREATE")
MAX_ROWS = 50

TEXT_TO_SQL_PROMPT = """You write SQLite queries against the schema below.

{schema}

Rules:
- Output ONLY the SQL query, no explanation, no markdown fences.
- Use only SELECT statements.
- Always include a LIMIT clause.

Task: {task}
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


class SqlAgent:
    name = "sql-agent"

    def __init__(self, llm: OllamaLLM | None = None) -> None:
        self.llm = llm or OllamaLLM()

    def run(self, task: str) -> AgentResult:
        schema = schema_description()
        raw_sql = self.llm.complete(TEXT_TO_SQL_PROMPT.format(schema=schema, task=task))
        sql = raw_sql.strip().strip("`").removeprefix("sql").strip()

        try:
            safe_sql = validate_sql(sql)
            conn = get_connection()
            try:
                rows = [dict(row) for row in conn.execute(safe_sql).fetchall()]
            finally:
                conn.close()
        except Exception as exc:  # noqa: BLE001 - unsafe/invalid SQL shouldn't crash the workflow
            return AgentResult(self.name, task, f"Could not query the database: {exc}", success=False, error=str(exc))

        return AgentResult(self.name, task, f"SQL: {safe_sql}\nRows: {rows}", evidence=[str(rows)])
