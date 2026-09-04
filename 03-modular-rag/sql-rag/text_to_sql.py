"""Text-to-SQL — translate a natural-language question into a SQL query
against the Chinook schema, using a local LLM.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modular_common.llm import OllamaLLM

sys.path.insert(0, str(Path(__file__).resolve().parent))
from schema import schema_prompt_block
from sql_guardrails import run_safely, validate_sql

TEXT_TO_SQL_PROMPT = """You write SQLite queries against the schema below.

{schema}

Rules:
- Output ONLY the SQL query, no explanation, no markdown code fences.
- Use only SELECT statements.
- Always include a LIMIT clause.

Question: {question}
SQL:"""


def generate_sql(question: str, llm: OllamaLLM | None = None) -> str:
    llm = llm or OllamaLLM()
    schema = schema_prompt_block()
    raw = llm.complete(TEXT_TO_SQL_PROMPT.format(schema=schema, question=question))
    # Strip markdown fences some models add despite instructions.
    return raw.strip().strip("`").removeprefix("sql").strip()


def answer_from_sql(question: str, llm: OllamaLLM | None = None) -> dict:
    """Generate SQL, validate it, run it, and return everything for
    transparency (the query itself is part of a trustworthy answer).
    """
    llm = llm or OllamaLLM()
    sql = generate_sql(question, llm=llm)
    validate_sql(sql)  # raises UnsafeQueryError before we even try to run it
    rows = run_safely(sql)
    return {"question": question, "sql": sql, "rows": rows}
