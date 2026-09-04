"""Postgres + pgvector — the "relational retrieval" backend: the same
embeddings Qdrant serves, also queryable with plain SQL (joins, filters,
transactions) when a question needs both semantic search *and* structured
lookups in one place. Requires `uv sync --extra postgres`.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from production_common.config import settings

CREATE_TABLE_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    title TEXT,
    text TEXT NOT NULL,
    embedding VECTOR(768)
);

CREATE INDEX IF NOT EXISTS documents_embedding_idx
    ON documents USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
"""


class PostgresStore:
    def __init__(self, dsn: str | None = None) -> None:
        import psycopg

        self.dsn = dsn or settings.postgres_dsn
        self._conn = psycopg.connect(self.dsn, autocommit=True)

    def ensure_schema(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute(CREATE_TABLE_SQL)

    def upsert(self, doc_id: str, title: str, text: str, embedding: list[float]) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                "INSERT INTO documents (doc_id, title, text, embedding) VALUES (%s, %s, %s, %s) "
                "ON CONFLICT (doc_id) DO UPDATE SET title = EXCLUDED.title, text = EXCLUDED.text, "
                "embedding = EXCLUDED.embedding",
                (doc_id, title, text, str(embedding)),
            )

    def search(self, query_vector: list[float], top_k: int = 5) -> list[dict]:
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT doc_id, title, text, 1 - (embedding <=> %s) AS score "
                "FROM documents ORDER BY embedding <=> %s LIMIT %s",
                (str(query_vector), str(query_vector), top_k),
            )
            rows = cur.fetchall()
        return [{"doc_id": r[0], "title": r[1], "text": r[2], "score": float(r[3])} for r in rows]

    def count(self) -> int:
        with self._conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM documents")
            return cur.fetchone()[0]

    def close(self) -> None:
        self._conn.close()
