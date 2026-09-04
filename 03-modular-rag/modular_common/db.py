"""SQL backend: the open-source (MIT-licensed) Chinook sample database.

Chinook models a digital media store (artists, albums, tracks, customers,
invoices) — deliberately a completely different domain from the PDF, the
way a real "enterprise assistant" pulls from genuinely different systems
depending on the question (see this level's mini project).
"""

from __future__ import annotations

import logging
import sqlite3

from .config import settings

logger = logging.getLogger(__name__)


def ensure_db() -> None:
    if settings.chinook_path.exists():
        return
    import requests

    logger.info("Downloading %s -> %s", settings.chinook_url, settings.chinook_path)
    settings.chinook_path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(settings.chinook_url, timeout=30)
    response.raise_for_status()
    settings.chinook_path.write_bytes(response.content)


def get_connection() -> sqlite3.Connection:
    ensure_db()
    conn = sqlite3.connect(str(settings.chinook_path))
    conn.row_factory = sqlite3.Row
    return conn


def schema_description(tables: list[str] | None = None) -> str:
    """A compact `CREATE TABLE` listing suitable for a text-to-SQL prompt."""
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL"
        ).fetchall()
    finally:
        conn.close()

    if tables:
        rows = [r for r in rows if r["name"] in tables]
    return "\n\n".join(row["sql"] for row in rows)
