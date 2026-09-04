"""SQL backend: the open-source Sakila sample database (DVD rental store)
-- a third distinct SQL domain in this repo, after Chinook (music,
Level 3) and Northwind (trade, Level 5).
"""

from __future__ import annotations

import logging
import sqlite3

from .config import settings

logger = logging.getLogger(__name__)


def ensure_db() -> None:
    if settings.sakila_path.exists():
        return
    import requests

    logger.info("Downloading %s -> %s", settings.sakila_url, settings.sakila_path)
    settings.sakila_path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(settings.sakila_url, timeout=60)
    response.raise_for_status()
    settings.sakila_path.write_bytes(response.content)


def get_connection() -> sqlite3.Connection:
    ensure_db()
    conn = sqlite3.connect(str(settings.sakila_path))
    conn.row_factory = sqlite3.Row
    return conn


CORE_TABLES = ("film", "actor", "customer", "rental", "payment", "store", "category")


def schema_description() -> str:
    conn = get_connection()
    try:
        placeholders = ",".join("?" for _ in CORE_TABLES)
        rows = conn.execute(
            f"SELECT name, sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL "
            f"AND name IN ({placeholders})",
            CORE_TABLES,
        ).fetchall()
    finally:
        conn.close()
    return "\n\n".join(row["sql"] for row in rows)
