"""SQL backend: the open-source Northwind sample database (business
orders/products/suppliers) — a different domain from Level 3's Chinook
(music store), so `tools/sql_tool.py` is exercising genuinely different
data, not the same DB again.
"""

from __future__ import annotations

import logging
import sqlite3

from .config import settings

logger = logging.getLogger(__name__)


def ensure_db() -> None:
    if settings.northwind_path.exists():
        return
    import requests

    logger.info("Downloading %s -> %s", settings.northwind_url, settings.northwind_path)
    settings.northwind_path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(settings.northwind_url, timeout=60)
    response.raise_for_status()
    settings.northwind_path.write_bytes(response.content)


def get_connection() -> sqlite3.Connection:
    ensure_db()
    conn = sqlite3.connect(str(settings.northwind_path))
    conn.row_factory = sqlite3.Row
    return conn


CORE_TABLES = ["Products", "Categories", "Suppliers", "Orders", "\"Order Details\"", "Customers"]


def schema_description() -> str:
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL "
            "AND name IN ('Products', 'Categories', 'Suppliers', 'Orders', 'Order Details', 'Customers')"
        ).fetchall()
    finally:
        conn.close()
    return "\n\n".join(row["sql"] for row in rows)
