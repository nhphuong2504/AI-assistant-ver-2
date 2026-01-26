import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Tuple

DB_PATH = Path("data/retail.sqlite")

# Block anything that's not read-only. This is conservative on purpose.
FORBIDDEN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|TRUNCATE|VACUUM|ATTACH|DETACH|PRAGMA)\b",
    re.IGNORECASE,
)

MULTI_STMT = re.compile(r";\s*\S", re.DOTALL)  # semicolon with more text after it


def get_conn() -> sqlite3.Connection:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found at {DB_PATH.resolve()}")
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_select_only(sql: str) -> str:
    s = sql.strip()

    # allow WITH ... SELECT ... (CTE) or SELECT ...
    if not (s.lower().startswith("select") or s.lower().startswith("with")):
        raise ValueError("Only SELECT queries are allowed.")

    if FORBIDDEN.search(s):
        raise ValueError("Forbidden keyword detected. Only read-only SELECT queries are allowed.")

    if MULTI_STMT.search(s):
        raise ValueError("Multiple statements are not allowed.")

    return s


def ensure_limit(sql: str, limit: int) -> str:
    # If LIMIT already exists, keep it but cap later by slicing results.
    if re.search(r"\blimit\b", sql, re.IGNORECASE):
        return sql
    return f"{sql.rstrip()}\nLIMIT {int(limit)}"


def run_query(sql: str, limit: int = 1000, max_rows: int = 10000) -> Tuple[List[Dict[str, Any]], List[str]]:
    sql = ensure_select_only(sql)
    sql = ensure_limit(sql, limit)

    with get_conn() as conn:
        cur = conn.execute(sql)
        rows = cur.fetchmany(max_rows)  # hard cap
        cols = [d[0] for d in cur.description] if cur.description else []
        out = [dict(r) for r in rows]
        return out, cols


def get_schema() -> Dict[str, Any]:
    with get_conn() as conn:
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()

        schema: Dict[str, Any] = {}
        for t in tables:
            table = t["name"]
            cols = conn.execute(f"PRAGMA table_info({table})").fetchall()
            schema[table] = [
                {
                    "cid": c["cid"],
                    "name": c["name"],
                    "type": c["type"],
                    "notnull": c["notnull"],
                    "pk": c["pk"],
                }
                for c in cols
            ]
        return schema

def run_query_internal(sql: str, max_rows: int = 2_000_000):
    s = sql.strip()
    if MULTI_STMT.search(s):
        raise ValueError("Multiple statements are not allowed.")
    if FORBIDDEN.search(s):
        raise ValueError("Forbidden keyword detected.")
    with get_conn() as conn:
        cur = conn.execute(s)
        rows = cur.fetchmany(max_rows)
        cols = [d[0] for d in cur.description] if cur.description else []
        out = [dict(r) for r in rows]
        return out, cols
