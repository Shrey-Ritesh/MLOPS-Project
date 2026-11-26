"""SQLite connection handling for the monitoring database (WAL, busy timeout, schema bootstrap)."""
from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from src.monitoring.schema import DDL

DEFAULT_DB_PATH = "data/state/monitoring.sqlite3"
BUSY_TIMEOUT_MS = 5000


def db_path_from_env() -> str:
    return os.environ.get("MONITORING_DB_PATH", DEFAULT_DB_PATH)


def connect(db_path: str | Path) -> sqlite3.Connection:
    """Open a connection with WAL so readers (monitoring, dashboard) never block the app's writes."""
    path = str(db_path)
    if path != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=BUSY_TIMEOUT_MS / 1000)
    conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(DDL)


@contextmanager
def session(db_path: str | Path) -> Iterator[sqlite3.Connection]:
    """Open, commit on success (roll back on error), and always close."""
    conn = connect(db_path)
    try:
        with conn:
            yield conn
    finally:
        conn.close()
