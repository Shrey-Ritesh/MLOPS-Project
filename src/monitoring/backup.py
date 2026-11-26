"""Consistent online backup of the monitoring database, with rotation.

Uses SQLite's backup API, so it is safe while the app is writing (WAL mode). Each backup is
verified with ``PRAGMA integrity_check`` before older copies are rotated out.

    python -m src.monitoring.backup --dest /app/data/backups [--keep 7]
    python -m src.monitoring.backup --restore /app/data/backups/monitoring-20261126T020000Z.sqlite3 --to restored.sqlite3

Scheduled nightly by the monitoring sidecar (Day 3).
"""
from __future__ import annotations

import argparse
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from src.monitoring import db
from src.monitoring.app_logging import log_event

PREFIX = "monitoring-"
SUFFIX = ".sqlite3"
DEFAULT_KEEP = 7


class BackupError(RuntimeError):
    pass


def _integrity_ok(path: Path) -> bool:
    conn = sqlite3.connect(path)
    try:
        return conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    finally:
        conn.close()


def backup(db_path: str | Path, dest_dir: str | Path, keep: int = DEFAULT_KEEP,
           now: datetime | None = None) -> Path:
    if keep < 1:
        raise ValueError("keep must be at least 1")
    source = Path(db_path)
    if not source.is_file():
        raise BackupError(f"database not found: {source}")
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    target = dest / f"{PREFIX}{stamp}{SUFFIX}"
    partial = target.with_suffix(".partial")

    src = db.connect(source)
    out = sqlite3.connect(partial)
    try:
        src.backup(out)
    finally:
        out.close()
        src.close()
    if not _integrity_ok(partial):
        partial.unlink(missing_ok=True)
        raise BackupError("integrity check failed on the new backup; older backups kept")
    partial.replace(target)

    backups = sorted(dest.glob(f"{PREFIX}*{SUFFIX}"))
    for old in backups[:-keep]:
        old.unlink()
    log_event("backup_completed", service="monitoring", path=target.name, count=min(len(backups), keep))
    return target


def restore(backup_file: str | Path, to: str | Path) -> Path:
    """Copy a backup to ``to`` (never over a live database by accident: ``to`` must not exist)."""
    source, target = Path(backup_file), Path(to)
    if target.exists():
        raise BackupError(f"refusing to overwrite existing file: {target}")
    if not _integrity_ok(source):
        raise BackupError(f"backup fails integrity check: {source}")
    src = sqlite3.connect(source)
    out = sqlite3.connect(target)
    try:
        src.backup(out)
    finally:
        out.close()
        src.close()
    log_event("backup_restored", service="monitoring", path=target.name)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", default=db.db_path_from_env())
    parser.add_argument("--dest", default="data/backups")
    parser.add_argument("--keep", type=int, default=DEFAULT_KEEP)
    parser.add_argument("--restore", help="backup file to restore")
    parser.add_argument("--to", help="restore target path (must not exist)")
    args = parser.parse_args(argv)
    if args.restore:
        if not args.to:
            parser.error("--restore needs --to")
        print(restore(args.restore, args.to))
    else:
        print(backup(args.db, args.dest, args.keep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
