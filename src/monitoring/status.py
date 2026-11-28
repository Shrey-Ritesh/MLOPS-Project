"""One-screen, read-only summary of logging and monitoring (for the demo and for operations).

Shows: prediction and outcome counts by source and label status, the latest windows, the
latest alerts, the heartbeat age and the newest backups. Opens the database read-only, so it
is safe to run against the live server at any time.

    python -m src.monitoring.status [--windows 8] [--alerts 8] [--format md|json]
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.monitoring import db
from src.monitoring.backup import PREFIX, SUFFIX
from src.monitoring.run_window import parse_iso
from src.monitoring.scheduler import HEARTBEAT_FILE


def _connect_ro(db_path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{Path(db_path).resolve()}?mode=ro", uri=True, timeout=5)
    conn.row_factory = sqlite3.Row
    return conn


def collect(db_path: str | Path, output_dir: str | Path | None = None, backup_dir: str | Path | None = None,
            n_windows: int = 8, n_alerts: int = 8, now: datetime | None = None) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    if not Path(db_path).is_file():
        return {"database": str(db_path), "error": "database not found (nothing logged yet?)"}
    conn = _connect_ro(db_path)
    try:
        def q(sql: str, args: tuple = ()) -> list[dict[str, Any]]:
            return [dict(r) for r in conn.execute(sql, args)]

        out: dict[str, Any] = {
            "database": str(db_path),
            "predictions": q("SELECT source, label_status, COUNT(*) AS n, MIN(occurred_at) AS first, "
                             "MAX(occurred_at) AS last FROM predictions GROUP BY source, label_status "
                             "ORDER BY source, label_status"),
            "predictions_by_model": q("SELECT contract_id, model_name, model_version, COUNT(*) AS n, "
                                      "SUM(fallback) AS fallbacks FROM predictions "
                                      "GROUP BY contract_id, model_name, model_version ORDER BY n DESC"),
            "outcomes": q("SELECT source, outcome, COUNT(*) AS n FROM attempt_outcomes "
                          "GROUP BY source, outcome ORDER BY source, outcome"),
            "windows": q("SELECT window_id, status, n_predictions, n_labelled, evaluated_at FROM monitoring_windows "
                         "ORDER BY window_end DESC LIMIT ?", (n_windows,)),
            "window_status_counts": q("SELECT status, COUNT(*) AS n FROM monitoring_windows GROUP BY status"),
            "alerts": q("SELECT created_at, window_id, severity, category, metric, value, threshold FROM alerts "
                        "ORDER BY created_at DESC LIMIT ?", (n_alerts,)),
            "state": {r["key"]: r["value"] for r in q("SELECT key, value FROM monitor_state")},
        }
    finally:
        conn.close()

    beat_age = None
    if output_dir:
        try:
            beat = parse_iso((Path(output_dir) / HEARTBEAT_FILE).read_text(encoding="utf-8").strip())
            beat_age = round((now - beat).total_seconds())
        except (OSError, ValueError):
            pass
    out["heartbeat_age_seconds"] = beat_age
    backups = sorted(Path(backup_dir).glob(f"{PREFIX}*{SUFFIX}")) if backup_dir and Path(backup_dir).is_dir() else []
    out["backups"] = [{"file": b.name, "bytes": b.stat().st_size} for b in backups[-3:]]
    out["backups_total"] = len(backups)
    return out


def _table(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["(none)"]
    cols = list(rows[0])
    return ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols),
            *["| " + " | ".join("" if r[c] is None else str(r[c]) for c in cols) + " |" for r in rows]]


def to_markdown(s: dict[str, Any]) -> str:
    if "error" in s:
        return f"{s['database']}: {s['error']}"
    beat = s["heartbeat_age_seconds"]
    lines = [f"## Monitoring status ({s['database']})", "",
             f"Heartbeat: {'missing' if beat is None else f'{beat} s ago'}"
             f" | last window end: {s['state'].get('last_window_end', '-')}"
             f" | last backup: {s['state'].get('last_backup_date', '-')} ({s['backups_total']} kept)", ""]
    for title, key in (("Predictions", "predictions"), ("By model", "predictions_by_model"),
                       ("Outcomes", "outcomes"), ("Window status", "window_status_counts"),
                       ("Latest windows", "windows"), ("Latest alerts", "alerts"), ("Newest backups", "backups")):
        lines += [f"### {title}", "", *_table(s[key]), ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", default=db.db_path_from_env())
    parser.add_argument("--output-dir", default=os.environ.get("MONITORING_OUTPUT_DIR", "data/state/monitoring"))
    parser.add_argument("--backup-dir", default=os.environ.get("MONITORING_BACKUP_DIR", "data/backups"))
    parser.add_argument("--windows", type=int, default=8)
    parser.add_argument("--alerts", type=int, default=8)
    parser.add_argument("--format", choices=("md", "json"), default="md")
    args = parser.parse_args(argv)
    s = collect(args.db, args.output_dir, args.backup_dir, args.windows, args.alerts)
    print(json.dumps(s, indent=2) if args.format == "json" else to_markdown(s))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
