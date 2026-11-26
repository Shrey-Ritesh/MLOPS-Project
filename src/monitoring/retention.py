"""Apply the data-retention rules in ``configs/monitoring/retention.yaml``.

Deletes raw ``predictions`` and ``attempt_outcomes`` rows and per-window snapshot folders older
than their retention period. Window reports and alerts (aggregates, no learner keys) are kept.
Rows whose label is still ``pending`` are never deleted, so a long outage cannot drop rows the
label joiner has not seen yet.

Run daily by the monitor after the backup, or by hand:

    python -m src.monitoring.retention [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml

from src.monitoring import db
from src.monitoring.app_logging import log_event

DEFAULT_CONFIG = "configs/monitoring/retention.yaml"
_WINDOW_DIR = re.compile(r"^w-(\d{8}T\d{4})Z-\d+m$")


def load_retention(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    for key in ("raw_predictions_days", "attempt_outcomes_days", "window_snapshot_days"):
        if not isinstance(cfg.get(key), int) or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive number of days")
    return cfg


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def apply_retention(db_path: str | Path, output_dir: str | Path | None, cfg: dict[str, Any] | None = None,
                    now: datetime | None = None, dry_run: bool = False) -> dict[str, int]:
    cfg = cfg or load_retention()
    now = now or datetime.now(timezone.utc)
    pred_cutoff = _iso(now - timedelta(days=cfg["raw_predictions_days"]))
    outcome_cutoff = _iso(now - timedelta(days=cfg["attempt_outcomes_days"]))
    snapshot_cutoff = now - timedelta(days=cfg["window_snapshot_days"])

    with db.session(db_path) as conn:
        db.ensure_schema(conn)
        pred_where = "occurred_at < ? AND label_status != 'pending'"
        out_where = "occurred_at < ?"
        if dry_run:
            predictions = conn.execute(f"SELECT COUNT(*) FROM predictions WHERE {pred_where}", (pred_cutoff,)).fetchone()[0]
            outcomes = conn.execute(f"SELECT COUNT(*) FROM attempt_outcomes WHERE {out_where}", (outcome_cutoff,)).fetchone()[0]
        else:
            predictions = conn.execute(f"DELETE FROM predictions WHERE {pred_where}", (pred_cutoff,)).rowcount
            outcomes = conn.execute(f"DELETE FROM attempt_outcomes WHERE {out_where}", (outcome_cutoff,)).rowcount
        kept_pending = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE occurred_at < ? AND label_status = 'pending'",
            (pred_cutoff,)).fetchone()[0]

    snapshots = 0
    windows_root = Path(output_dir) / "windows" if output_dir else None
    if windows_root and windows_root.is_dir():
        for folder in windows_root.iterdir():
            match = _WINDOW_DIR.match(folder.name)
            if not (folder.is_dir() and match):
                continue  # never touch anything that is not a window snapshot folder
            started = datetime.strptime(match.group(1), "%Y%m%dT%H%M").replace(tzinfo=timezone.utc)
            if started < snapshot_cutoff:
                snapshots += 1
                if not dry_run:
                    shutil.rmtree(folder)

    result = {"predictions_deleted": predictions, "outcomes_deleted": outcomes,
              "snapshots_deleted": snapshots, "old_pending_kept": kept_pending}
    log_event("retention_applied" if not dry_run else "retention_dry_run", service="monitoring",
              count=predictions, reason=" ".join(f"{k}={v}" for k, v in result.items()))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", default=db.db_path_from_env())
    parser.add_argument("--output-dir", default=os.environ.get("MONITORING_OUTPUT_DIR", "data/state/monitoring"))
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true", help="count what would be deleted, delete nothing")
    args = parser.parse_args(argv)
    print(json.dumps(apply_retention(args.db, args.output_dir, load_retention(args.config), dry_run=args.dry_run)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
