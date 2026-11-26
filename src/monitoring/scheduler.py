"""Monitoring scheduler: the long-running process in the ``monitor`` compose service.

Each tick (every ``tick_seconds``):
  1. join delayed labels onto the prediction log;
  2. evaluate every complete window that has not been evaluated yet (catching up after
     downtime, at most ``max_catch_up_windows``; a larger gap raises a ``monitoring_gap`` alert);
  3. once per UTC day after ``backup_hour_utc``, back up the monitoring database;
  4. write a heartbeat (``monitor_state`` row + file) used by the container health check.

A window [start, end) is evaluated once ``now >= end + label_lag``. Windows are aligned to
multiples of the window length since the Unix epoch, so every run agrees on window IDs.

    python -m src.monitoring.scheduler            # forever (container)
    python -m src.monitoring.scheduler --once     # one tick, then exit (manual / tests)

Environment: MONITORING_DB_PATH, MONITORING_OUTPUT_DIR, MONITORING_BACKUP_DIR,
MONITORING_WINDOW_MINUTES, MONITORING_TICK_SECONDS, MONITORING_BACKUP_HOUR_UTC.
"""
from __future__ import annotations

import argparse
import logging
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from src.monitoring import db
from src.monitoring.app_logging import log_event
from src.monitoring.backup import backup
from src.monitoring.label_joiner import join_labels
from src.monitoring.retention import apply_retention, load_retention
from src.monitoring.run_window import DEFAULT_CONFIG, evaluate_window, iso, load_config, parse_iso

HEARTBEAT_FILE = "monitor.heartbeat"
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


@dataclass(frozen=True)
class Settings:
    db_path: str
    output_dir: str
    backup_dir: str
    window: timedelta
    label_lag: timedelta
    max_catch_up: int
    tick_seconds: int
    backup_hour_utc: int
    config: dict[str, Any]
    retention: dict[str, Any] | None = None  # configs/monitoring/retention.yaml; loaded when None

    @classmethod
    def from_env(cls, config_path: str = DEFAULT_CONFIG) -> "Settings":
        cfg = load_config(config_path)
        win = cfg["window"]
        minutes = int(os.environ.get("MONITORING_WINDOW_MINUTES", win["minutes"]))
        if minutes < 1:
            raise ValueError("MONITORING_WINDOW_MINUTES must be at least 1")
        return cls(
            db_path=db.db_path_from_env(),
            output_dir=os.environ.get("MONITORING_OUTPUT_DIR", "data/state/monitoring"),
            backup_dir=os.environ.get("MONITORING_BACKUP_DIR", "data/backups"),
            window=timedelta(minutes=minutes),
            label_lag=timedelta(minutes=int(win["label_lag_minutes"])),
            max_catch_up=int(win["max_catch_up_windows"]),
            tick_seconds=int(os.environ.get("MONITORING_TICK_SECONDS", 60)),
            backup_hour_utc=int(os.environ.get("MONITORING_BACKUP_HOUR_UTC", 2)),
            config=cfg,
            retention=load_retention(),
        )


def floor_to_window(t: datetime, window: timedelta) -> datetime:
    elapsed = (t - EPOCH) // window
    return EPOCH + elapsed * window


def _get_state(conn, key: str) -> str | None:
    row = conn.execute("SELECT value FROM monitor_state WHERE key = ?", (key,)).fetchone()
    return row[0] if row else None


def _set_state(conn, key: str, value: str, now: datetime) -> None:
    conn.execute(
        "INSERT INTO monitor_state (key, value, updated_at) VALUES (?, ?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
        (key, value, iso(now)),
    )


def due_windows(last_end: datetime | None, now: datetime, s: Settings) -> tuple[list[tuple[datetime, datetime]], int]:
    """Return (windows to evaluate, number of windows skipped because the gap was too long)."""
    latest_end = floor_to_window(now - s.label_lag, s.window)
    windows: list[tuple[datetime, datetime]] = []
    if last_end is None:
        t = latest_end - s.window  # first run: just the most recent complete window
    else:
        t = floor_to_window(last_end, s.window)
        if t < last_end:
            # The window length changed (e.g. for the experiment) and the last end is not on the new
            # grid: a short bridge window up to the next boundary keeps coverage gap- and overlap-free.
            t += s.window
            if t <= latest_end:
                windows.append((last_end, t))
    while t + s.window <= latest_end:
        windows.append((t, t + s.window))
        t += s.window
    skipped = max(0, len(windows) - s.max_catch_up)
    return windows[skipped:], skipped


def tick(s: Settings, now: datetime | None = None) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    summary: dict[str, Any] = {"windows": [], "skipped_windows": 0, "backup": None}

    with db.session(s.db_path) as conn:
        db.ensure_schema(conn)
    summary["labels"] = join_labels(s.db_path, now=now).__dict__

    with db.session(s.db_path) as conn:
        last = _get_state(conn, "last_window_end")
    windows, skipped = due_windows(parse_iso(last) if last else None, now, s)

    if skipped:
        summary["skipped_windows"] = skipped
        _record_gap_alert(s, skipped, now)

    for start, end in windows:
        report = evaluate_window(s.db_path, start, end, config=s.config, output_dir=s.output_dir, now=now)
        summary["windows"].append((report["window_id"], report["status"]))
        with db.session(s.db_path) as conn:
            _set_state(conn, "last_window_end", iso(end), now)

    summary["backup"] = _maybe_backup(s, now)
    summary["retention"] = _maybe_retention(s, now)

    with db.session(s.db_path) as conn:
        _set_state(conn, "heartbeat", iso(now), now)
    Path(s.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(s.output_dir) / HEARTBEAT_FILE).write_text(iso(now), encoding="utf-8")
    return summary


def _record_gap_alert(s: Settings, skipped: int, now: datetime) -> None:
    message = (f"monitoring was down too long: {skipped} window(s) of {int(s.window.total_seconds() // 60)} min "
               f"were not evaluated (catch-up limit {s.max_catch_up})")
    with db.session(s.db_path) as conn:
        conn.execute(
            """INSERT INTO alerts (alert_id, created_at, window_id, severity, category, metric, value, threshold, message)
               VALUES (?, ?, NULL, 'warning', 'monitoring', 'monitoring_gap', ?, ?, ?)""",
            (str(uuid.uuid4()), iso(now), float(skipped), float(s.max_catch_up), message),
        )
    log_event("alert_raised", level=logging.WARNING, service="monitoring", severity="warning",
              category="monitoring", metric="monitoring_gap", value=skipped, threshold=s.max_catch_up)


def _maybe_backup(s: Settings, now: datetime) -> str | None:
    today = now.strftime("%Y-%m-%d")
    if now.hour < s.backup_hour_utc:
        return None
    with db.session(s.db_path) as conn:
        if _get_state(conn, "last_backup_date") == today:
            return None
    try:
        keep = int((s.retention or load_retention()).get("backups_keep", 7))
        path = backup(s.db_path, s.backup_dir, keep=keep, now=now)
    except Exception as exc:
        log_event("backup_failed", level=logging.ERROR, service="monitoring", error_type=type(exc).__name__)
        return None
    with db.session(s.db_path) as conn:
        _set_state(conn, "last_backup_date", today, now)
    return path.name


def _maybe_retention(s: Settings, now: datetime) -> dict[str, int] | None:
    """Once a day, and only after today's backup succeeded, so nothing is deleted un-backed-up."""
    today = now.strftime("%Y-%m-%d")
    with db.session(s.db_path) as conn:
        if _get_state(conn, "last_backup_date") != today or _get_state(conn, "last_retention_date") == today:
            return None
    try:
        result = apply_retention(s.db_path, s.output_dir, s.retention or load_retention(), now=now)
    except Exception as exc:
        log_event("retention_failed", level=logging.ERROR, service="monitoring", error_type=type(exc).__name__)
        return None
    with db.session(s.db_path) as conn:
        _set_state(conn, "last_retention_date", today, now)
    return result


def heartbeat_is_fresh(output_dir: str, max_age_seconds: int) -> bool:
    """Used by the container health check."""
    try:
        beat = parse_iso((Path(output_dir) / HEARTBEAT_FILE).read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return False
    return (datetime.now(timezone.utc) - beat).total_seconds() <= max_age_seconds


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--once", action="store_true", help="run one tick and exit")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--check-heartbeat", type=int, metavar="SECONDS",
                        help="exit 0 if the heartbeat is newer than SECONDS (health check)")
    args = parser.parse_args(argv)

    if args.check_heartbeat is not None:
        out = os.environ.get("MONITORING_OUTPUT_DIR", "data/state/monitoring")
        return 0 if heartbeat_is_fresh(out, args.check_heartbeat) else 1

    s = Settings.from_env(args.config)
    log_event("monitor_started", service="monitoring",
              reason=f"window={int(s.window.total_seconds() // 60)}m tick={s.tick_seconds}s")
    while True:
        ok = True
        try:
            summary = tick(s)
            log_event("monitor_tick", service="monitoring", count=len(summary["windows"]),
                      reason=f"skipped={summary['skipped_windows']} backup={summary['backup']}")
        except Exception as exc:  # keep running; the next tick catches up the missed windows
            ok = False
            log_event("monitor_tick_failed", level=logging.ERROR, service="monitoring",
                      error_type=type(exc).__name__)
        if args.once:
            return 0 if ok else 1
        time.sleep(s.tick_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
