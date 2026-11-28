"""Detection-delay report for the drift / degradation experiment (graded topic 4).

Reads the evaluated monitoring windows and, given the time the shift was injected, reports:

* control phase (windows that END at or before the shift): how many windows alerted, per alert
  category -- for drift/performance these are false alarms; health alerts are listed separately
  because they are usually real incidents, not drift;
* detection (windows that START at or after the shift): for each alert metric, the first window
  that alerted, the delay from the shift to when that window was EVALUATED (the earliest moment
  a person or the retraining trigger could act), and how many shifted windows alerted;
* data drift vs performance degradation, kept separate: first input-drift detection, first
  performance detection, and mean F1 / ROC-AUC / Brier before vs after the shift.

A window that straddles the shift is excluded from both phases and listed.

    python -m src.monitoring.detection_report --shift-at 2026-11-28T10:00:00Z [--start ...] [--end ...] [--format md]
"""
from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any

from src.monitoring import db
from src.monitoring.run_window import iso, parse_iso

DATA_DRIFT = "drift"
PERFORMANCE = "performance"
SIGNAL_CATEGORIES = (DATA_DRIFT, PERFORMANCE)  # alerts in these categories are false alarms before the shift


def _load_reports(db_path: str | Path, start: datetime | None, end: datetime | None) -> list[dict[str, Any]]:
    query, args = "SELECT report FROM monitoring_windows WHERE 1=1", []
    if start:
        query += " AND window_start >= ?"
        args.append(iso(start))
    if end:
        query += " AND window_end <= ?"
        args.append(iso(end))
    with db.session(db_path) as conn:
        db.ensure_schema(conn)
        return [json.loads(r) for (r,) in conn.execute(query + " ORDER BY window_start", args)]


def _perf(report: dict[str, Any]) -> dict[str, float] | None:
    perf = report.get("performance")
    return perf if isinstance(perf, dict) and "f1" in perf else None


def _mean(values: list[float]) -> float | None:
    return round(statistics.fmean(values), 4) if values else None


def build_report(db_path: str | Path, shift_at: datetime, start: datetime | None = None,
                 end: datetime | None = None) -> dict[str, Any]:
    reports = _load_reports(db_path, start, end)
    control = [r for r in reports if parse_iso(r["end"]) <= shift_at]
    shifted = [r for r in reports if parse_iso(r["start"]) >= shift_at]
    straddling = [r["window_id"] for r in reports if r not in control and r not in shifted]

    # control phase
    by_category: dict[str, list[str]] = {}
    for r in control:
        for category in sorted({a["category"] for a in r["alerts"]}):
            by_category.setdefault(category, []).append(r["window_id"])
    evaluable_control = [r for r in control if r["status"] != "insufficient_data"]
    signal_false_alarms = sorted({w for c in SIGNAL_CATEGORIES for w in by_category.get(c, [])})

    # detection per metric
    detection: dict[str, dict[str, Any]] = {}
    for r in shifted:  # in time order, so setdefault keeps the FIRST alerting window per metric
        for a in r["alerts"]:
            detection.setdefault(a["metric"], {
                "category": a["category"], "first_window": r["window_id"], "first_window_end": r["end"],
                "first_evaluated_at": r["evaluated_at"],
                "delay_minutes": round((parse_iso(r["evaluated_at"]) - shift_at).total_seconds() / 60, 1),
            })
    for metric, d in detection.items():
        d["windows_alerting"] = sum(1 for r in shifted if any(a["metric"] == metric for a in r["alerts"]))
        d["share_of_shifted_windows"] = round(d["windows_alerting"] / len(shifted), 3) if shifted else None
        d["control_windows_alerting"] = sum(1 for r in control if any(a["metric"] == metric for a in r["alerts"]))

    def first(category: str) -> dict[str, Any] | None:
        hits = [dict(metric=m, **d) for m, d in detection.items() if d["category"] == category]
        return min(hits, key=lambda h: h["delay_minutes"]) if hits else None

    perf_before = [p for p in map(_perf, control) if p]
    perf_after = [p for p in map(_perf, shifted) if p]

    def perf_summary(rows: list[dict[str, float]]) -> dict[str, Any]:
        return {"windows_with_performance": len(rows),
                **{f"mean_{k}": _mean([p[k] for p in rows if k in p]) for k in ("f1", "roc_auc", "brier")}}

    return {
        "shift_at": iso(shift_at),
        "windows": {"total": len(reports), "control": len(control), "shifted": len(shifted),
                    "straddling_excluded": straddling,
                    "insufficient_data": {"control": sum(r["status"] == "insufficient_data" for r in control),
                                          "shifted": sum(r["status"] == "insufficient_data" for r in shifted)}},
        "control_phase": {
            "windows_alerting_by_category": {c: len(w) for c, w in by_category.items()},
            "alerting_windows_by_category": by_category,
            "false_alarm_windows_drift_or_performance": signal_false_alarms,
            "false_alarm_rate_drift_or_performance": (round(len(signal_false_alarms) / len(evaluable_control), 3)
                                                       if evaluable_control else None),
            "note": "health/validity alerts before the shift are listed but are not counted as drift false alarms",
        },
        "detection_by_metric": dict(sorted(detection.items(), key=lambda kv: kv[1]["delay_minutes"])),
        "data_drift_detection": first(DATA_DRIFT),
        "performance_detection": first(PERFORMANCE),
        "performance_trend": {"before_shift": perf_summary(perf_before), "after_shift": perf_summary(perf_after)},
        "contract_ids": sorted({r.get("contract_id") for r in reports if r.get("contract_id")}),
    }


def to_markdown(rep: dict[str, Any]) -> str:
    w, c = rep["windows"], rep["control_phase"]
    lines = [
        f"## Detection report (shift at {rep['shift_at']})", "",
        f"Windows: {w['total']} total, {w['control']} control, {w['shifted']} shifted"
        f" ({w['insufficient_data']['control']} / {w['insufficient_data']['shifted']} insufficient_data);"
        f" straddling excluded: {', '.join(w['straddling_excluded']) or 'none'}.", "",
        "### Control phase (before the shift)", "",
        f"Drift/performance false alarms: **{len(c['false_alarm_windows_drift_or_performance'])}**"
        f" windows (rate {c['false_alarm_rate_drift_or_performance']}).", "",
        "| Category | Windows alerting |", "|---|---|",
        *[f"| {k} | {v} |" for k, v in c["windows_alerting_by_category"].items()], "",
        "### Detection after the shift", "",
        "| Metric | Category | First window | Evaluated at | Delay (min) | Shifted windows alerting | Control windows alerting |",
        "|---|---|---|---|---|---|---|",
        *[f"| {m} | {d['category']} | {d['first_window']} | {d['first_evaluated_at']} | {d['delay_minutes']} |"
          f" {d['windows_alerting']} ({d['share_of_shifted_windows']}) | {d['control_windows_alerting']} |"
          for m, d in rep["detection_by_metric"].items()], "",
        "### Data drift vs performance degradation", "",
    ]
    for label, key in (("Data drift", "data_drift_detection"), ("Performance", "performance_detection")):
        hit = rep[key]
        lines.append(f"- {label}: " + (f"first detected by `{hit['metric']}` after {hit['delay_minutes']} min"
                                         if hit else "**not detected** by any alert"))
    before, after = rep["performance_trend"]["before_shift"], rep["performance_trend"]["after_shift"]
    lines += ["", "| | Windows | Mean F1 | Mean ROC-AUC | Mean Brier |", "|---|---|---|---|---|",
              f"| Before shift | {before['windows_with_performance']} | {before['mean_f1']} | {before['mean_roc_auc']} | {before['mean_brier']} |",
              f"| After shift | {after['windows_with_performance']} | {after['mean_f1']} | {after['mean_roc_auc']} | {after['mean_brier']} |",
              ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", default=db.db_path_from_env())
    parser.add_argument("--shift-at", required=True, help="UTC time the shift was injected, e.g. 2026-11-28T10:00:00Z")
    parser.add_argument("--start", help="ignore windows starting before this")
    parser.add_argument("--end", help="ignore windows ending after this")
    parser.add_argument("--format", choices=("json", "md"), default="json")
    args = parser.parse_args(argv)
    rep = build_report(args.db, parse_iso(args.shift_at), parse_iso(args.start) if args.start else None,
                       parse_iso(args.end) if args.end else None)
    print(json.dumps(rep, indent=2) if args.format == "json" else to_markdown(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
