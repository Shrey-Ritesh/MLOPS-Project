# Monitoring job and window report (Person 1)

**Implements:** graded topic 1 (Model Monitoring) and §4.2 of the Phase 3 plan.
**Code:** `src/monitoring/run_window.py`, `scheduler.py`, `drift_hook.py`
**Thresholds:** `configs/monitoring/health.yaml`
**Tests:** `tests/test_monitoring_window.py`

## 1. What runs where

The `monitor` service in Docker Compose uses the same image as the app and is capped at 0.5 CPU and 512 MB. It runs `python -m src.monitoring.scheduler`. Every `MONITORING_TICK_SECONDS` (default 60) it:

1. **Joins labels.** It fills in the true answer for each prediction from the learner's next interaction (see the schema doc, §3).
2. **Evaluates due windows.** A window `[start, end)` is evaluated once `now ≥ end + label_lag` (5 min), which gives next-interaction labels time to arrive. Windows line up on multiples of the window length counted from the Unix epoch, so window IDs are the same on every run.
3. **Catches up after downtime.** It re-runs up to 96 missed windows. If the gap is longer, it raises a `monitoring_gap` alert instead of silently skipping. If the window length changes, a short bridge window keeps coverage free of gaps and overlaps.
4. **Backs up nightly.** Once per UTC day after 02:00 it backs up the database to the `monitoring_backups` volume and keeps 7 copies.
5. **Writes a heartbeat.** The container health check fails if no tick has finished in 5 minutes.

To change the window length for the drift experiment, set `MONITORING_WINDOW_MINUTES=2` in the server's `.env`, then run `docker compose up -d monitor`.

## 2. Where results are stored

All results live in `/app/data/state/monitoring.sqlite3` on the `app_state` volume.

| Table | One row per | Used by |
|---|---|---|
| `predictions` | prediction | all |
| `monitoring_windows` | evaluated window (`report` = full JSON below) | dashboard panels 1–4, retraining trigger |
| `alerts` | threshold breach (severity, metric, value, threshold, window) | dashboard panel 5, demo |
| `monitor_state` | scheduler key (`last_window_end`, `last_backup_date`, `heartbeat`) | operations |

There is also a file snapshot per window: `/app/data/state/monitoring/windows/<window_id>/report.json` and `predictions.csv`. The CSV has the features in separate columns and contains no learner IDs, only `learner_key`.

## 3. Window report (`report_version` 1.0.0)

```json
{
  "report_version": "1.0.0",
  "window_id": "w-20261126T0900Z-15m",
  "start": "2026-11-26T09:00:00.000Z", "end": "2026-11-26T09:15:00.000Z",
  "evaluated_at": "2026-11-26T09:20:00.000Z",
  "status": "ok | alert | insufficient_data",
  "n_predictions": 42, "n_labelled": 30, "n_label_pending": 12, "n_label_expired": 0,
  "health":   {"p50": 4.1, "p95": 9.8, "error_rate": 0.0, "timeout_rate": 0.0, "fallback_rate": 0.02,
               "by_source": {"live": 40, "simulated": 2}, "by_model_version": {"2": 42}},
  "validity": {"schema_failures": 0, "out_of_range": 0, "unseen_topics": 1,
               "schema_failure_rate": 0.0, "out_of_range_rate": 0.0, "unseen_topic_rate": 0.0238,
               "unseen_topic_names": {"unknown": 1}},
  "prediction_distribution": {"n_scored": 42, "mean_score": 0.41, "positive_rate": 0.21,
                              "score_bins": [0.0, 0.1, "...", 1.0], "score_counts": [3, 5, "..."]},
  "drift": "not_configured | error | {feature: {psi, ks_stat, ks_p, status}}",
  "prediction_psi": "not_configured | error | float",
  "performance": "not_configured | error | insufficient_data | {f1, roc_auc, brier}",
  "alerts": [{"severity": "warning", "category": "health", "metric": "latency_p95_ms",
              "value": 310.0, "threshold": 250, "message": "..."}],
  "reference_dataset_sha256": "set by Person 2's evaluator",
  "champion_version": "2",
  "thresholds": {"min_predictions_for_rates": 10, "config_version": 1}
}
```

Rules:

- If a window has fewer than `min_predictions_for_rates` predictions (10), `status = insufficient_data`. `health` and `validity` then carry `"status": "insufficient_data"` and raise no alerts, but the raw numbers are still reported. **An empty or tiny window is never reported as healthy.**
- `drift`, `prediction_psi` and `performance` stay `"not_configured"` until Person 2's evaluator exists. The report never claims "no drift" when drift was not measured.
- Evaluating the same window again replaces its report and its alerts, so there are no duplicates.

## 4. Plug point for Person 2

Add `src/monitoring/drift_windows.py` with `evaluate_window(rows: list[dict]) -> dict`. The contract is in the docstring of `src/monitoring/drift_hook.py`. The monitoring job finds it automatically, with no wiring needed. If it crashes, monitoring keeps running and the report shows `"drift": "error"` with the exception type.

## 5. Running it by hand

```bash
docker compose exec monitor python -m src.monitoring.status                 # read-only summary: counts, windows, alerts, heartbeat, backups
docker compose exec monitor python -m src.monitoring.scheduler --once        # one tick now
docker compose exec monitor python -m src.monitoring.run_window \
    --start 2026-11-26T09:00:00Z --end 2026-11-26T09:15:00Z                  # one specific window
docker compose exec monitor python -m src.monitoring.label_joiner           # labels only
docker compose logs monitor --since 1h | grep alert_raised                   # alerts in the logs
```

## 6. Detection-delay report (for the Day 4 experiment)

```bash
docker compose exec monitor python -m src.monitoring.detection_report \
    --shift-at 2026-11-28T10:00:00Z --format md        # or --format json
```

Given the time the shift was injected, it reads the evaluated windows and reports:

- **Control phase:** windows that end at or before the shift. Drift and performance alerts here are **false alarms**. Health and validity alerts are listed separately, because they are usually real incidents.
- **Detection:** windows that start at or after the shift. For each metric it gives the first alerting window and the delay from the shift to when that window was **evaluated**, which is the earliest moment anyone could react. It also gives the share of shifted windows that alerted.
- **Data drift and performance degradation, kept separate:** first input-drift detection, first performance detection, and mean F1 / ROC-AUC / Brier before and after the shift.

A window that straddles the shift is excluded and listed.

On the fixture data (shift at 03:00), data drift is detected after 20 minutes, with 3 false alarms in 11 usable control windows. **No performance alert fires, even though mean F1 falls from 0.68 to 0.59.** Person 2's evaluator should raise performance alerts, or the retraining trigger never sees degradation.

## 7. Known limits (to revisit on Day 5)

- Window snapshot folders grow by about 96 per day at 15 minutes. Their retention (proposed: 30 days) belongs in the governance retention rules (Person 4).
- Alerts go to the `alerts` table, the JSON logs, and (Day 3, Person 4) a dashboard banner. An external channel, email or a GitHub issue, is P2.
- Health thresholds were set from engineering judgement before any live traffic existed. The reason for each is in `health.yaml`.
