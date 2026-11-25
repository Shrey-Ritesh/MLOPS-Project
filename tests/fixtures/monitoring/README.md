# Shared monitoring fixtures (SYNTHETIC)

Built by `scripts/make_monitoring_fixtures.py` from the Track B synthetic learners. Nothing here describes real students. Use these fixtures to build and test the dashboard (Person 4), the drift code (Person 2) and the comparison and governance views (Persons 3 and 4) before live traffic exists.

| File | What it is |
|---|---|
| `monitoring_sample.sqlite3` | A full monitoring database: `predictions` (about 2,400 rows, all `source = simulated`), `monitoring_windows` (24 reports), `alerts`. It has the same tables and the same report shape as production. |
| `timeline.json` | A one-line summary per window: status, counts, F1, alert metrics. |
| `window_reports/*.json` | Seven representative reports: `ok_baseline`, `outage_fallback`, `latency_spike`, `insufficient_data`, `first_shifted`, `unseen_topic`, `late_shifted`. |
| `registry_snapshot.json` | Made-up registry versions, alias history, approvals and a champion vs challenger comparison, for dashboard panels 7 and 8. Every value is fictional. |

## Timeline (2026-11-20, UTC, 15-minute windows)

| Time | Window | What happens |
|---|---|---|
| 00:00 to 03:00 | 0–11 | Baseline traffic (`learner_interactions_v1`), the control phase |
| 01:00 | 4 | Model outage: about 70% of calls fall back, giving a **critical `fallback_rate`** alert |
| 01:45 | 7 | Latency spike, giving a `latency_p95_ms` alert |
| 02:15 | 9 | Quiet window with 4 predictions: **`insufficient_data`** |
| **03:00** | **12** | **Shift point.** Shifted traffic (`learner_interactions_shifted_v1`) plus an injected input shift (response time × 0.45). Every window from here on has a `psi.response_time_seconds` alert. |
| 04:30 | 18 | Burst of an unseen topic tag: `unseen_topic_rate` alert |

Control-phase false drift alarms: 3 of 11 baseline windows (windows 0, 4 and 5). This is the kind of number the Day 4 experiment report needs.

## What the fixture shows about the Day 4 experiment

1. **The documented Track B shift barely changes the inputs.** Over the full datasets the largest input PSI is 0.056 (`hint_count`, from `reports/track_b/drift_report.json`), far below the 0.2 alert level. It is mostly **concept drift**: the link between features and target changes. On its own it shows up only in label-based performance, never in input drift alerts. That is why the plan asks for at least one extra input shift. This fixture adds one (faster responses).
2. **Sampling matters for the traffic generator.**
   - If every learner is replayed from interaction 0, `prior_interaction_count` appears to drift.
   - If a few learners each play long sessions, the rows in a window are strongly correlated, so per-window PSI behaves as if the sample were small and false alarms rise.

   The fixture uses one short session (4–8 interactions) per learner, started around a randomly chosen interaction.
3. **`prior_interaction_count` is noisy per window**, even with that sampling, because a session's counts are consecutive. The fixture evaluator reports its PSI but does not alert on it. Person 2 decides the real rule in `configs/monitoring/thresholds.yaml`.

## How it was produced (and the limits)

- **Model:** a stand-in logistic regression trained inside the generator on baseline rows (`model_version = "fixture-1"`). It is not a registered model. It only exists so the scores react to the shift.
- **Drift and performance:** computed by a *fixture* evaluator inside the generator (PSI and KS from `src/track_b/drift.py`; F1, ROC-AUC and Brier on joined labels), standing in for Person 2's `src/monitoring/drift_windows.py`.
- **Labels:** joined by the real label joiner, 1 hour after the last window. The last interaction of each session stays `pending`.
- **Pseudonym key:** a fixed, non-secret fixture key.

To regenerate (needs `dvc pull` for `data/synthetic/` and scikit-learn):

```bash
python scripts/make_monitoring_fixtures.py
```

`tests/test_monitoring_fixtures.py` fails if these files stop matching the current schema or report shape. If that happens, regenerate them.
