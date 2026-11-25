# Prediction log schema: DRAFT v0.2, implemented (Person 1)

**Status:** implemented in `src/monitoring/` (tests: `tests/test_monitoring_logging.py`), still waiting for review by Person 2 (drift and performance inputs) and Person 3 (inference hook). Once they agree, it is frozen as `schema_version = "1.0.0"`. Any change after that needs a new schema version.
**Implements:** §4.1 of `docs/planning/phase3_weekly_work_split_roles_milestones.md`.
**Assumes:** model gate outcome B or C (the Track B feature contract). If the outcome is A, `features` changes to the XGBoost feature list. Nothing else in the schema changes.

## 1. Storage decision

| Choice | Decision | Why |
|---|---|---|
| Store | SQLite file `/app/data/state/monitoring.sqlite3` on the existing `app_state` volume, in WAL mode | The volume is already shown to survive deploys. SQLite costs nothing and needs no new service. WAL lets the monitoring job read while the app writes. |
| Separate file, not the `learner_history` DB | Yes | This file has its own retention and backup rules. A monitoring query can never lock the app's progress table. |
| Who writes | The Streamlit `app` process inserts rows (`PredictionLogger`). The label joiner only updates the three label columns. | SQLite WAL mode plus a 5 s busy timeout handles these two writers. No other column is ever updated. |
| Who reads | The monitoring job (sidecar), the dashboard, and the retraining trigger, all read-only | |
| Window export | The monitoring job writes `windows/<window_id>/predictions.parquet` plus `report.json` | The drift code and the dashboard read fixed snapshots. Reports can be reproduced later. |
| Backup | `python -m src.monitoring.backup` uses SQLite's online backup API, runs an integrity check, and keeps 7 copies. It is scheduled nightly by the Day 3 monitoring sidecar. | A consistent copy is taken while the app keeps writing. `--restore` will never overwrite an existing file. |

## 2. Table `predictions`: one row per prediction

| Column | Type | Null | Notes |
|---|---|---|---|
| `prediction_id` | TEXT PK | no | UUID4, created by the logger |
| `schema_version` | TEXT | no | `"1.0.0"` once frozen |
| `occurred_at` | TEXT | no | UTC ISO-8601, e.g. `2026-11-26T09:14:03.120Z` |
| `request_id` | TEXT | no | One per practice submission. It is also written into the JSON app log, so the two can be matched. |
| `source` | TEXT | no | `live` or `simulated`. Generator traffic is always `simulated`. |
| `learner_key` | TEXT | no | **Pseudonym:** `HMAC-SHA256(LOG_PSEUDONYM_KEY, learner_id)`, first 16 hex characters. The raw `learner_id` contains the username (see the audit, §4), so it is never stored. |
| `learner_seq` | INTEGER | no | This learner's interaction count when the prediction was made. It equals the `prior_interaction_count` feature. Label joins use it. |
| `question_id` | TEXT | no | |
| `topic` | TEXT | no | A topic from the Track B list, or `unknown`. `unknown` counts as an input-validity failure. |
| `features` | TEXT (JSON) | no | **Exactly what the model received, in contract order.** Draft contents: `{topic, selected_difficulty, recent_accuracy, mistake_count, hint_count, response_time_seconds, prior_interaction_count, previous_difficulty}` |
| `feature_hash` | TEXT | no | sha256 of the canonical JSON of `features` (sorted keys, no whitespace) |
| `score` | REAL | yes | Probability of class 1. NULL when `status` is `error`. |
| `label` | INTEGER | yes | `score >= threshold` gives 0 or 1 (or the fallback rule's output) |
| `threshold` | REAL | yes | The threshold the model used (0.7 for `candidate_hgb_shifted_v2`) |
| `model_name` | TEXT | no | Registered model name, e.g. `learner-difficulty`, or `rule_fallback` |
| `model_version` | TEXT | no | Registry version, e.g. `"3"`, or `"fallback-v1"` |
| `model_id` | TEXT | yes | Model card `model_id`, e.g. `candidate_hgb_shifted_v2` |
| `registry_alias` | TEXT | yes | The alias the model was loaded by (`production`) |
| `fallback` | INTEGER | no | 0 or 1 |
| `latency_ms` | REAL | no | Time spent in `predict()` only, measured with a monotonic clock |
| `status` | TEXT | no | `ok`, `timeout`, `error`, or `fallback` |
| `error_type` | TEXT | yes | Exception class name only. No message text, because messages can contain input values. |
| `label_observed` | INTEGER | yes | Filled in by the label joiner (§3) |
| `label_status` | TEXT | no | `pending`, `joined`, or `expired`. Starts as `pending`. |
| `label_joined_at` | TEXT | yes | UTC ISO-8601 |

**Indexes:** `(occurred_at)`, `(learner_key, learner_seq, source)` unique, and `(label_status, occurred_at)`.

**Never stored:** username, email, raw `learner_id`, question text, answer text, OCR output, image data, exception messages.

```sql
CREATE TABLE IF NOT EXISTS predictions (
  prediction_id    TEXT PRIMARY KEY,
  schema_version   TEXT NOT NULL,
  occurred_at      TEXT NOT NULL,
  request_id       TEXT NOT NULL,
  source           TEXT NOT NULL CHECK (source IN ('live','simulated')),
  learner_key      TEXT NOT NULL,
  learner_seq      INTEGER NOT NULL,
  question_id      TEXT NOT NULL,
  topic            TEXT NOT NULL,
  features         TEXT NOT NULL,
  feature_hash     TEXT NOT NULL,
  score            REAL,
  label            INTEGER CHECK (label IN (0,1)),
  threshold        REAL,
  model_name       TEXT NOT NULL,
  model_version    TEXT NOT NULL,
  model_id         TEXT,
  registry_alias   TEXT,
  fallback         INTEGER NOT NULL CHECK (fallback IN (0,1)),
  latency_ms       REAL NOT NULL,
  status           TEXT NOT NULL CHECK (status IN ('ok','timeout','error','fallback')),
  error_type       TEXT,
  label_observed   INTEGER CHECK (label_observed IN (0,1)),
  label_status     TEXT NOT NULL DEFAULT 'pending' CHECK (label_status IN ('pending','joined','expired')),
  label_joined_at  TEXT
);
CREATE INDEX IF NOT EXISTS idx_pred_time ON predictions (occurred_at);
CREATE UNIQUE INDEX IF NOT EXISTS idx_pred_learner_seq ON predictions (learner_key, learner_seq, source);
CREATE INDEX IF NOT EXISTS idx_pred_label ON predictions (label_status, occurred_at);
```

## 3. Label joining

The target, as defined in `src/track_b/generate_learners.py:166`, is: `switch_to_lower_difficulty = 1` if the learner's **next** interaction has a lower `selected_difficulty` (Easy < Medium < Hard).

The joiner runs at the start of each monitoring window:

1. For every `pending` row P, find row N with the same `learner_key` and `source`, where `N.learner_seq = P.learner_seq + 1`.
2. If N exists, set `label_observed = ord(N.selected_difficulty) < ord(P.selected_difficulty)`, `label_status = 'joined'`, and `label_joined_at = now`.
3. If no N appears within `LABEL_EXPIRY_HOURS` (draft: 72 h, so the learner effectively stopped), set `label_status = 'expired'`. The row stays out of performance metrics, and expired rows are counted in the window report.

Each update writes only the three label columns. Every other column stays as it was written.

Because the label is "the learner's next choice", logging must happen on **every** practice submission, including ones where the model fell back. Otherwise the chain of `learner_seq` values has gaps.

## 4. Structured app log line (stdout, JSON)

One line per request, collected by Docker:

```json
{"ts":"2026-11-26T09:14:03.121Z","level":"INFO","service":"app","event":"prediction",
 "request_id":"…","prediction_id":"…","model_version":"3","status":"ok","latency_ms":12.4,"source":"live"}
```

No learner identifiers and no feature values appear here. The `predictions` table is the record of what the model saw. This line is only for operations.

## 5. How to use it (for Person 3's inference hook)

```python
from src.monitoring.prediction_log import PredictionLogger, PredictionRecord
from src.monitoring.app_logging import new_request_id

prediction_logger = PredictionLogger.from_env()   # once per process; raises if LOG_PSEUDONYM_KEY is missing
prediction_logger.log(PredictionRecord(
    request_id=new_request_id(), source="live",
    learner_id=current_user["learner_id"], learner_seq=prior_interaction_count,
    question_id=qid, features=features,           # dict with exactly the 8 contract features
    score=score, label=label, threshold=0.7,
    model_name="learner-difficulty", model_version="2", model_id="candidate_hgb_shifted_v2",
    registry_alias="production", fallback=False, latency_ms=latency_ms, status="ok",
))  # returns prediction_id, or None if the write failed; it never raises
```

Rules the logger enforces:

- Features must be exactly the 8 contract keys. They are reordered to contract order.
- `status="ok"` needs a score and a label. `status="fallback"` needs `fallback=True`.
- A record that breaks the contract is rejected and reported as a `prediction_log_write_failed` event. It is never partly written.
- A repeated `(learner, learner_seq, source)`, for example from a double submit, is rejected the same way.
- `request_id` and `question_id` are stored exactly as given, so they **must not contain the learner ID**. Records that do are rejected. Use `new_request_id()`, which returns a random hex string.

Label joining: `python -m src.monitoring.label_joiner` (Day 3: run at the start of every monitoring window).

## 6. Questions for reviewers

- **Person 3:** can the hook provide `learner_seq`, `selected_difficulty`, `hint_count` and `response_time_seconds` on every submission? Is `predict()` called before or after the answer is evaluated? The features must use only information from before the prediction.
- **Person 2:** is the JSON `features` column enough for you, or do you want one column per feature in the Parquet export? (I plan to flatten it in the export.)
- **All:** `LOG_PSEUDONYM_KEY` becomes one new secret in the server's `.env`. If it rotates, the link between old and new rows for the same learner breaks, so it stays the same for the whole project.
- **Retention (input to Person 4):** draft rule is to keep raw rows for 30 days, then keep only window aggregates.
