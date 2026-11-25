"""Prediction-log contract: schema version, feature order, and table definition.

The feature order and difficulty levels mirror ``src/track_b/schema.py`` (model gate outcome B/C).
They are copied rather than imported so the app does not pull in pandas on the request path;
``tests/test_monitoring_logging.py`` fails if the two ever disagree.
"""
from __future__ import annotations

SCHEMA_VERSION = "1.0.0"

FEATURE_COLUMNS: tuple[str, ...] = (
    "topic",
    "selected_difficulty",
    "recent_accuracy",
    "mistake_count",
    "hint_count",
    "response_time_seconds",
    "prior_interaction_count",
    "previous_difficulty",
)

DIFFICULTY_LEVELS: tuple[str, ...] = ("Easy", "Medium", "Hard")  # ordinal: Easy < Medium < Hard

SOURCES = ("live", "simulated")
STATUSES = ("ok", "timeout", "error", "fallback")
LABEL_STATUSES = ("pending", "joined", "expired")

DDL = """
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
  label_status     TEXT NOT NULL DEFAULT 'pending'
                   CHECK (label_status IN ('pending','joined','expired')),
  label_joined_at  TEXT
);
CREATE INDEX IF NOT EXISTS idx_pred_time ON predictions (occurred_at);
CREATE UNIQUE INDEX IF NOT EXISTS idx_pred_learner_seq ON predictions (learner_key, learner_seq, source);
CREATE INDEX IF NOT EXISTS idx_pred_label ON predictions (label_status, occurred_at);

-- One row per evaluated monitoring window; `report` is the full window report JSON (§4.2).
CREATE TABLE IF NOT EXISTS monitoring_windows (
  window_id      TEXT PRIMARY KEY,
  window_start   TEXT NOT NULL,
  window_end     TEXT NOT NULL,
  evaluated_at   TEXT NOT NULL,
  n_predictions  INTEGER NOT NULL,
  n_labelled     INTEGER NOT NULL,
  status         TEXT NOT NULL CHECK (status IN ('ok','alert','insufficient_data')),
  report         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_windows_end ON monitoring_windows (window_end);

-- Alerts raised by monitoring (and, later, by the retraining trigger).
CREATE TABLE IF NOT EXISTS alerts (
  alert_id     TEXT PRIMARY KEY,
  created_at   TEXT NOT NULL,
  window_id    TEXT,
  severity     TEXT NOT NULL CHECK (severity IN ('warning','critical')),
  category     TEXT NOT NULL,
  metric       TEXT NOT NULL,
  value        REAL,
  threshold    REAL,
  message      TEXT NOT NULL,
  acknowledged INTEGER NOT NULL DEFAULT 0 CHECK (acknowledged IN (0,1))
);
CREATE INDEX IF NOT EXISTS idx_alerts_time ON alerts (created_at);

-- Small key/value store for the scheduler (last backup date, heartbeat).
CREATE TABLE IF NOT EXISTS monitor_state (
  key        TEXT PRIMARY KEY,
  value      TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
"""
