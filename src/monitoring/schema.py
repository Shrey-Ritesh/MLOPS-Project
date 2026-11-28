"""Monitoring database layout: schema version and table definitions.

Features and labels are NOT defined here: they come from the active model contract
(``src/monitoring/contract.py``, ``configs/monitoring/contracts/*.yaml``).
"""
from __future__ import annotations

SCHEMA_VERSION = "1.1.0"  # 1.1.0: contract_id column, attempt_outcomes table, 'excluded' label status

SOURCES = ("live", "simulated")
STATUSES = ("ok", "timeout", "error", "fallback")
LABEL_STATUSES = ("pending", "joined", "expired", "excluded")

DDL = """
CREATE TABLE IF NOT EXISTS predictions (
  prediction_id    TEXT PRIMARY KEY,
  schema_version   TEXT NOT NULL,
  contract_id      TEXT NOT NULL,
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
                   CHECK (label_status IN ('pending','joined','expired','excluded')),
  label_joined_at  TEXT
);
CREATE INDEX IF NOT EXISTS idx_pred_time ON predictions (occurred_at);
CREATE UNIQUE INDEX IF NOT EXISTS idx_pred_learner_seq ON predictions (learner_key, learner_seq, source);
CREATE INDEX IF NOT EXISTS idx_pred_label ON predictions (label_status, occurred_at);

-- Answer outcomes, logged by the app after each evaluated answer. Used by label rules that need
-- them (next_outcome_same_topic). Same privacy rules as predictions: pseudonymous learner key only.
CREATE TABLE IF NOT EXISTS attempt_outcomes (
  outcome_id   TEXT PRIMARY KEY,
  occurred_at  TEXT NOT NULL,
  source       TEXT NOT NULL CHECK (source IN ('live','simulated')),
  learner_key  TEXT NOT NULL,
  topic        TEXT NOT NULL,
  question_id  TEXT NOT NULL,
  request_id   TEXT,
  outcome      TEXT NOT NULL CHECK (outcome IN ('correct','incorrect','partial','uncertain'))
);
CREATE INDEX IF NOT EXISTS idx_outcomes_lookup ON attempt_outcomes (learner_key, source, topic, occurred_at);
CREATE INDEX IF NOT EXISTS idx_outcomes_time ON attempt_outcomes (occurred_at);

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
