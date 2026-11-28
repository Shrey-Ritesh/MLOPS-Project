"""Build the shared Phase 3 monitoring fixtures in tests/fixtures/monitoring/.

SYNTHETIC ONLY. Replays Track B synthetic learners through the real PredictionLogger, joins
labels with the real label joiner, and evaluates every window with the real monitoring job,
so the fixture has exactly the production table layout and report shape.

Timeline (2026-11-20, UTC, 15-minute windows, 24 windows = 6 hours):
  00:00-03:00  baseline traffic (learner_interactions_v1)            windows 0-11
  03:00-06:00  shifted traffic  (learner_interactions_shifted_v1)    windows 12-23  <- shift point
  shift:       documented Track B shift (concept drift, little input change) PLUS a fixture-injected
               input shift: response_time_seconds x0.45 from 03:00 (faster answers)
  incidents:   01:00 model outage (most calls fall back)             window 4
               01:45 latency spike                                   window 7
               02:15 quiet window (4 predictions)                    window 9
               04:30 burst of an unseen topic tag                    window 18

The "model" is a stand-in logistic regression trained here on baseline rows (not a registered
model, model_version "fixture-1"); it only exists to give realistic scores that degrade under
the shift. Drift and performance sections come from a FIXTURE evaluator (PSI/KS via
src/track_b/drift.py on numeric features, F1/ROC-AUC/Brier on joined labels), standing in for
Person 2's src/monitoring/drift_windows.py so the dashboard has realistic data to draw.

    python scripts/make_monitoring_fixtures.py        # needs data/synthetic (dvc pull) + scikit-learn
"""
from __future__ import annotations

import hashlib
import json
import random
import uuid
import shutil
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.monitoring.label_joiner import join_labels  # noqa: E402
from src.monitoring.prediction_log import PredictionLogger, PredictionRecord  # noqa: E402
from src.monitoring.run_window import evaluate_window, iso, load_config  # noqa: E402
from src.monitoring.contract import load_contract  # noqa: E402
from src.track_b import drift  # noqa: E402

CONTRACT = load_contract("configs/monitoring/contracts/track_b_difficulty_v1.yaml")
FEATURE_COLUMNS = list(CONTRACT.feature_names)
OUT = Path("tests/fixtures/monitoring")
DB = OUT / "monitoring_sample.sqlite3"
KEY = "fixture-pseudonym-key-not-a-secret"
T0 = datetime(2026, 11, 20, 0, 0, tzinfo=timezone.utc)
WINDOW = timedelta(minutes=15)
N_WINDOWS = 24
SHIFT_WINDOW = 12
PER_WINDOW = 100
OUTAGE, LATENCY_SPIKE, QUIET, UNSEEN = 4, 7, 9, 18
THRESHOLD = 0.5
RESPONSE_TIME_FACTOR = 0.45
NO_ALERT_FEATURES = {"prior_interaction_count"}  # reported, not alerted (see README)
NUMERIC = ["recent_accuracy", "mistake_count", "hint_count", "response_time_seconds", "prior_interaction_count"]
CATEGORICAL = ["topic", "selected_difficulty", "previous_difficulty"]
PSI_ALERT, KS_ALPHA, MIN_LABELLED = 0.2, 0.01, 30

rng = random.Random(20261120)


def win_start(i: int) -> datetime:
    return T0 + i * WINDOW


def train_stand_in(baseline: pd.DataFrame):
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("num", StandardScaler(), NUMERIC),
    ])
    model = make_pipeline(pre, LogisticRegression(max_iter=500, random_state=0))
    model.fit(baseline[FEATURE_COLUMNS], baseline["switch_to_lower_difficulty"])
    return model


def schedule(df: pd.DataFrame, phase_start: datetime, n_windows: int, prefix: str,
             skip: tuple[datetime, datetime] | None) -> list[tuple[datetime, str, dict]]:
    """Steady-state replay: ~PER_WINDOW predictions in every window of the phase.

    Each learner plays ONE short session (4-8 consecutive interactions) starting at a random point
    of their real history, at a random time that may be before the phase starts. Only events
    inside the phase are kept. Two lessons for the traffic generator (Person 2):
      * replaying every learner from interaction 0 creates fake drift in prior_interaction_count;
      * long sessions from few learners make a window's rows strongly correlated, so per-window
        PSI behaves as if the sample were much smaller and false alarms rise.
    """
    phase_end = phase_start + n_windows * WINDOW
    lead_in = timedelta(minutes=15)  # roughly one session length
    target = n_windows * PER_WINDOW
    by_learner = {lid: rows.sort_values("prior_interaction_count").reset_index(drop=True)
                  for lid, rows in df.groupby("learner_id", sort=True)}
    anchors = list(range(len(df)))
    rng.shuffle(anchors)  # every interaction equally likely to anchor a session (row-uniform)
    used: set[str] = set()
    events: list[tuple[datetime, str, dict]] = []
    span = (phase_end - phase_start + lead_in).total_seconds()
    for idx in anchors:
        if len(events) >= target:
            break
        anchor = df.iloc[idx]
        if anchor["learner_id"] in used:
            continue
        used.add(anchor["learner_id"])
        rows = by_learner[anchor["learner_id"]]
        session = rng.randint(4, 8)                       # one short practice session per learner
        pos = int(anchor["prior_interaction_count"])
        first = max(0, min(pos - rng.randrange(session), len(rows) - session))  # session around the anchor
        rows = rows.iloc[first:first + session]
        t = phase_start - lead_in + timedelta(seconds=rng.uniform(0, span))
        gap = rng.uniform(45, 150)
        for _, row in rows.iterrows():
            t_out = t + (skip[1] - skip[0]) if skip and t >= skip[0] else t  # traffic pauses in the quiet window
            if t_out >= phase_end:
                break
            if t_out >= phase_start:
                events.append((t_out, f"{prefix}-{anchor['learner_id']}", row.to_dict()))
            t += timedelta(seconds=gap * rng.uniform(0.7, 1.3))
    return events


def fixture_evaluator(reference: pd.DataFrame):
    """Stand-in for Person 2's drift_windows.evaluate_window (see module docstring)."""
    ref_scores = reference["_score"]

    def evaluate(rows: list[dict]) -> dict:
        scored = [r for r in rows if isinstance(r["features"], dict) and r["score"] is not None]
        if len(scored) < 20:
            return {"drift": "insufficient_data", "prediction_psi": "insufficient_data",
                    "performance": "insufficient_data", "reference_dataset_sha256": REF_SHA}
        cur = pd.DataFrame([r["features"] for r in scored])
        result, alerts = {}, []
        for name in NUMERIC:
            psi = drift.psi_numeric(reference[name], cur[name].astype(float))
            ks = drift.ks_test(reference[name], cur[name].astype(float))
            status = "drift" if psi >= PSI_ALERT or ks["p_value"] < KS_ALPHA else "stable"
            result[name] = {"psi": round(psi, 4), "ks_stat": ks["statistic"], "ks_p": ks["p_value"], "status": status}
            if psi >= PSI_ALERT and name not in NO_ALERT_FEATURES:
                alerts.append({"severity": "warning", "category": "drift", "metric": f"psi.{name}",
                               "value": round(psi, 4), "threshold": PSI_ALERT,
                               "message": f"{name} PSI {psi:.3f} >= {PSI_ALERT}"})
        for name in ("selected_difficulty", "previous_difficulty"):
            psi = drift.psi_categorical(reference[name], cur[name])
            result[name] = {"psi": round(psi, 4), "ks_stat": None, "ks_p": None,
                            "status": "drift" if psi >= PSI_ALERT else "stable"}
        pred_psi = round(drift.psi_numeric(ref_scores, pd.Series([r["score"] for r in scored])), 4)
        labelled = [r for r in scored if r["label_observed"] is not None]
        if len(labelled) >= MIN_LABELLED and len({r["label_observed"] for r in labelled}) == 2:
            y = [r["label_observed"] for r in labelled]
            p = [r["score"] for r in labelled]
            yhat = [int(s >= THRESHOLD) for s in p]
            performance = {"f1": round(f1_score(y, yhat), 4), "roc_auc": round(roc_auc_score(y, p), 4),
                           "brier": round(brier_score_loss(y, p), 4), "n": len(y)}
        else:
            performance = "insufficient_data"
        return {"drift": result, "prediction_psi": pred_psi, "performance": performance,
                "reference_dataset_sha256": REF_SHA, "alerts": alerts}

    return evaluate


REF_SHA = hashlib.sha256(Path("data/synthetic/learner_interactions_v1.csv").read_bytes()).hexdigest()


def main() -> int:
    baseline = pd.read_csv("data/synthetic/learner_interactions_v1.csv")
    shifted = pd.read_csv("data/synthetic/learner_interactions_shifted_v1.csv")
    model = train_stand_in(baseline)
    reference = baseline.sample(5000, random_state=0).reset_index(drop=True)
    reference["_score"] = model.predict_proba(reference[FEATURE_COLUMNS])[:, 1]

    quiet = (win_start(QUIET), win_start(QUIET + 1))
    events = schedule(baseline, T0, SHIFT_WINDOW, "base", skip=quiet)
    events += schedule(shifted, win_start(SHIFT_WINDOW), N_WINDOWS - SHIFT_WINDOW, "shift", skip=None)
    # A short session inside the quiet window so it is small but not empty.
    lone = baseline[baseline.learner_id == baseline.learner_id.iloc[0]].head(4)
    for k, (_, row) in enumerate(lone.iterrows()):
        events.append((quiet[0] + timedelta(minutes=2 + 3 * k), "quiet-L0", row.to_dict()))
    events.sort(key=lambda e: e[0])

    OUT.mkdir(parents=True, exist_ok=True)
    for stale in (DB, DB.with_name(DB.name + "-wal"), DB.with_name(DB.name + "-shm")):
        stale.unlink(missing_ok=True)
    logger = PredictionLogger(DB, KEY, contract=CONTRACT)

    for t, learner, row in events:
        window = int((t - T0) / WINDOW)
        features = {name: row[name] for name in FEATURE_COLUMNS}
        for name in NUMERIC:
            features[name] = float(features[name]) if name in ("recent_accuracy", "response_time_seconds") else int(features[name])
        if window >= SHIFT_WINDOW:
            # Fixture-injected INPUT shift (faster answers, plan Day 4 example). The documented
            # Track B shift alone barely moves input distributions (max PSI 0.056, hint_count).
            features["response_time_seconds"] = round(features["response_time_seconds"] * RESPONSE_TIME_FACTOR, 1)
        if window == UNSEEN and rng.random() < 0.3:
            features["topic"] = "Organic Name Reactions"
        learner_seq = int(row["prior_interaction_count"])  # consecutive within each replayed learner

        latency = rng.lognormvariate(1.4, 0.35)  # ~4 ms typical
        if window == LATENCY_SPIKE:
            latency = rng.uniform(300, 900)
        common = dict(request_id=uuid.UUID(int=rng.getrandbits(128)).hex, source="simulated", learner_id=learner,
                      learner_seq=learner_seq, question_id=f"fixture-q-{rng.randint(1, 400)}",
                      features=features, occurred_at=iso(t))
        if window == OUTAGE and rng.random() < 0.7:
            rule = int(features["recent_accuracy"] < 0.4)
            record = PredictionRecord(**common, model_name="rule_fallback", model_version="fallback-v1",
                                      latency_ms=round(rng.uniform(0.2, 0.6), 3), status="fallback",
                                      fallback=True, label=rule, error_type="ModelUnavailable")
        else:
            score = float(model.predict_proba(pd.DataFrame([features])[FEATURE_COLUMNS])[0, 1])
            record = PredictionRecord(**common, model_name="learner-difficulty", model_version="fixture-1",
                                      model_id="fixture_logreg_stand_in", registry_alias="production",
                                      latency_ms=round(latency, 3), status="ok", score=round(score, 6),
                                      label=int(score >= THRESHOLD), threshold=THRESHOLD)
        assert logger.log(record), "fixture row rejected by the logger"

    end = win_start(N_WINDOWS)
    join_labels(DB, now=end + timedelta(hours=1), contract=CONTRACT)
    cfg = load_config()
    evaluator = fixture_evaluator(reference)
    reports = []
    for i in range(N_WINDOWS):
        reports.append(evaluate_window(DB, win_start(i), win_start(i + 1), config=cfg, evaluator=evaluator,
                                       now=win_start(i + 1) + timedelta(minutes=5), contract=CONTRACT))

    conn = sqlite3.connect(DB)
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.execute("PRAGMA journal_mode = DELETE")
    conn.execute("VACUUM")
    conn.close()

    picks = {"ok_baseline": 2, "outage_fallback": OUTAGE, "latency_spike": LATENCY_SPIKE,
             "insufficient_data": QUIET, "first_shifted": SHIFT_WINDOW, "unseen_topic": UNSEEN, "late_shifted": 22}
    rep_dir = OUT / "window_reports"
    shutil.rmtree(rep_dir, ignore_errors=True)
    rep_dir.mkdir()
    for name, i in picks.items():
        (rep_dir / f"{name}.json").write_text(json.dumps(reports[i], indent=2, default=str), encoding="utf-8")

    summary = [{"window": r["window_id"], "status": r["status"], "n": r["n_predictions"],
                "labelled": r["n_labelled"],
                "f1": r["performance"]["f1"] if isinstance(r["performance"], dict) else r["performance"],
                "alerts": [a["metric"] for a in r["alerts"]]} for r in reports]
    (OUT / "timeline.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"{DB} {DB.stat().st_size // 1024} KB, {len(events)} predictions, {N_WINDOWS} windows")
    for s in summary:
        print(s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
