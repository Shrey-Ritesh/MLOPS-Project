"""Train the candidate on the shifted population and compare it with the baseline.

Both models are evaluated under the same protocol on the same fixed held-out test learners
(the shifted dataset's and the original dataset's). Test learners are never used for training.

    python -m src.track_b.train_candidate
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
from pathlib import Path

import joblib
import sklearn
import yaml

from src.track_b.modeling import build_pipeline, evaluate, file_sha256, load_dataset, split_by_learner
from src.track_b.schema import FEATURE_COLUMNS, SCHEMA_VERSION, TARGET


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/track_b/drift_experiment.yaml"))
    parser.add_argument("--baseline-dir", type=Path, default=Path("models/track_b/baseline"))
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    cand = cfg["candidate"]

    shifted, shifted_meta = load_dataset(cand["dataset"])
    original, original_meta = load_dataset(cfg["datasets"]["baseline"])
    shifted_train, shifted_test = split_by_learner(shifted, shifted_meta)
    _, original_test = split_by_learner(original, original_meta)

    model = build_pipeline(cand["model"])
    model.fit(shifted_train[FEATURE_COLUMNS], shifted_train[TARGET])
    thr = cand["decision_threshold"]

    model_dir = Path(cand["outputs"]["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "model.joblib")
    card = {
        "model_id": cand["model_id"],
        "feature_columns": FEATURE_COLUMNS,
        "target": TARGET,
        "decision_threshold": thr,
        "schema_version": SCHEMA_VERSION,
        "dataset": cand["dataset"],
        "dataset_sha256": file_sha256(cand["dataset"]),
        "scenario_id": shifted_meta["scenario_id"],
        "scenario_sha256": shifted_meta["scenario_sha256"],
        "config": str(args.config),
        "config_sha256": file_sha256(args.config),
        "seed": shifted_meta["seed"],
        "git_commit": git_commit(),
        "sklearn_version": sklearn.__version__,
        "trained_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "trained_on": {"n_rows": int(len(shifted_train)), "n_learners": int(shifted_train["learner_id"].nunique())},
        "test_learners_used_for_training": sorted(set(shifted_train["learner_id"]) & set(shifted_test["learner_id"])),
        "label": "Trained and evaluated on SYNTHETIC data only; not validated for real students.",
    }
    (model_dir / "model_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")

    baseline = joblib.load(args.baseline_dir / "model.joblib")
    base_card = json.loads((args.baseline_dir / "model_card.json").read_text(encoding="utf-8"))
    results = {
        "baseline": {
            "model_id": base_card["model_id"],
            "shifted_test": evaluate(baseline, shifted_test, base_card["decision_threshold"]),
            "original_test": evaluate(baseline, original_test, base_card["decision_threshold"]),
        },
        "candidate": {
            "model_id": cand["model_id"],
            "shifted_test": evaluate(model, shifted_test, thr),
            "original_test": evaluate(model, original_test, thr),
        },
    }
    delta = {}
    for ds in ("shifted_test", "original_test"):
        delta[ds] = {
            m: round(results["candidate"][ds]["model"][m] - results["baseline"][ds]["model"][m], 4)
            for m in ("precision", "recall", "f1", "roc_auc")
        }
    report = {
        "label": "Candidate vs baseline on SYNTHETIC data, same fixed held-out learners",
        "protocol": "both models scored on the same shifted_test and original_test learners; test learners never used for training",
        "datasets": {"shifted": card["dataset_sha256"], "original": file_sha256(cfg["datasets"]["baseline"])},
        **results,
        "candidate_minus_baseline": delta,
    }
    out = Path(cand["outputs"]["comparison_report"])
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for ds in ("shifted_test", "original_test"):
        b, c = results["baseline"][ds]["model"], results["candidate"][ds]["model"]
        print(f"{ds:14} F1 baseline {b['f1']} -> candidate {c['f1']} | AUC {b['roc_auc']} -> {c['roc_auc']}")


if __name__ == "__main__":
    main()
