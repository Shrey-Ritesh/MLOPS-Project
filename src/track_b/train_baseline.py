"""Train and evaluate the Track B baseline (logistic regression vs majority class).

    python -m src.track_b.train_baseline --config configs/track_b/baseline.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import sklearn
import yaml

from src.track_b.modeling import build_pipeline, evaluate, file_sha256, load_dataset, split_by_learner
from src.track_b.schema import FEATURE_COLUMNS, SCHEMA_VERSION, TARGET


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/track_b/baseline.yaml"))
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    df, meta = load_dataset(cfg["dataset"])
    train, test = split_by_learner(df, meta)
    model = build_pipeline(cfg["model"])
    model.fit(train[FEATURE_COLUMNS], train[TARGET])
    threshold = cfg["decision_threshold"]

    model_dir = Path(cfg["outputs"]["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "model.joblib")
    card = {
        "model_id": cfg["model_id"],
        "feature_columns": FEATURE_COLUMNS,
        "target": TARGET,
        "decision_threshold": threshold,
        "schema_version": SCHEMA_VERSION,
        "dataset": cfg["dataset"],
        "dataset_sha256": file_sha256(cfg["dataset"]),
        "dataset_scenario_id": meta["scenario_id"],
        "dataset_scenario_sha256": meta["scenario_sha256"],
        "config": str(args.config),
        "config_sha256": file_sha256(args.config),
        "sklearn_version": sklearn.__version__,
        "trained_on": {"n_rows": int(len(train)), "n_learners": int(train["learner_id"].nunique())},
        "label": "Trained and evaluated on SYNTHETIC data only; not validated for real students.",
    }
    (model_dir / "model_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")

    report = {
        "label": "Performance on SYNTHETIC data (held-out learners)",
        "model_id": cfg["model_id"],
        "dataset": cfg["dataset"],
        "dataset_sha256": card["dataset_sha256"],
        "split": "by learner; test learners listed in the dataset's .meta.json",
        "train": evaluate(model, train, threshold),
        "test": evaluate(model, test, threshold),
    }
    out = Path(cfg["outputs"]["evaluation_report"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    t = report["test"]
    print(f"held-out test: F1={t['model']['f1']} P={t['model']['precision']} R={t['model']['recall']} "
          f"AUC={t['model']['roc_auc']} | majority F1={t['majority_class_baseline']['f1']} "
          f"acc={t['majority_class_baseline']['accuracy']} | model acc={t['model_accuracy']}")


if __name__ == "__main__":
    main()
