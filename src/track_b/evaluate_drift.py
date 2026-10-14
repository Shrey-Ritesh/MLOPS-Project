"""Measure data drift and baseline performance on the shifted dataset, as separate reports.

    python -m src.track_b.evaluate_drift
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import yaml

from src.track_b.drift import ks_test, psi_categorical, psi_label, psi_numeric
from src.track_b.modeling import evaluate, file_sha256, load_dataset, split_by_learner
from src.track_b.schema import TARGET


def build_drift_report(ref, cur, cfg: dict) -> dict:
    dm = cfg["drift_measures"]
    numeric = {}
    for col in dm["numeric_features"]:
        psi = psi_numeric(ref[col], cur[col], dm["psi_bins"])
        ks = ks_test(ref[col], cur[col])
        numeric[col] = {
            "psi": round(psi, 4), "psi_reading": psi_label(psi),
            "ks_statistic": ks["statistic"], "ks_p_value": ks["p_value"],
            "ks_significant": bool(ks["p_value"] < dm["ks_alpha"]),
            "reference_mean": round(float(ref[col].mean()), 4), "current_mean": round(float(cur[col].mean()), 4),
        }
    categorical = {}
    for col in dm["categorical_features"]:
        psi = psi_categorical(ref[col], cur[col])
        rp = ref[col].value_counts(normalize=True)
        cp = cur[col].value_counts(normalize=True)
        shifts = (cp.reindex(rp.index.union(cp.index), fill_value=0) - rp.reindex(rp.index.union(cp.index), fill_value=0))
        top = shifts.abs().sort_values(ascending=False).head(5).index
        categorical[col] = {
            "psi": round(psi, 4), "psi_reading": psi_label(psi),
            "largest_share_changes": {k: round(float(shifts[k]), 4) for k in top},
        }
    target = {
        "reference_prevalence": round(float(ref[TARGET].mean()), 4),
        "current_prevalence": round(float(cur[TARGET].mean()), 4),
        "psi": round(psi_categorical(ref[TARGET], cur[TARGET]), 4),
    }
    flagged = [c for c, v in {**numeric, **categorical}.items() if v["psi"] >= dm["psi_alert_threshold"]]
    return {
        "numeric_features": numeric,
        "categorical_features": categorical,
        "target": target,
        "features_over_psi_alert_threshold": flagged,
        "psi_alert_threshold": dm["psi_alert_threshold"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/track_b/drift_experiment.yaml"))
    parser.add_argument("--baseline-dir", type=Path, default=Path("models/track_b/baseline"))
    parser.add_argument("--drift-out", type=Path, default=Path("reports/track_b/drift_report.json"))
    parser.add_argument("--shifted-eval-out", type=Path, default=Path("reports/track_b/shifted_evaluation.json"))
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    ref, ref_meta = load_dataset(cfg["datasets"]["baseline"])
    cur, cur_meta = load_dataset(cfg["datasets"]["shifted"])
    model = joblib.load(args.baseline_dir / "model.joblib")
    card = json.loads((args.baseline_dir / "model_card.json").read_text(encoding="utf-8"))

    drift = {
        "kind": "DATA DRIFT (input and target distributions); says nothing by itself about model quality",
        "reference": {"dataset": cfg["datasets"]["baseline"], "sha256": file_sha256(cfg["datasets"]["baseline"]),
                      "scenario_id": ref_meta["scenario_id"]},
        "current": {"dataset": cfg["datasets"]["shifted"], "sha256": file_sha256(cfg["datasets"]["shifted"]),
                    "scenario_id": cur_meta["scenario_id"]},
        "documented_changes": cfg["overrides"],
        **build_drift_report(ref, cur, cfg),
    }
    args.drift_out.parent.mkdir(parents=True, exist_ok=True)
    args.drift_out.write_text(json.dumps(drift, indent=2), encoding="utf-8")

    _, ref_test = split_by_learner(ref, ref_meta)
    _, cur_test = split_by_learner(cur, cur_meta)
    thr = card["decision_threshold"]
    on_original = evaluate(model, ref_test, thr)
    on_shifted = evaluate(model, cur_test, thr)
    perf = {
        "kind": "PERFORMANCE CHANGE of the ORIGINAL baseline model on shifted data (synthetic labels)",
        "baseline_model": card["model_id"],
        "original_test": on_original,
        "shifted_test": on_shifted,
        "change": {
            m: round(on_shifted["model"][m] - on_original["model"][m], 4)
            for m in ("precision", "recall", "f1", "roc_auc")
        },
        "note": "Drift does not by itself imply degradation, and errors alone do not establish drift; see drift_report.json.",
    }
    args.shifted_eval_out.write_text(json.dumps(perf, indent=2), encoding="utf-8")

    print("PSI > alert:", drift["features_over_psi_alert_threshold"])
    print("target prevalence:", drift["target"])
    print("baseline F1 original->shifted:", on_original["model"]["f1"], "->", on_shifted["model"]["f1"],
          "| AUC:", on_original["model"]["roc_auc"], "->", on_shifted["model"]["roc_auc"], "| change:", perf["change"])


if __name__ == "__main__":
    main()
