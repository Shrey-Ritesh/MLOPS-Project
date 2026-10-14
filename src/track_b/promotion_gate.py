"""Apply the predefined promotion gate to the candidate and record the decision.

    python -m src.track_b.promotion_gate

Reads thresholds from configs/track_b/promotion_gate.yaml (committed before the candidate was
trained). If any mandatory criterion fails, the candidate is rejected and the baseline retained.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
from pathlib import Path

import yaml

from src.track_b.modeling import file_sha256, load_dataset, split_by_learner
from src.track_b.schema import check_feature_list, SchemaError, validate_dataset


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()


def gate_integrity(gate_path: Path, candidate_trained_at: str) -> dict:
    """Evidence that the gate file was committed, unmodified, before the candidate was trained."""
    try:
        commit = _git("log", "-1", "--format=%H", "--", str(gate_path))
        committed_at = _git("log", "-1", "--format=%cI", "--", str(gate_path))
        clean = subprocess.run(["git", "diff", "--quiet", "--", str(gate_path)]).returncode == 0
        before = dt.datetime.fromisoformat(committed_at) <= dt.datetime.fromisoformat(candidate_trained_at)
        return {"gate_commit": commit, "gate_committed_at": committed_at, "gate_file_unmodified": clean,
                "committed_before_candidate_trained": before}
    except Exception as exc:  # no git history available
        return {"gate_commit": None, "error": str(exc), "gate_file_unmodified": False,
                "committed_before_candidate_trained": False}


def evaluate_gate(gate: dict, comparison: dict, checks: dict) -> list[dict]:
    c = gate["criteria"]
    base, cand = comparison["baseline"], comparison["candidate"]
    results = []

    q = c["minimum_predictive_quality"]
    value = cand["shifted_test"]["model"][q["metric"]]
    results.append({"criterion": "minimum_predictive_quality", "value": value, "threshold": q["minimum"], "passed": value >= q["minimum"]})

    i = c["improvement_over_baseline"]
    gain = cand["shifted_test"]["model"][i["metric"]] - base["shifted_test"]["model"][i["metric"]]
    results.append({"criterion": "improvement_over_baseline", "value": round(gain, 4), "threshold": i["minimum_gain"], "passed": gain >= i["minimum_gain"]})

    r = c["regression_tolerance"]
    f1_drop = base["original_test"]["model"]["f1"] - cand["original_test"]["model"]["f1"]
    auc_drop = base["original_test"]["model"]["roc_auc"] - cand["original_test"]["model"]["roc_auc"]
    results.append({"criterion": "regression_tolerance_f1", "value": round(f1_drop, 4), "threshold": r["max_f1_drop"], "passed": f1_drop <= r["max_f1_drop"]})
    results.append({"criterion": "regression_tolerance_roc_auc", "value": round(auc_drop, 4), "threshold": r["max_roc_auc_drop"], "passed": auc_drop <= r["max_roc_auc_drop"]})

    for name in ("schema_validation", "leakage_check", "learner_level_split"):
        results.append({"criterion": f"data_validity.{name}", "value": checks[name], "threshold": "required", "passed": bool(checks[name])})

    present = checks["reproducibility_fields_present"]
    results.append({"criterion": "reproducibility", "value": present, "threshold": c["reproducibility"]["required_fields"],
                    "passed": all(present.values())})
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", type=Path, default=Path("configs/track_b/promotion_gate.yaml"))
    parser.add_argument("--config", type=Path, default=Path("configs/track_b/drift_experiment.yaml"))
    parser.add_argument("--out", type=Path, default=Path("reports/track_b/promotion_decision.json"))
    args = parser.parse_args()

    gate = yaml.safe_load(args.gate.read_text(encoding="utf-8"))
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    cand_cfg = cfg["candidate"]
    comparison = json.loads(Path(cand_cfg["outputs"]["comparison_report"]).read_text(encoding="utf-8"))
    card = json.loads((Path(cand_cfg["outputs"]["model_dir"]) / "model_card.json").read_text(encoding="utf-8"))

    # Data validity checks, recomputed here rather than trusted from earlier steps.
    original, original_meta = load_dataset(cfg["datasets"]["baseline"])
    shifted, shifted_meta = load_dataset(cfg["datasets"]["shifted"])
    schema_ok = not validate_dataset(original) and not validate_dataset(shifted)
    try:
        check_feature_list(card["feature_columns"])
        leakage_ok = True
    except SchemaError:
        leakage_ok = False
    train, test = split_by_learner(shifted, shifted_meta)
    split_ok = not (set(train["learner_id"]) & set(test["learner_id"])) and not card["test_learners_used_for_training"]
    checks = {
        "schema_validation": schema_ok,
        "leakage_check": leakage_ok,
        "learner_level_split": split_ok,
        "reproducibility_fields_present": {f: bool(card.get(f)) for f in gate["criteria"]["reproducibility"]["required_fields"]},
    }

    results = evaluate_gate(gate, comparison, checks)
    integrity = gate_integrity(args.gate, card["trained_at_utc"])
    promoted = all(r["passed"] for r in results)
    decision = {
        "gate_id": gate["gate_id"],
        "decision": "PROMOTE_CANDIDATE" if promoted else "REJECT_CANDIDATE_RETAIN_BASELINE",
        "candidate": card["model_id"],
        "baseline": comparison["baseline"]["model_id"],
        "criteria_results": results,
        "failed_criteria": [r["criterion"] for r in results if not r["passed"]],
        "gate_integrity": integrity,
        "versions": {
            "gate_config_sha256": file_sha256(args.gate),
            "experiment_config_sha256": file_sha256(args.config),
            "baseline_dataset_sha256": comparison["datasets"]["original"],
            "shifted_dataset_sha256": comparison["datasets"]["shifted"],
            "candidate_git_commit": card["git_commit"],
            "candidate_seed": card["seed"],
        },
        "decided_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "label": "Decision within a SYNTHETIC experiment; not a decision about real students.",
    }
    args.out.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    for r in results:
        print(f"{'PASS' if r['passed'] else 'FAIL'}  {r['criterion']:34} value={r['value']} threshold={r['threshold']}")
    print("DECISION:", decision["decision"], "| gate integrity:", {k: integrity[k] for k in ("gate_file_unmodified", "committed_before_candidate_trained")})


if __name__ == "__main__":
    main()
