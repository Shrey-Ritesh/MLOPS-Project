"""Generate reproducible synthetic learner interaction sequences (Track B demo).

Every behavioural rule and number lives in a scenario YAML (see
configs/track_b/learner_scenarios.yaml); they are simulation assumptions, not
findings about real students. A fixed seed makes the output reproducible.

    python -m src.track_b.generate_learners \
        --scenario configs/track_b/learner_scenarios.yaml \
        --out data/synthetic/learner_interactions_v1.csv
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.track_b.schema import (
    COLUMNS,
    DIFFICULTY_LEVELS,
    SCHEMA_VERSION,
    SchemaError,
    validate_dataset,
)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def deep_merge(base: dict, overrides: dict) -> dict:
    """Return `base` with `overrides` merged in recursively (overrides win)."""
    out = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_scenario(scenario_path: Path, overrides_path: Path | None = None) -> dict:
    scenario = yaml.safe_load(Path(scenario_path).read_text(encoding="utf-8"))
    if overrides_path is not None:
        overrides = yaml.safe_load(Path(overrides_path).read_text(encoding="utf-8"))
        scenario = deep_merge(scenario, overrides.get("overrides", {}))
        scenario["scenario_id"] = overrides.get("scenario_id", scenario["scenario_id"])
        scenario["seed"] = overrides.get("seed", scenario["seed"])
    return scenario


def load_priors(path: str | Path) -> pd.DataFrame:
    priors = pd.read_csv(path)
    priors["success_rate"] = priors["success_rate_pct"] / 100.0
    priors["p_easy"] = priors["easy_pct"] / 100.0
    priors["p_medium"] = priors["medium_pct"] / 100.0
    # Very Hard is folded into Hard for the Easy/Medium/Hard schema.
    priors["p_hard"] = (priors["hard_pct"] + priors["very_hard_pct"]) / 100.0
    return priors.set_index("topic")


def _simulate_learner(idx: int, scenario: dict, priors: pd.DataFrame, topics: list[str]) -> list[dict]:
    seed = scenario["seed"]
    rng = np.random.default_rng([seed, idx])
    pop, att, hnt, rt = scenario["population"], scenario["attempt_model"], scenario["hints"], scenario["response_time"]
    pol = scenario["next_difficulty_policy"]
    w = pol["weights"]
    overrides = scenario.get("topic_overrides", {})

    patterns = list(pop["pattern_mix"])
    pattern_name = rng.choice(patterns, p=[pop["pattern_mix"][p] for p in patterns])
    pat = scenario["patterns"][pattern_name]
    ability = rng.normal(pop["ability"]["mean"], pop["ability"]["sd"])

    n_rows = int(rng.integers(pop["interactions_per_learner"]["min"], pop["interactions_per_learner"]["max"] + 1))
    n_total = n_rows + 1  # the extra interaction provides the last row's "next" difficulty

    first_try: list[int] = []
    rows: list[dict] = []
    difficulty: int | None = None
    topic = None
    run_left = 0

    for t in range(n_total):
        if run_left == 0:
            topic = topics[int(rng.integers(len(topics)))]
            run_left = int(rng.integers(pop["topic_run_length"]["min"], pop["topic_run_length"]["max"] + 1))
        run_left -= 1
        pr = priors.loc[topic]
        score = float(pr["difficulty_score"]) + overrides.get(topic, {}).get("score_delta", 0.0)
        logit_shift = overrides.get(topic, {}).get("success_logit_delta", 0.0)

        if difficulty is None:
            difficulty = int(rng.choice(3, p=[pr["p_easy"], pr["p_medium"], pr["p_hard"]]))

        lo, hi = att["success_prob_clip"]
        q = min(max(_sigmoid(
            _logit(min(max(float(pr["success_rate"]), lo), hi))
            + att["ability_coef"] * ability
            - att["difficulty_coef"] * (difficulty - 1)
            + pat["success_logit_shift"]
            + logit_shift
        ), lo), hi)
        mistakes = int(min(rng.geometric(q) - 1, att["max_mistakes"]))

        lam = (
            hnt["base_rate"] * pat["hint_mult"]
            * max(0.1, 1 + hnt["score_slope"] * (score - 2.5))
            * (1 + hnt["mistake_slope"] * mistakes)
        )
        hints = int(min(rng.poisson(lam), hnt["max_hints"]))

        expected = float(pr["avg_time_min"]) * 60.0 * (1 + rt["difficulty_mult_per_level"] * difficulty)
        median = (
            expected * pat["speed_mult"] * math.exp(-rt["ability_exp"] * ability)
            * (1 + rt["mistake_mult"] * mistakes) * (1 + rt["hint_mult"] * hints)
        )
        seconds = float(np.clip(median * math.exp(rng.normal(0, rt["sigma"])), rt["min_seconds"], rt["max_seconds"]))

        window = scenario["recent_accuracy"]["window"]
        recent = first_try[-window:]
        recent_acc = float(np.mean(recent)) if recent else scenario["recent_accuracy"]["neutral_default"]
        prev_difficulty = rows[-1]["_difficulty"] if rows else difficulty

        # Next-difficulty choice from a struggle score plus logistic noise.
        z = (
            w["mistakes"] * mistakes
            + w["hints"] * pat["policy_hint_weight"] * hints
            + w["log_time_ratio"] * math.log(seconds / expected)
            + w["recent_accuracy"] * (0.5 - recent_acc)
            + w["topic_score"] * (score - 2.5)
        )
        down_thr = pol["switch_down_threshold"] + pat["switch_down_bias"]
        s = pol["noise_scale"]
        next_difficulty = difficulty
        if difficulty > 0 and rng.random() < _sigmoid((z - down_thr) / s):
            next_difficulty = difficulty - 1
        elif difficulty < 2 and rng.random() < _sigmoid((pol["switch_up_threshold"] - z) / s):
            next_difficulty = difficulty + 1

        rows.append({
            "_difficulty": difficulty,
            "_next": next_difficulty,
            "learner_id": f"L{idx:05d}",
            "interaction_id": f"L{idx:05d}-{t:03d}",
            "topic": topic,
            "selected_difficulty": DIFFICULTY_LEVELS[difficulty],
            "recent_accuracy": round(recent_acc, 4),
            "mistake_count": mistakes,
            "hint_count": hints,
            "response_time_seconds": round(seconds, 1),
            "prior_interaction_count": t,
            "previous_difficulty": DIFFICULTY_LEVELS[prev_difficulty],
            "next_selected_difficulty": DIFFICULTY_LEVELS[next_difficulty],
            "switch_to_lower_difficulty": int(next_difficulty < difficulty),
        })
        first_try.append(int(mistakes == 0))
        difficulty = next_difficulty

    kept = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows[:n_rows]]
    return kept, pattern_name


def generate(scenario: dict) -> tuple[pd.DataFrame, dict]:
    priors = load_priors(scenario["topic_priors_csv"])
    topics = list(priors.index)
    unknown = set(scenario.get("topic_overrides", {})) - set(topics)
    if unknown:
        raise SchemaError(f"topic_overrides name unknown topics: {sorted(unknown)}")

    all_rows: list[dict] = []
    patterns: dict[str, str] = {}
    for idx in range(scenario["population"]["n_learners"]):
        rows, pattern = _simulate_learner(idx, scenario, priors, topics)
        all_rows.extend(rows)
        patterns[f"L{idx:05d}"] = pattern
    df = pd.DataFrame(all_rows, columns=COLUMNS)

    problems = validate_dataset(df)
    if problems:
        raise SchemaError("generated dataset failed validation: " + "; ".join(problems[:5]))

    # Split by learner (never by interaction): deterministic from the seed.
    learners = sorted(df["learner_id"].unique())
    order = np.random.default_rng([scenario["seed"], 999_999]).permutation(len(learners))
    n_test = int(round(len(learners) * scenario["population"]["test_fraction"]))
    test_ids = sorted(learners[i] for i in order[:n_test])

    prevalence = float(df["switch_to_lower_difficulty"].mean())
    low, high = scenario["next_difficulty_policy"]["expected_target_prevalence"]
    meta = {
        "scenario_id": scenario["scenario_id"],
        "schema_version": SCHEMA_VERSION,
        "seed": scenario["seed"],
        "scenario_sha256": hashlib.sha256(json.dumps(scenario, sort_keys=True).encode()).hexdigest(),
        "n_learners": len(learners),
        "n_rows": int(len(df)),
        "target_prevalence": round(prevalence, 4),
        "target_prevalence_within_expected_range": bool(low <= prevalence <= high),
        "pattern_counts": {p: sum(1 for v in patterns.values() if v == p) for p in scenario["population"]["pattern_mix"]},
        "test_learner_ids": test_ids,
        "note": "Synthetic data: every behavioural rule is a documented simulation assumption.",
    }
    return df, meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=Path, default=Path("configs/track_b/learner_scenarios.yaml"))
    parser.add_argument("--overrides", type=Path, default=None, help="drift YAML with an `overrides:` block")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    scenario = load_scenario(args.scenario, args.overrides)
    df, meta = generate(scenario)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    meta_path = args.out.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"{args.out}: {meta['n_rows']} rows, {meta['n_learners']} learners, "
          f"target prevalence {meta['target_prevalence']}")


if __name__ == "__main__":
    main()
