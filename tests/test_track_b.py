import copy

import numpy as np
import pandas as pd
import pytest
import yaml

from src.track_b.drift import psi_categorical, psi_from_proportions, psi_numeric
from src.track_b.generate_learners import deep_merge, generate, load_scenario
from src.track_b.modeling import split_by_learner
from src.track_b.promotion_gate import evaluate_gate
from src.track_b.schema import (
    BANNED_FEATURES,
    COLUMNS,
    FEATURE_COLUMNS,
    SCHEMA_VERSION,
    SchemaError,
    TARGET,
    check_feature_list,
    validate_dataset,
)

SCENARIO = "configs/track_b/learner_scenarios.yaml"
DRIFT = "configs/track_b/drift_experiment.yaml"
PRIORS = "configs/track_b/topic_difficulty_priors.csv"


@pytest.fixture(scope="module")
def small_scenario():
    scenario = load_scenario(SCENARIO)
    scenario["population"]["n_learners"] = 40
    return scenario


@pytest.fixture(scope="module")
def small_dataset(small_scenario):
    return generate(small_scenario)


def test_priors_have_40_valid_rows():
    p = pd.read_csv(PRIORS)
    assert len(p) == 40
    assert p["subject"].value_counts().to_dict() == {"Chemistry": 22, "Physics": 18}
    assert p["topic"].is_unique
    assert p["difficulty_score"].between(1, 5).all()
    assert p["success_rate_pct"].between(0, 100).all()
    assert ((p[["easy_pct", "medium_pct", "hard_pct", "very_hard_pct"]].sum(axis=1)) == 100).all()
    assert p["cross_check_note"].str.len().gt(10).all()


def test_generator_is_deterministic_for_a_fixed_seed(small_scenario):
    a, meta_a = generate(small_scenario)
    b, meta_b = generate(copy.deepcopy(small_scenario))
    pd.testing.assert_frame_equal(a, b)
    assert meta_a == meta_b
    other = copy.deepcopy(small_scenario)
    other["seed"] += 1
    c, _ = generate(other)
    assert not a.equals(c)


def test_generated_dataset_matches_schema_and_chains(small_dataset):
    df, meta = small_dataset
    assert list(df.columns) == COLUMNS
    assert validate_dataset(df) == []
    assert meta["n_learners"] == 40 and meta["n_rows"] == len(df)
    # selected difficulty of the next row equals the observed next choice of this row
    for _, g in df.groupby("learner_id"):
        assert (g["selected_difficulty"].to_numpy()[1:] == g["next_selected_difficulty"].to_numpy()[:-1]).all()


def test_validator_rejects_broken_target_and_chain(small_dataset):
    df = small_dataset[0].copy()
    broken = df.copy()
    broken.loc[broken.index[0], "switch_to_lower_difficulty"] = 1 - broken.loc[broken.index[0], "switch_to_lower_difficulty"]
    assert any("target" in p for p in validate_dataset(broken))
    assert validate_dataset(df.drop(columns=["hint_count"]))


def test_leakage_guard_rejects_banned_features():
    check_feature_list(FEATURE_COLUMNS)  # the real feature list is clean
    for banned in BANNED_FEATURES:
        with pytest.raises(SchemaError):
            check_feature_list(FEATURE_COLUMNS + [banned])
    with pytest.raises(SchemaError):
        check_feature_list(["not_a_column"])


def test_split_is_by_learner_and_disjoint(small_dataset):
    df, meta = small_dataset
    train, test = split_by_learner(df, meta)
    assert set(train["learner_id"]).isdisjoint(set(test["learner_id"]))
    assert len(test["learner_id"].unique()) == round(40 * 0.3)
    assert len(train) + len(test) == len(df)


def test_drift_overrides_change_only_documented_parameters():
    base = load_scenario(SCENARIO)
    shifted = load_scenario(SCENARIO, DRIFT)
    assert shifted["scenario_id"] == "shifted_v1" and shifted["seed"] != base["seed"]
    assert shifted["population"]["pattern_mix"]["hint_reliant"] > base["population"]["pattern_mix"]["hint_reliant"]
    assert set(shifted["topic_overrides"]) <= set(pd.read_csv(PRIORS)["topic"])
    unchanged = ["patterns", "attempt_model", "hints", "response_time", "next_difficulty_policy"]
    assert all(shifted[k] == base[k] for k in unchanged)


def test_deep_merge_does_not_mutate_inputs():
    base = {"a": {"b": 1, "c": 2}, "d": 3}
    merged = deep_merge(base, {"a": {"b": 9}})
    assert merged == {"a": {"b": 9, "c": 2}, "d": 3} and base["a"]["b"] == 1


def test_psi_known_values():
    assert psi_from_proportions(np.array([0.5, 0.5]), np.array([0.5, 0.5])) == pytest.approx(0.0, abs=1e-9)
    shifted = psi_from_proportions(np.array([0.5, 0.5]), np.array([0.9, 0.1]))
    assert shifted == pytest.approx((0.9 - 0.5) * np.log(0.9 / 0.5) + (0.1 - 0.5) * np.log(0.1 / 0.5))
    rng = np.random.default_rng(0)
    same = pd.Series(rng.normal(size=5000))
    assert psi_numeric(same, pd.Series(rng.normal(size=5000))) < 0.05
    assert psi_numeric(same, pd.Series(rng.normal(loc=2, size=5000))) > 0.25
    assert psi_categorical(pd.Series(list("aabb")), pd.Series(list("aabb"))) == pytest.approx(0.0, abs=1e-9)


def _comparison(cand_shifted_f1, base_shifted_f1=0.54, cand_orig_f1=0.58, cand_orig_auc=0.89):
    def block(f1, auc):
        return {"model": {"f1": f1, "roc_auc": auc}}
    return {
        "baseline": {"shifted_test": block(base_shifted_f1, 0.88), "original_test": block(0.59, 0.90)},
        "candidate": {"shifted_test": block(cand_shifted_f1, 0.88), "original_test": block(cand_orig_f1, cand_orig_auc)},
    }


GATE = yaml.safe_load(open("configs/track_b/promotion_gate.yaml", encoding="utf-8"))
OK_CHECKS = {
    "schema_validation": True,
    "leakage_check": True,
    "learner_level_split": True,
    "reproducibility_fields_present": {f: True for f in GATE["criteria"]["reproducibility"]["required_fields"]},
}


def test_gate_promotes_when_every_criterion_passes():
    results = evaluate_gate(GATE, _comparison(cand_shifted_f1=0.60), OK_CHECKS)
    assert all(r["passed"] for r in results)


@pytest.mark.parametrize(
    "comparison, failed",
    [
        (_comparison(cand_shifted_f1=0.50, base_shifted_f1=0.40), "minimum_predictive_quality"),
        (_comparison(cand_shifted_f1=0.60, base_shifted_f1=0.595), "improvement_over_baseline"),
        (_comparison(cand_shifted_f1=0.60, cand_orig_f1=0.50), "regression_tolerance_f1"),
        (_comparison(cand_shifted_f1=0.60, cand_orig_auc=0.80), "regression_tolerance_roc_auc"),
    ],
)
def test_gate_rejects_each_failing_criterion(comparison, failed):
    results = evaluate_gate(GATE, comparison, OK_CHECKS)
    assert [r["criterion"] for r in results if not r["passed"]] == [failed]


def test_gate_rejects_failed_validity_or_missing_reproducibility_fields():
    bad = {**OK_CHECKS, "leakage_check": False}
    assert "data_validity.leakage_check" in [r["criterion"] for r in evaluate_gate(GATE, _comparison(0.6), bad) if not r["passed"]]
    missing = {**OK_CHECKS, "reproducibility_fields_present": {**OK_CHECKS["reproducibility_fields_present"], "git_commit": False}}
    assert "reproducibility" in [r["criterion"] for r in evaluate_gate(GATE, _comparison(0.6), missing) if not r["passed"]]


def test_difficulty_boundary_guard_zeros_easy():
    from src.track_b.modeling import apply_difficulty_boundary_guard
    proba = np.array([0.9, 0.8, 0.7])
    diff = pd.Series(["Easy", "Medium", "Hard"])
    guarded = apply_difficulty_boundary_guard(proba, diff)
    assert guarded[0] == 0.0
    assert guarded[1] == 0.8
    assert guarded[2] == 0.7


def test_hgb_pipeline_construction_and_tuning(small_dataset):
    from src.track_b.modeling import build_pipeline, tune_threshold_on_train
    df, _ = small_dataset
    pipe = build_pipeline({"type": "hist_gradient_boosting", "max_iter": 10})
    pipe.fit(df[FEATURE_COLUMNS], df[TARGET])
    proba = pipe.predict_proba(df[FEATURE_COLUMNS])[:, 1]
    thr = tune_threshold_on_train(df[TARGET].to_numpy(), proba)
    assert 0.2 <= thr <= 0.85

