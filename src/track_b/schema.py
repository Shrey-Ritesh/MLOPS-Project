"""Interaction schema, leakage guard and dataset validation for the Track B demo."""

from __future__ import annotations

import pandas as pd

SCHEMA_VERSION = "1.0.0"

DIFFICULTY_LEVELS = ["Easy", "Medium", "Hard"]  # ordinal: Easy < Medium < Hard
DIFFICULTY_ORDINAL = {name: i for i, name in enumerate(DIFFICULTY_LEVELS)}

COLUMNS = [
    "learner_id",
    "interaction_id",
    "topic",
    "selected_difficulty",
    "recent_accuracy",
    "mistake_count",
    "hint_count",
    "response_time_seconds",
    "prior_interaction_count",
    "previous_difficulty",
    "next_selected_difficulty",
    "switch_to_lower_difficulty",
]
TARGET = "switch_to_lower_difficulty"

# Model inputs. Identifiers and anything from the future are excluded.
FEATURE_COLUMNS = [
    "topic",
    "selected_difficulty",
    "recent_accuracy",
    "mistake_count",
    "hint_count",
    "response_time_seconds",
    "prior_interaction_count",
    "previous_difficulty",
]
BANNED_FEATURES = {"learner_id", "interaction_id", "next_selected_difficulty", TARGET}


class SchemaError(ValueError):
    """Raised when a dataset or feature list violates the contract."""


def check_feature_list(features: list[str]) -> None:
    """Fail loudly if a banned (identifier / future / target) column is used as a feature."""
    banned = sorted(set(features) & BANNED_FEATURES)
    if banned:
        raise SchemaError(f"leakage: banned columns used as features: {banned}")
    unknown = sorted(set(features) - set(COLUMNS))
    if unknown:
        raise SchemaError(f"unknown feature columns: {unknown}")


def validate_dataset(df: pd.DataFrame) -> list[str]:
    """Return a list of schema/consistency problems (empty list = valid)."""
    problems: list[str] = []
    if list(df.columns) != COLUMNS:
        return [f"columns must be exactly {COLUMNS}, got {list(df.columns)}"]
    if df.isna().any().any():
        problems.append("dataset contains missing values")
    if df["interaction_id"].duplicated().any():
        problems.append("duplicate interaction_id values")

    for col in ("selected_difficulty", "previous_difficulty", "next_selected_difficulty"):
        bad = set(df[col].unique()) - set(DIFFICULTY_LEVELS)
        if bad:
            problems.append(f"{col} has invalid levels: {sorted(bad)}")
    if not df["recent_accuracy"].between(0, 1).all():
        problems.append("recent_accuracy must be within [0, 1]")
    for col in ("mistake_count", "hint_count", "prior_interaction_count"):
        if (df[col] < 0).any():
            problems.append(f"{col} must be non-negative")
    if (df["response_time_seconds"] <= 0).any():
        problems.append("response_time_seconds must be positive")
    if not set(df[TARGET].unique()) <= {0, 1}:
        problems.append(f"{TARGET} must be 0/1")

    if not problems:
        cur = df["selected_difficulty"].map(DIFFICULTY_ORDINAL)
        nxt = df["next_selected_difficulty"].map(DIFFICULTY_ORDINAL)
        if ((nxt < cur).astype(int) != df[TARGET]).any():
            problems.append("target is not (next difficulty < current difficulty)")
        for learner_id, g in df.groupby("learner_id", sort=False):
            if (g["prior_interaction_count"].to_numpy() != range(len(g))).any():
                problems.append(f"{learner_id}: prior_interaction_count is not 0..n-1 in order")
                break
            if (g["selected_difficulty"].to_numpy()[1:] != g["next_selected_difficulty"].to_numpy()[:-1]).any():
                problems.append(f"{learner_id}: next_selected_difficulty does not chain to the next row")
                break
    return problems
