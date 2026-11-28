"""Model contract: the feature list, feature types/limits, and the labelling rule for one model.

Loaded from ``configs/monitoring/contracts/<id>.yaml``. The active contract is chosen with
``MONITORING_CONTRACT`` (a path); the default is the Track B difficulty model. Swapping the model
task (e.g. to the chapter-mastery model) is a contract change, not a code change.
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONTRACT = "configs/monitoring/contracts/track_b_difficulty_v1.yaml"
FEATURE_TYPES = ("categorical", "numeric", "integer")
LABEL_RULES = ("next_interaction_lower_difficulty", "next_outcome_same_topic")
OUTCOMES = ("correct", "incorrect", "partial", "uncertain")


class ContractError(ValueError):
    """The contract file is invalid."""


@dataclass(frozen=True)
class Feature:
    name: str
    type: str
    levels: tuple[str, ...] | None = None
    vocabulary: frozenset[str] | None = None
    min: float | None = None
    max: float | None = None

    def check(self, value: Any) -> str | None:
        """Return None if valid, 'unseen' for a value outside a vocabulary, else 'out_of_range'."""
        if self.type == "categorical":
            if self.levels is not None:
                return None if value in self.levels else "out_of_range"
            if self.vocabulary is not None:
                return None if str(value or "unknown") in self.vocabulary else "unseen"
            return None if isinstance(value, str) and value else "out_of_range"
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return "out_of_range"
        if self.type == "integer" and float(value) != int(value):
            return "out_of_range"
        if (self.min is not None and value < self.min) or (self.max is not None and value > self.max):
            return "out_of_range"
        return None


@dataclass(frozen=True)
class Contract:
    contract_id: str
    status: str
    features: tuple[Feature, ...]
    label_name: str
    label_rule: str
    label_params: dict[str, Any] = field(default_factory=dict)
    expiry_hours: float = 72.0
    path: str = ""

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.features)

    def feature(self, name: str) -> Feature:
        return next(f for f in self.features if f.name == name)


def _load_vocabulary(path: str) -> frozenset[str]:
    with open(path, encoding="utf-8", newline="") as fh:
        return frozenset(row["topic"].strip() for row in csv.DictReader(fh) if row.get("topic"))


def load_contract(path: str | Path | None = None) -> Contract:
    path = str(path or os.environ.get("MONITORING_CONTRACT", DEFAULT_CONTRACT))
    return _load_cached(path)


@lru_cache(maxsize=8)
def _load_cached(path: str) -> Contract:
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    try:
        features = []
        for name, spec in raw["features"].items():
            ftype = spec["type"]
            if ftype not in FEATURE_TYPES:
                raise ContractError(f"{name}: unknown type {ftype!r}")
            features.append(Feature(
                name=name, type=ftype,
                levels=tuple(spec["levels"]) if "levels" in spec else None,
                vocabulary=_load_vocabulary(spec["vocabulary_file"]) if "vocabulary_file" in spec else None,
                min=spec.get("min"), max=spec.get("max"),
            ))
        label = raw["label"]
        contract = Contract(
            contract_id=str(raw["contract_id"]), status=str(raw.get("status", "draft")),
            features=tuple(features), label_name=str(label["name"]), label_rule=str(label["rule"]),
            label_params=dict(label.get("params") or {}), expiry_hours=float(label.get("expiry_hours", 72)),
            path=path,
        )
    except KeyError as exc:
        raise ContractError(f"{path}: missing key {exc}") from exc
    _validate(contract)
    return contract


def _validate(c: Contract) -> None:
    if not c.features:
        raise ContractError("contract has no features")
    if len(set(c.feature_names)) != len(c.feature_names):
        raise ContractError("duplicate feature names")
    if c.label_rule not in LABEL_RULES:
        raise ContractError(f"unknown label rule {c.label_rule!r}; known: {LABEL_RULES}")
    if c.label_rule == "next_interaction_lower_difficulty":
        feat = c.label_params.get("difficulty_feature")
        if feat not in c.feature_names or not c.label_params.get("levels"):
            raise ContractError("next_interaction_lower_difficulty needs difficulty_feature (a feature) and levels")
    if c.label_rule == "next_outcome_same_topic":
        if "topic" not in c.feature_names:
            raise ContractError("next_outcome_same_topic needs a 'topic' feature")
        listed = [o for key in ("positive", "negative", "excluded") for o in c.label_params.get(key, [])]
        if sorted(listed) != sorted(OUTCOMES):
            raise ContractError(f"positive/negative/excluded must cover each of {OUTCOMES} exactly once")
