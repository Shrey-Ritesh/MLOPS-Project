"""Simulated traffic generator for Phase 3 drift and degradation experiments.

Replays held-out test learners (split_by_learner, 240 learners) in per-learner
interaction order through ``InferenceService.predict_and_log(source="simulated")``
so predictions come from the active MLflow ``production`` model (or local fallback)
and ``label_joiner`` can join delayed ``next_interaction_lower_difficulty`` labels.

Note on shifted mode:
    ``mode="shifted"`` injects ``response_time_seconds * 0.45`` and ``hint_count + 2``
    without altering difficulty choices, so it produces data drift without artificial
    performance drops (as expected when difficulty transitions stay unchanged).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import random
import sys
import time
from typing import Any
import uuid

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import requests

from src.track_b.modeling import load_dataset, split_by_learner
from src.track_b.schema import FEATURE_COLUMNS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TrafficGenerator")

BASELINE_DATASET = Path("data/synthetic/learner_interactions_shifted_v1.csv")
SHIFT_SOURCE_DATASET = Path("data/synthetic/learner_interactions_v1.csv")


def load_traffic_dataset(mode: str, *, test_only: bool = True) -> pd.DataFrame:
    """Load held-out test learners for baseline (control) or shifted synthetic traffic."""
    if mode == "baseline":
        csv_path = BASELINE_DATASET
    elif mode == "shifted":
        csv_path = SHIFT_SOURCE_DATASET
    else:
        raise ValueError(f"Unknown mode: {mode}. Must be 'baseline' or 'shifted'.")

    if not csv_path.exists():
        raise FileNotFoundError(f"Traffic dataset not found at {csv_path}")

    logger.info("Loading %s dataset from %s (test_only=%s)", mode, csv_path, test_only)
    if test_only:
        df, meta = load_dataset(csv_path)
        _, df = split_by_learner(df, meta)
    else:
        df = pd.read_csv(csv_path)

    df = df.copy()
    if mode == "shifted":
        df["response_time_seconds"] = (df["response_time_seconds"] * 0.45).round(2)
        df["hint_count"] = df["hint_count"] + 2
    return df


def iter_learner_ordered_rows(
    df: pd.DataFrame,
    *,
    limit: int | None = None,
    seed: int = 20261128,
    min_steps_per_learner: int = 15,
) -> list[tuple[str, int, pd.Series]]:
    """Order rows so each learner's interactions replay sequentially (learner_seq = 0, 1, 2...).

    Interleaves across shuffled held-out test learners so windows stay diverse in topic
    while giving label_joiner consecutive (learner_seq, learner_seq + 1) pairs.
    """
    if df.empty:
        return []

    sort_cols = ["learner_id", "prior_interaction_count"] if "prior_interaction_count" in df.columns else ["learner_id"]
    ordered = df.sort_values(sort_cols, kind="mergesort")

    by_learner: dict[str, list[pd.Series]] = {}
    for learner_id, group in ordered.groupby("learner_id", sort=False):
        by_learner[str(learner_id)] = [row for _, row in group.iterrows()]

    learner_ids = list(by_learner.keys())
    random.Random(seed).shuffle(learner_ids)

    n_target_learners = (
        len(learner_ids)
        if limit is None or limit <= 0
        else min(len(learner_ids), max(1, limit // max(1, min_steps_per_learner)))
    )
    active = learner_ids[:n_target_learners]
    reserve = learner_ids[n_target_learners:]
    positions = {lid: 0 for lid in learner_ids}

    result: list[tuple[str, int, pd.Series]] = []
    while active and (limit is None or limit <= 0 or len(result) < limit):
        next_active: list[str] = []
        for lid in active:
            if limit is not None and limit > 0 and len(result) >= limit:
                break
            pos = positions[lid]
            rows_for_learner = by_learner[lid]
            if pos < len(rows_for_learner):
                result.append((lid, pos, rows_for_learner[pos]))
                positions[lid] = pos + 1
            if positions[lid] < len(rows_for_learner):
                next_active.append(lid)
            elif reserve:
                next_active.append(reserve.pop(0))
        active = next_active

    return result


def format_prediction_payload(row: pd.Series, mode: str, *, learner_seq: int | None = None) -> dict[str, Any]:
    """Format single interaction conforming to Section 4.1 prediction record contract."""
    feature_dict = {
        col: (row[col].item() if hasattr(row[col], "item") else row[col])
        for col in FEATURE_COLUMNS
        if col in row
    }
    seq = int(row.get("prior_interaction_count", 0)) if learner_seq is None else int(learner_seq)

    return {
        "prediction_id": f"pred_sim_{uuid.uuid4().hex[:12]}",
        "request_id": f"req_sim_{uuid.uuid4().hex[:10]}",
        "occurred_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "schema_version": "1.1.0",
        "source": "simulated",
        "simulation_mode": mode,
        "learner_id": str(row["learner_id"]),
        "learner_seq": seq,
        "question_id": f"q_sim_{row.get('interaction_id', seq)}",
        "topic": str(row["topic"]),
        "features": feature_dict,
        "selected_difficulty": str(row["selected_difficulty"]),
    }


def _load_inference_service() -> Any:
    try:
        from src.learner.inference_hook import InferenceService
    except ImportError as exc:
        raise RuntimeError(
            "src.learner.inference_hook is required for live simulated replay; merge dev/PR #26 first."
        ) from exc
    return InferenceService.from_env()


def stream_traffic(
    mode: str,
    target_url: str | None = None,
    output_log: Path | None = None,
    limit: int | None = 300,
    rate_per_second: float = 0.0,
    use_inference_service: bool | None = None,
    inference_service: Any | None = None,
    seed: int = 20261128,
) -> int:
    """Replay test learners in sequence through InferenceService, HTTP endpoint, or JSONL."""
    if use_inference_service is None:
        use_inference_service = (target_url is None and output_log is None) or inference_service is not None

    df = load_traffic_dataset(mode, test_only=True)
    replay_rows = iter_learner_ordered_rows(df, limit=limit, seed=seed)
    logger.info(
        "Starting traffic simulation: mode=%s, total_rows=%d, use_inference_service=%s",
        mode,
        len(replay_rows),
        use_inference_service,
    )

    service = inference_service if inference_service is not None else (_load_inference_service() if use_inference_service else None)
    run_tag = uuid.uuid4().hex[:6]

    sent_count = 0
    start_time = time.time()

    if output_log:
        output_log.parent.mkdir(parents=True, exist_ok=True)
        log_file = output_log.open("a", encoding="utf-8")
    else:
        log_file = None

    try:
        for idx, (learner_id, seq, row) in enumerate(replay_rows):
            payload = format_prediction_payload(row, mode, learner_seq=seq)
            scoped_learner = f"sim_{mode}_{run_tag}_{learner_id}"

            if service is not None:
                service.predict_and_log(
                    learner_id=scoped_learner,
                    learner_seq=seq,
                    question_id=payload["question_id"],
                    features=payload["features"],
                    source="simulated",
                    request_id=payload["request_id"],
                )

            if target_url:
                try:
                    resp = requests.post(target_url, json=payload, timeout=2.0)
                    resp.raise_for_status()
                except Exception as e:
                    logger.warning("Failed sending record %d to %s: %s", idx, target_url, e)

            if log_file:
                log_file.write(json.dumps(payload) + "\n")
                log_file.flush()

            sent_count += 1

            if rate_per_second > 0:
                time.sleep(1.0 / rate_per_second)

            if (idx + 1) % 50 == 0 or (idx + 1) == len(replay_rows):
                logger.info("Progress: %d / %d records sent (mode=%s)", idx + 1, len(replay_rows), mode)

    finally:
        if log_file:
            log_file.close()

    elapsed = time.time() - start_time
    logger.info("Completed simulation: %d records processed in %.2fs (mode=%s)", sent_count, elapsed, mode)
    return sent_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["baseline", "shifted"], default="baseline",
                        help="Traffic mode: baseline (control) or shifted (drifted)")
    parser.add_argument("--target-url", type=str, default=None,
                        help="Optional HTTP endpoint URL")
    parser.add_argument("--output-log", type=Path, default=None,
                        help="Optional JSONL file path to append simulated traffic records")
    parser.add_argument("--use-inference-service", action="store_true",
                        help="Call InferenceService.from_env().predict_and_log(source='simulated') (default when no URL/log)")
    parser.add_argument("--limit", type=int, default=300,
                        help="Maximum records to send (default: 300; >=300 per 2m window avoids small-window noise)")
    parser.add_argument("--rate", type=float, default=0.0,
                        help="Sending rate in requests per second (e.g. 2.5 for 300 rows/2m; 0 for max speed)")
    parser.add_argument("--seed", type=int, default=20261128,
                        help="Random seed for shuffling held-out test learners")
    args = parser.parse_args()

    stream_traffic(
        mode=args.mode,
        target_url=args.target_url,
        output_log=args.output_log,
        limit=args.limit,
        rate_per_second=args.rate,
        use_inference_service=args.use_inference_service or None,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
