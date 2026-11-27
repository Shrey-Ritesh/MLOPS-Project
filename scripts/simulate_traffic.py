"""Simulated traffic generator for Phase 3 drift and degradation experiments.

Replays synthetic learner interaction sequences through the real inference path
(either via HTTP POST or writing directly to the prediction log).
All generated records are strictly tagged with `source="simulated"` conforming
to Section 4.1 contract.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
import time
import uuid

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import requests

from src.track_b.schema import FEATURE_COLUMNS, TARGET

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TrafficGenerator")


def load_traffic_dataset(mode: str) -> pd.DataFrame:
    """Load baseline or shifted synthetic traffic."""
    if mode == "baseline":
        csv_path = Path("data/synthetic/learner_interactions_v1.csv")
    elif mode == "shifted":
        csv_path = Path("data/synthetic/learner_interactions_shifted_v1.csv")
    else:
        raise ValueError(f"Unknown mode: {mode}. Must be 'baseline' or 'shifted'.")

    if not csv_path.exists():
        raise FileNotFoundError(f"Traffic dataset not found at {csv_path}")

    logger.info("Loading %s dataset from %s", mode, csv_path)
    return pd.read_csv(csv_path)


def format_prediction_payload(row: pd.Series, mode: str) -> dict:
    """Format single interaction conforming to Section 4.1 prediction record contract."""
    feature_dict = {col: row[col] for col in FEATURE_COLUMNS if col in row}
    
    return {
        "prediction_id": f"pred_sim_{uuid.uuid4().hex[:12]}",
        "request_id": f"req_sim_{uuid.uuid4().hex[:8]}",
        "occurred_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "schema_version": "1.0.0",
        "source": "simulated",
        "simulation_mode": mode,
        "learner_id": str(row["learner_id"]),
        "question_id": f"q_sim_{row.get('interaction_id', '0')}",
        "topic": str(row["topic"]),
        "features": feature_dict,
        "selected_difficulty": str(row["selected_difficulty"]),
        "ground_truth_label": int(row[TARGET]) if TARGET in row else None,
    }


def stream_traffic(
    mode: str,
    target_url: str | None = None,
    output_log: Path | None = None,
    limit: int | None = None,
    rate_per_second: float = 0.0,
) -> int:
    """Stream or batch replay traffic to an HTTP endpoint or a JSONL file."""
    df = load_traffic_dataset(mode)
    if limit is not None and limit > 0:
        df = df.iloc[:limit]

    logger.info("Starting traffic simulation: mode=%s, total_rows=%d", mode, len(df))

    sent_count = 0
    start_time = time.time()

    if output_log:
        output_log.parent.mkdir(parents=True, exist_ok=True)
        log_file = output_log.open("a", encoding="utf-8")
    else:
        log_file = None

    try:
        for idx, (_, row) in enumerate(df.iterrows()):
            payload = format_prediction_payload(row, mode)

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
                delay = 1.0 / rate_per_second
                time.sleep(delay)

            if (idx + 1) % 50 == 0 or (idx + 1) == len(df):
                logger.info("Progress: %d / %d records sent (mode=%s)", idx + 1, len(df), mode)

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
                        help="HTTP endpoint URL (e.g. http://localhost:8000/predict)")
    parser.add_argument("--output-log", type=Path, default=None,
                        help="Optional JSONL file path to append simulated traffic records")
    parser.add_argument("--limit", type=int, default=100,
                        help="Maximum records to send (default: 100)")
    parser.add_argument("--rate", type=float, default=0.0,
                        help="Sending rate in requests per second (0 for max speed)")
    args = parser.parse_args()

    stream_traffic(
        mode=args.mode,
        target_url=args.target_url,
        output_log=args.output_log,
        limit=args.limit,
        rate_per_second=args.rate,
    )


if __name__ == "__main__":
    main()
