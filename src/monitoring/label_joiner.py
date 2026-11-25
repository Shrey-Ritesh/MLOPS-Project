"""Join delayed ground truth onto logged predictions.

Target (``src/track_b/generate_learners.py``): ``switch_to_lower_difficulty = 1`` when the learner's
NEXT interaction has a lower ``selected_difficulty`` (Easy < Medium < Hard).

For each pending prediction P, the next row N is the one with the same ``learner_key`` and
``source`` and ``learner_seq = P.learner_seq + 1``. Rows with no successor after
``expiry_hours`` are marked ``expired`` and stay out of performance metrics.

Only ``label_observed``, ``label_status`` and ``label_joined_at`` are ever updated.

    python -m src.monitoring.label_joiner [--db PATH] [--expiry-hours 72]
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.monitoring import db
from src.monitoring.app_logging import log_event
from src.monitoring.schema import DIFFICULTY_LEVELS

DEFAULT_EXPIRY_HOURS = 72
_ORDINAL = {name: i for i, name in enumerate(DIFFICULTY_LEVELS)}


@dataclass(frozen=True)
class JoinResult:
    joined: int
    expired: int
    still_pending: int
    invalid_difficulty: int


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _difficulty(features_json: str) -> int | None:
    try:
        return _ORDINAL.get(json.loads(features_json).get("selected_difficulty"))
    except (ValueError, AttributeError):
        return None


def join_labels(
    db_path: str | Path,
    now: datetime | None = None,
    expiry_hours: float = DEFAULT_EXPIRY_HOURS,
) -> JoinResult:
    now = now or datetime.now(timezone.utc)
    now_iso = _iso(now)
    cutoff_iso = _iso(now - timedelta(hours=expiry_hours))

    with db.session(db_path) as conn:
        db.ensure_schema(conn)
        pairs = conn.execute(
            """
            SELECT p.prediction_id, p.features, n.features
            FROM predictions AS p
            JOIN predictions AS n
              ON n.learner_key = p.learner_key
             AND n.source = p.source
             AND n.learner_seq = p.learner_seq + 1
            WHERE p.label_status = 'pending'
            """
        ).fetchall()

        updates, invalid = [], 0
        for prediction_id, current_features, next_features in pairs:
            current, nxt = _difficulty(current_features), _difficulty(next_features)
            if current is None or nxt is None:
                invalid += 1  # stays pending; expires later and is counted by monitoring
                continue
            updates.append((int(nxt < current), now_iso, prediction_id))

        conn.executemany(
            """
            UPDATE predictions
               SET label_observed = ?, label_status = 'joined', label_joined_at = ?
             WHERE prediction_id = ? AND label_status = 'pending'
            """,
            updates,
        )
        expired = conn.execute(
            """
            UPDATE predictions
               SET label_status = 'expired', label_joined_at = ?
             WHERE label_status = 'pending' AND occurred_at < ?
            """,
            (now_iso, cutoff_iso),
        ).rowcount
        still_pending = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE label_status = 'pending'"
        ).fetchone()[0]

    result = JoinResult(len(updates), expired, still_pending, invalid)
    log_event("labels_joined", service="monitoring", count=result.joined,
              reason=f"expired={result.expired} pending={result.still_pending} invalid={result.invalid_difficulty}")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", default=db.db_path_from_env())
    parser.add_argument("--expiry-hours", type=float, default=DEFAULT_EXPIRY_HOURS)
    args = parser.parse_args(argv)
    print(json.dumps(asdict(join_labels(args.db, expiry_hours=args.expiry_hours))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
