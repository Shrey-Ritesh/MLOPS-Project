"""Join delayed ground truth onto logged predictions, using the active contract's label rule.

Rules (``configs/monitoring/contracts/*.yaml``, ``label.rule``):

``next_interaction_lower_difficulty`` (Track B difficulty model)
    For pending prediction P, the next row N has the same ``learner_key`` and ``source`` and
    ``learner_seq = P.learner_seq + 1``; label = 1 if N's difficulty is lower than P's.

``next_outcome_same_topic`` (chapter-mastery model)
    The first ``attempt_outcomes`` row with the same ``learner_key``, ``source`` and ``topic`` at
    or after P; label = 1 for a positive outcome (``correct``), 0 for a negative one; outcomes
    listed as excluded (``partial``, ``uncertain``) set ``label_status = 'excluded'``.

Only predictions made under the active contract are labelled. Rows with no ground truth after
``expiry_hours`` are marked ``expired``. Only ``label_observed``, ``label_status`` and
``label_joined_at`` are ever updated.

    python -m src.monitoring.label_joiner [--db PATH]
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.monitoring import db
from src.monitoring.app_logging import log_event
from src.monitoring.contract import Contract, load_contract


@dataclass(frozen=True)
class JoinResult:
    joined: int
    expired: int
    excluded: int
    still_pending: int
    invalid: int


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _difficulty_pairs(conn, contract: Contract) -> tuple[list[tuple], int]:
    feature = contract.label_params["difficulty_feature"]
    ordinal = {name: i for i, name in enumerate(contract.label_params["levels"])}
    pairs = conn.execute(
        """
        SELECT p.prediction_id, p.features, n.features
        FROM predictions AS p
        JOIN predictions AS n
          ON n.learner_key = p.learner_key AND n.source = p.source
         AND n.learner_seq = p.learner_seq + 1 AND n.contract_id = p.contract_id
        WHERE p.label_status = 'pending' AND p.contract_id = ?
        """,
        (contract.contract_id,),
    ).fetchall()

    def level(features_json: str) -> int | None:
        try:
            return ordinal.get(json.loads(features_json).get(feature))
        except (ValueError, AttributeError):
            return None

    updates, invalid = [], 0
    for prediction_id, current_json, next_json in pairs:
        current, nxt = level(current_json), level(next_json)
        if current is None or nxt is None:
            invalid += 1  # stays pending; expires later and is counted by monitoring
            continue
        updates.append((int(nxt < current), "joined", prediction_id))
    return updates, invalid


def _outcome_pairs(conn, contract: Contract) -> tuple[list[tuple], int]:
    positive = set(contract.label_params.get("positive", []))
    negative = set(contract.label_params.get("negative", []))
    rows = conn.execute(
        """
        SELECT p.prediction_id,
               (SELECT o.outcome FROM attempt_outcomes AS o
                 WHERE o.learner_key = p.learner_key AND o.source = p.source AND o.topic = p.topic
                   AND o.occurred_at >= p.occurred_at
                 ORDER BY o.occurred_at, o.outcome_id LIMIT 1) AS outcome
        FROM predictions AS p
        WHERE p.label_status = 'pending' AND p.contract_id = ?
        """,
        (contract.contract_id,),
    ).fetchall()
    updates = []
    for prediction_id, outcome in rows:
        if outcome is None:
            continue
        if outcome in positive:
            updates.append((1, "joined", prediction_id))
        elif outcome in negative:
            updates.append((0, "joined", prediction_id))
        else:
            updates.append((None, "excluded", prediction_id))
    return updates, 0


def join_labels(db_path: str | Path, now: datetime | None = None, contract: Contract | None = None,
                expiry_hours: float | None = None) -> JoinResult:
    contract = contract or load_contract()
    now = now or datetime.now(timezone.utc)
    now_iso = _iso(now)
    expiry = contract.expiry_hours if expiry_hours is None else expiry_hours
    cutoff_iso = _iso(now - timedelta(hours=expiry))

    with db.session(db_path) as conn:
        db.ensure_schema(conn)
        if contract.label_rule == "next_interaction_lower_difficulty":
            updates, invalid = _difficulty_pairs(conn, contract)
        else:
            updates, invalid = _outcome_pairs(conn, contract)
        conn.executemany(
            """
            UPDATE predictions SET label_observed = ?, label_status = ?, label_joined_at = ?
             WHERE prediction_id = ? AND label_status = 'pending'
            """,
            [(label, status, now_iso, pid) for label, status, pid in updates],
        )
        expired = conn.execute(
            """
            UPDATE predictions SET label_status = 'expired', label_joined_at = ?
             WHERE label_status = 'pending' AND occurred_at < ?
            """,
            (now_iso, cutoff_iso),
        ).rowcount
        still_pending = conn.execute("SELECT COUNT(*) FROM predictions WHERE label_status = 'pending'").fetchone()[0]

    joined = sum(1 for _, status, _ in updates if status == "joined")
    excluded = sum(1 for _, status, _ in updates if status == "excluded")
    result = JoinResult(joined, expired, excluded, still_pending, invalid)
    log_event("labels_joined", service="monitoring", count=joined,
              reason=f"rule={contract.label_rule} expired={expired} excluded={excluded} "
                     f"pending={still_pending} invalid={invalid}")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", default=db.db_path_from_env())
    parser.add_argument("--contract", help="contract YAML (default: MONITORING_CONTRACT or Track B)")
    args = parser.parse_args(argv)
    print(json.dumps(asdict(join_labels(args.db, contract=load_contract(args.contract)))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
