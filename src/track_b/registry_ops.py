"""Approval-gated MLflow registry promotion, rejection, and rollback operations (Person 3).

Governance rules enforced in code:
1. Moving the ``production`` alias to a challenger is REFUSED unless:
   - the Promotion Gate v2 decision is ``PROMOTE_CHALLENGER_PENDING_APPROVAL``, AND
   - a recorded human approval JSON exists with ``approval_id``, ``approved_by``,
     ``approved_at``, ``challenger_version``, and ``rationale``.
2. Upon promotion, the previous champion version is tagged ``status=archived`` and
   given the ``archived`` alias so rollback is one command away.
3. Upon gate failure, ``reject`` sets the ``rejected_candidate`` alias on the challenger
   and never touches ``production``.
4. ``rollback`` moves the ``production`` alias back to the target (or previous ``archived``)
   version and records the rollback event in the alias history log; ``ProgressModel`` picks
   up the alias change without rebuilding or restarting the app.

CLI usage::

    python -m src.track_b.registry_ops promote --decision reports/track_b/promotion_decision_v2.json --approval docs/governance/approval.json
    python -m src.track_b.registry_ops reject  --decision reports/track_b/promotion_decision_v2.json
    python -m src.track_b.registry_ops rollback --to-version 2 --reason "post-deploy latency spike" --operator "person3"
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

from mlflow import MlflowClient

from src.track_b import tracking
from src.track_b.modeling import file_sha256

DEFAULT_HISTORY_OUT = Path("reports/track_b/registry_alias_history.json")
REQUIRED_APPROVAL_FIELDS = ("approval_id", "approved_by", "approved_at", "challenger_version", "rationale")


class GovernanceApprovalError(ValueError):
    """Raised when a promotion attempt lacks a passing gate decision or valid human approval."""


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _append_history(history_path: Path, event: dict[str, Any]) -> list[dict[str, Any]]:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    if history_path.is_file():
        data = json.loads(history_path.read_text(encoding="utf-8"))
        events = data.get("alias_history", []) if isinstance(data, dict) else list(data)
    else:
        events = []
    events.append(event)
    payload = {
        "registered_model": event.get("model_name", tracking.REGISTERED_MODEL_NAME),
        "updated_at_utc": _utc_now(),
        "alias_history": events,
    }
    history_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return events


def validate_approval_record(
    decision_path: Path,
    approval_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate both the gate v2 decision and human approval record before moving production."""
    if not decision_path.is_file():
        raise GovernanceApprovalError(f"Gate decision file not found: {decision_path}")
    if not approval_path.is_file():
        raise GovernanceApprovalError(f"Human approval record not found: {approval_path}")

    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    approval = json.loads(approval_path.read_text(encoding="utf-8"))

    if decision.get("decision") != "PROMOTE_CHALLENGER_PENDING_APPROVAL":
        raise GovernanceApprovalError(
            f"Cannot promote: gate decision is {decision.get('decision')!r}, "
            "expected 'PROMOTE_CHALLENGER_PENDING_APPROVAL'."
        )

    missing = [f for f in REQUIRED_APPROVAL_FIELDS if not str(approval.get(f, "")).strip()]
    if missing:
        raise GovernanceApprovalError(f"Approval record missing required fields: {missing}")

    if str(approval.get("approved", True)).strip().lower() in {"false", "0", "no"}:
        raise GovernanceApprovalError("Approval record explicitly denied promotion (approved=false).")

    decision_ver = str(decision.get("challenger", {}).get("model_version", ""))
    approval_ver = str(approval["challenger_version"])
    if decision_ver and approval_ver != decision_ver:
        raise GovernanceApprovalError(
            f"Approval challenger_version ({approval_ver}) does not match gate decision ({decision_ver})."
        )

    expected_sha = approval.get("gate_decision_sha256")
    if expected_sha:
        actual_sha = file_sha256(decision_path)
        if expected_sha != actual_sha:
            raise GovernanceApprovalError(
                f"Approval gate_decision_sha256 mismatch: expected {expected_sha}, got {actual_sha}."
            )

    return decision, approval


def promote_challenger(
    *,
    decision_path: Path,
    approval_path: Path,
    model_name: str = tracking.REGISTERED_MODEL_NAME,
    tracking_uri: str | None = None,
    history_path: Path = DEFAULT_HISTORY_OUT,
) -> dict[str, Any]:
    """Move ``production`` alias to the approved challenger and archive the previous champion."""
    decision, approval = validate_approval_record(decision_path, approval_path)
    uri = tracking_uri or tracking.configure()
    client = MlflowClient(tracking_uri=uri)

    challenger_version = str(approval["challenger_version"])
    previous_version: str | None = None
    try:
        prev_mv = client.get_model_version_by_alias(model_name, "production")
        previous_version = str(prev_mv.version)
    except Exception:
        previous_version = None

    # Archive previous champion first (if different from challenger)
    if previous_version and previous_version != challenger_version:
        client.set_model_version_tag(model_name, previous_version, "status", "archived")
        client.set_model_version_tag(model_name, previous_version, "archived_at_utc", _utc_now())
        client.set_registered_model_alias(model_name, "archived", previous_version)

    # Promote challenger to production
    client.set_registered_model_alias(model_name, "production", challenger_version)
    client.set_model_version_tag(model_name, challenger_version, "status", "production")
    client.set_model_version_tag(model_name, challenger_version, "gate_id", str(decision["gate_id"]))
    client.set_model_version_tag(model_name, challenger_version, "gate_decision", str(decision["decision"]))
    client.set_model_version_tag(model_name, challenger_version, "approved_by", str(approval["approved_by"]))
    client.set_model_version_tag(model_name, challenger_version, "approval_id", str(approval["approval_id"]))

    event = {
        "at": _utc_now(),
        "action": "promote",
        "model_name": model_name,
        "alias": "production",
        "from_version": previous_version,
        "to_version": challenger_version,
        "gate": decision["gate_id"],
        "gate_decision": decision["decision"],
        "approved_by": approval["approved_by"],
        "approval_id": approval["approval_id"],
        "rationale": approval["rationale"],
    }
    _append_history(history_path, event)
    return event


def reject_challenger(
    *,
    decision_path: Path,
    model_name: str = tracking.REGISTERED_MODEL_NAME,
    tracking_uri: str | None = None,
    history_path: Path = DEFAULT_HISTORY_OUT,
) -> dict[str, Any]:
    """Tag a rejected challenger with ``rejected_candidate`` alias without modifying ``production``."""
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    challenger_version = str(decision["challenger"]["model_version"])
    uri = tracking_uri or tracking.configure()
    client = MlflowClient(tracking_uri=uri)

    client.set_registered_model_alias(model_name, "rejected_candidate", challenger_version)
    client.set_model_version_tag(model_name, challenger_version, "status", "rejected_candidate")
    client.set_model_version_tag(model_name, challenger_version, "gate_id", str(decision["gate_id"]))
    client.set_model_version_tag(model_name, challenger_version, "gate_decision", str(decision["decision"]))

    event = {
        "at": _utc_now(),
        "action": "reject",
        "model_name": model_name,
        "alias": "rejected_candidate",
        "from_version": None,
        "to_version": challenger_version,
        "gate": decision["gate_id"],
        "gate_decision": decision["decision"],
        "failed_criteria": decision.get("failed_criteria", []),
        "approved_by": None,
        "approval_id": None,
    }
    _append_history(history_path, event)
    return event


def rollback_production(
    *,
    to_version: str | None = None,
    reason: str,
    operator: str,
    model_name: str = tracking.REGISTERED_MODEL_NAME,
    tracking_uri: str | None = None,
    history_path: Path = DEFAULT_HISTORY_OUT,
) -> dict[str, Any]:
    """Move ``production`` alias back to ``to_version`` (or the ``archived`` alias) without rebuilding."""
    if not reason.strip() or not operator.strip():
        raise GovernanceApprovalError("Rollback requires non-empty --reason and --operator.")
    uri = tracking_uri or tracking.configure()
    client = MlflowClient(tracking_uri=uri)

    current_mv = client.get_model_version_by_alias(model_name, "production")
    from_version = str(current_mv.version)

    if to_version is None:
        archived_mv = client.get_model_version_by_alias(model_name, "archived")
        target_version = str(archived_mv.version)
    else:
        target_version = str(to_version)

    client.set_registered_model_alias(model_name, "production", target_version)
    client.set_model_version_tag(model_name, target_version, "status", "production")
    client.set_model_version_tag(model_name, from_version, "status", "rolled_back")
    client.set_model_version_tag(model_name, from_version, "rollback_reason", reason.strip())

    event = {
        "at": _utc_now(),
        "action": "rollback",
        "model_name": model_name,
        "alias": "production",
        "from_version": from_version,
        "to_version": target_version,
        "operator": operator.strip(),
        "reason": reason.strip(),
    }
    _append_history(history_path, event)
    return event


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-name", default=tracking.REGISTERED_MODEL_NAME)
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--history-out", type=Path, default=DEFAULT_HISTORY_OUT)
    sub = parser.add_subparsers(dest="command", required=True)

    p_prom = sub.add_parser("promote", help="Promote challenger after passing gate v2 + human approval")
    p_prom.add_argument("--decision", type=Path, required=True)
    p_prom.add_argument("--approval", type=Path, required=True)

    p_rej = sub.add_parser("reject", help="Tag failed challenger as rejected_candidate")
    p_rej.add_argument("--decision", type=Path, required=True)

    p_rb = sub.add_parser("rollback", help="Roll back production alias to a previous version")
    p_rb.add_argument("--to-version", default=None, help="Target model version (defaults to 'archived' alias)")
    p_rb.add_argument("--reason", required=True)
    p_rb.add_argument("--operator", required=True)

    args = parser.parse_args(argv)
    if args.command == "promote":
        ev = promote_challenger(
            decision_path=args.decision,
            approval_path=args.approval,
            model_name=args.model_name,
            tracking_uri=args.tracking_uri,
            history_path=args.history_out,
        )
    elif args.command == "reject":
        ev = reject_challenger(
            decision_path=args.decision,
            model_name=args.model_name,
            tracking_uri=args.tracking_uri,
            history_path=args.history_out,
        )
    else:
        ev = rollback_production(
            to_version=args.to_version,
            reason=args.reason,
            operator=args.operator,
            model_name=args.model_name,
            tracking_uri=args.tracking_uri,
            history_path=args.history_out,
        )
    print(json.dumps(ev, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
