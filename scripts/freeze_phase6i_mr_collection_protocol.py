#!/usr/bin/env python3
"""Freeze the Phase 6I-MR diagnostic branch and full-collection protocol."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/phase6i_mr_live_utility_revision.json"
COMMAND_PATH = ROOT / "configs/phase6i_mr_command_manifest.json"
PROTOCOL_PATH = ROOT / "docs/reports/phase6i_mr_preregistered_protocol.md"
OUT = ROOT / "outputs/phase6i_mr/frozen/collection_protocol.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def require_pass(path: Path, *, r10_locked: bool = True) -> dict:
    payload = load_json(path)
    if payload.get("status") != "PASS":
        raise RuntimeError(f"required PASS artifact is incomplete: {path}")
    if r10_locked and payload.get("r10_accessed") is not False:
        raise RuntimeError(f"R10 access invariant failed: {path}")
    if payload.get("r11_accessed") is not False:
        raise RuntimeError(f"R11 access invariant failed: {path}")
    return payload


def main() -> None:
    config = load_json(CONFIG_PATH)
    commands = load_json(COMMAND_PATH)
    instance_audit_path = ROOT / config["instance_suite"]["integrity_audit"]
    instance_audit = load_json(instance_audit_path)
    if instance_audit.get("status") != "PASS":
        raise RuntimeError("Phase 6I-MR instance integrity is not PASS")
    if commands.get("phase6i_config_sha256") != digest(CONFIG_PATH):
        raise RuntimeError("command manifest/config hash mismatch")

    pilot_path = ROOT / config["pilot"]["output_root"] / "pilot_integrity.json"
    analysis_path = (
        ROOT / config["pilot"]["output_root"] / "analysis/analysis_integrity.json"
    )
    continuation_path = ROOT / "outputs/phase6i_mr/continuation/continuation_integrity.json"
    branch_path = ROOT / "outputs/phase6i_mr/continuation/continuation_branch_decision.json"
    pilot = require_pass(pilot_path)
    analysis = require_pass(analysis_path)
    continuation = require_pass(continuation_path)
    branch = load_json(branch_path)
    if not all([
        branch.get("status") == "PASS",
        branch.get("branch") in {"IMMEDIATE_TARGET_VALID", "TARGET_MISMATCH"},
        branch.get("u2h_activated") == (
            branch.get("branch") == "TARGET_MISMATCH"
        ),
        branch.get("r10_accessed") is False,
        branch.get("r11_accessed") is False,
        branch.get("config_sha256") == digest(CONFIG_PATH),
        branch.get("pilot_integrity_sha256") == digest(pilot_path),
    ]):
        raise RuntimeError("continuation branch is not internally consistent")
    if not all([
        pilot.get("complete_runs") == 9,
        pilot.get("complete_states") == 54,
        pilot.get("complete_forced_actions") == 216,
        continuation["checks"].get("exactly_27_states"),
        continuation["checks"].get("exactly_216_action_continuations"),
        continuation["checks"].get("all_feasible"),
        continuation["checks"].get("all_replays_exact"),
    ]):
        raise RuntimeError("pilot or continuation cardinality check failed")

    progress_centers = [round((index + 0.5) / 30.0, 12) for index in range(30)]
    payload = {
        "schema": "phase6i-mr-collection-protocol-freeze-v1.2",
        "status": "FROZEN_BEFORE_FORMAL_R09_COLLECTION",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "starting_git_commit": config["starting_git_commit"],
        "input_hashes": {
            "phase6i_config_sha256": digest(CONFIG_PATH),
            "command_manifest_sha256": digest(COMMAND_PATH),
            "preregistered_protocol_sha256": digest(PROTOCOL_PATH),
            "instance_manifest_sha256": digest(
                ROOT / config["instance_suite"]["manifest"]
            ),
            "instance_integrity_audit_sha256": digest(instance_audit_path),
            "pilot_integrity_sha256": digest(pilot_path),
            "pilot_analysis_integrity_sha256": digest(analysis_path),
            "continuation_integrity_sha256": digest(continuation_path),
            "continuation_branch_decision_sha256": digest(branch_path),
            "phase6f_checkpoint_sha256": config["locked_inputs"][
                "phase6f_checkpoint_sha256"
            ],
            "phase6h_policy_sha256": config["locked_inputs"][
                "phase6h_policy_sha256"
            ],
        },
        "diagnostic_branch": {
            "branch": branch["branch"],
            "u2h_activated": branch["u2h_activated"],
            "median_within_state_spearman": branch[
                "median_within_state_spearman"
            ],
            "top1_agreement": branch["top1_agreement"],
            "branch_thresholds": branch["thresholds"],
        },
        "collection": {
            "r09_split": "LIVE_REV_FIT",
            "r09_seeds": config["seeds"]["R09_FULL_TRAJECTORY"],
            "r10_split": "LIVE_REV_SELECT",
            "r10_seeds": config["seeds"]["R10_FULL_TRAJECTORY"],
            "instances_per_split": 18,
            "trajectories_per_instance": 3,
            "sample_states_per_trajectory": 30,
            "samples_per_stage": 10,
            "normalized_progress_centers": progress_centers,
            "expected_trajectories_per_split": 54,
            "expected_states_per_split": 1620,
            "maximum_broad_actions_per_split": 6480,
            "wall_clock_seconds_per_operation": config["search"][
                "wall_clock_seconds_per_operation"
            ],
            "labels_generated_after_source_trajectory": True,
        },
        "candidate_contract": {
            "roles": config["forced_candidate_roles"]["roles_in_order"],
            "repair_decoder_trials_per_target": config["search"][
                "repair_decoder_trials_per_target"
            ],
            "bank_semantics": config["search"]["candidate_bank_contract"],
            "candidate_bank_change_allowed": False,
        },
        "audit_allocation": {
            "audit_trajectory_seed_rule": "minimum preregistered seed for the split",
            "top_eight": {
                "states_per_split": 18,
                "one_per_instance": True,
                "stage_targets": [0.15, 0.50, 0.85],
                "assignment": "sort scale, CF_level, cell_replicate, instance_id; index modulo three",
            },
            "true_full_bank": {
                "states_per_split": 9,
                "cell_replicate": "C02",
                "stage_target_by_cf": {"CF1": 0.15, "CF2": 0.50, "CF3": 0.85},
            },
            "diagnostic_only": True,
        },
        "access_control": {
            "r09_collection_authorized": True,
            "r10_collection_authorized": False,
            "r10_unlock": "only after every training candidate, calibration, threshold grid, and lexicographic rule is frozen",
            "r11_content_authorized": False,
            "r10_accessed": False,
            "r11_accessed": False,
        },
    }
    if OUT.exists():
        existing = load_json(OUT)
        comparable = dict(existing)
        comparable.pop("frozen_at_utc", None)
        expected = dict(payload)
        expected.pop("frozen_at_utc", None)
        if comparable != expected:
            raise RuntimeError("existing collection protocol freeze differs")
        print(json.dumps(existing, indent=2, sort_keys=True))
        return
    atomic_json(payload, OUT)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
