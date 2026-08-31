#!/usr/bin/env python3
"""Freeze Phase 6A–6E evidence and the pre-registered Phase 6F protocol."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "phase6f"
CONFIG = ROOT / "configs" / "phase6f_revision.json"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def canonical_hash(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    freeze_path = OUT / "environment" / "freeze_record.json"
    state_freeze = OUT / "revision_holdout" / "state_freeze.json"
    if freeze_path.exists() and state_freeze.exists():
        existing = load(freeze_path)
        if existing["phase6f_config_sha256"] != digest(CONFIG):
            raise RuntimeError("Phase 6F config changed after revision-holdout state freeze")
        print("PHASE6F_ENVIRONMENT_FREEZE_VERIFIED")
        return

    phase_gates = {
        phase: load(ROOT / "outputs" / phase / audit)
        for phase, audit in {
            "phase6a": "diagnostics/completion_gate.json",
            "phase6b": "audit/completion_gate.json",
            "phase6c": "audit/completion_gate.json",
            "phase6d": "audit/completion_gate.json",
            "phase6e": "audit/completion_gate.json",
        }.items()
    }
    phase6e_conclusions_path = ROOT / "outputs" / "phase6e" / "audit" / "scientific_conclusions.json"
    phase6e_conclusions = load(phase6e_conclusions_path)
    phase6e_repository_path = ROOT / "outputs" / "phase6e" / "audit" / "repository_audit.json"
    phase6e_repository = load(phase6e_repository_path)
    artifact_manifest_path = ROOT / "outputs" / "phase6e" / "audit" / "artifact_manifest.json"
    artifact_manifest = load(artifact_manifest_path)
    artifact_hashes_exact = all(
        (ROOT / relative).exists()
        and digest(ROOT / relative) == record["sha256"]
        and (ROOT / relative).stat().st_size == record["bytes"]
        for relative, record in artifact_manifest["artifacts"].items()
    )

    csg_freeze_path = ROOT / "outputs" / "phase6d" / "schema_freeze_record.json"
    csg_freeze = load(csg_freeze_path)
    csg_sources_exact = csg_freeze["csg_source_sha256"] == {
        str(path.relative_to(ROOT)): digest(path)
        for path in sorted((ROOT / "rcias_clgri" / "csg").glob("*.py"))
    }
    checkpoint_manifest_path = (
        ROOT / "outputs" / "phase6e" / "training" / "final_seeds" / "checkpoint_manifest.json"
    )
    checkpoint_manifest = load(checkpoint_manifest_path)
    checkpoints_exact = all(
        Path(record["checkpoint_path"]).exists()
        and digest(Path(record["checkpoint_path"])) == record["checkpoint_sha256"]
        for record in checkpoint_manifest["checkpoints"]
    )

    config = load(CONFIG)
    checks = {
        "phase6a_complete": bool(phase_gates["phase6a"]["PHASE6A_COMPLETE"]),
        "phase6b_complete": bool(phase_gates["phase6b"]["PHASE6B_COMPLETE"]),
        "phase6c_complete": bool(phase_gates["phase6c"]["PHASE6C_COMPLETE"]),
        "phase6d_complete": bool(phase_gates["phase6d"]["PHASE6D_COMPLETE"]),
        "phase6e_complete": bool(phase_gates["phase6e"]["PHASE6E_COMPLETE"]),
        "phase6e_recommendation_is_revise_model": (
            phase6e_conclusions["conclusions"]["PHASE6F_RECOMMENDATION"] == "REVISE_MODEL"
        ),
        "phase6e_live_integration_not_approved": not bool(
            phase_gates["phase6e"]["PHASE6F_LIVE_INTEGRATION_APPROVED"]
        ),
        "phase6e_repository_audit_passed": phase6e_repository["audit_status"] == "PASS",
        "phase6e_artifact_hashes_exact": artifact_hashes_exact,
        "phase6e_teacher_checkpoints_exact": checkpoints_exact,
        "csg_schema_hash_exact": (
            csg_freeze["csg_schema_sha256"]
            == digest(ROOT / "configs" / "csg_v1_schema.json")
        ),
        "csg_sources_exact": csg_sources_exact,
        "phase6c_freeze_hash_exact": (
            config["frozen_boundaries"]["phase6c_dataset_freeze_hash"]
            == load(ROOT / "outputs" / "phase6c" / "audit" / "dataset_freeze_record.json")[
                "freeze_hash"
            ]
        ),
        "old_internal_holdout_forbidden_for_selection": (
            config["frozen_boundaries"]["phase6e_internal_holdout_selection_use"] == "FORBIDDEN"
        ),
        "live_solver_scope_forbidden": "live ALNS integration" in config["forbidden"],
    }
    if not all(checks.values()):
        raise RuntimeError({key: value for key, value in checks.items() if not value})

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True
    ).stdout.strip()
    worktree = subprocess.run(
        ["git", "status", "--short"], cwd=ROOT, text=True, capture_output=True, check=True
    ).stdout.splitlines()
    payload = {
        "schema": "phase6f-environment-freeze-v1",
        "status": "FROZEN_BEFORE_R06_GENERATION",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "worktree_status_at_freeze": worktree,
        "phase6f_config_sha256": digest(CONFIG),
        "phase6e_completion_gate_sha256": digest(
            ROOT / "outputs" / "phase6e" / "audit" / "completion_gate.json"
        ),
        "phase6e_scientific_conclusions_sha256": digest(phase6e_conclusions_path),
        "phase6e_repository_audit_sha256": digest(phase6e_repository_path),
        "phase6e_artifact_manifest_sha256": digest(artifact_manifest_path),
        "phase6e_checkpoint_manifest_sha256": digest(checkpoint_manifest_path),
        "phase6d_schema_freeze_sha256": digest(csg_freeze_path),
        "checks": checks,
    }
    payload["freeze_hash"] = canonical_hash(payload)
    freeze_path.parent.mkdir(parents=True, exist_ok=True)
    freeze_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    protocol = {
        "schema": "phase6f-preregistered-protocol-v1",
        "status": "FROZEN",
        "phase6f_config_sha256": payload["phase6f_config_sha256"],
        "environment_freeze_hash": payload["freeze_hash"],
        "quality_gates": config["success_gates"],
        "objective_candidates": list(config["objective_candidates"]),
        "compact_model_candidates": list(config["compact_model_candidates"]),
        "deployment_seed_rule": config["final_training"]["deployment_seed_rule"],
        "selection_splits": config["development_protocol"],
        "revision_holdout_label_access": "FORBIDDEN_UNTIL_MODEL_CONFIG_FREEZE",
    }
    protocol["protocol_hash"] = canonical_hash(protocol)
    protocol_path = OUT / "audit" / "preregistered_protocol.json"
    protocol_path.parent.mkdir(parents=True, exist_ok=True)
    protocol_path.write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("PHASE6F_ENVIRONMENT_FROZEN", payload["freeze_hash"])


if __name__ == "__main__":
    main()
