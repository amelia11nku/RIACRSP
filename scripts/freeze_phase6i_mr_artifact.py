#!/usr/bin/env python3
"""Freeze the single Phase 6I-MR artifact and protocol before R11 access."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "outputs/phase6i_mr/frozen"
TRANSLATION_POLICY = FROZEN / "r10_translation_policy.json"
TRANSLATION_AUTHORIZATION = FROZEN / "r10_translation_authorization.json"
TRANSLATION_DECISION = ROOT / "outputs/phase6i_mr/r10_translation/translation_decision.json"
REPLAY = ROOT / "outputs/phase6i_mr/r10_translation/deterministic_replay_fixture.json"
SELECTION = ROOT / "outputs/phase6i_mr/r10_selection/selection_decision.json"
CONFIG = ROOT / "configs/phase6i_mr_live_utility_revision.json"
MANIFEST = (
    ROOT / "instances/controlled/RCIAS-CB1-TRAIN-LIVE-R09R11/manifests"
    / "phase6i_instance_manifest.csv"
)
INTEGRITY = MANIFEST.with_name("phase6i_integrity_audit.json")
ARTIFACT = FROZEN / "selected_artifact.json"
FREEZE_RECORD = FROZEN / "artifact_freeze.json"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def main() -> None:
    if ARTIFACT.exists() or FREEZE_RECORD.exists():
        if not (ARTIFACT.is_file() and FREEZE_RECORD.is_file()):
            raise RuntimeError("partial R11 artifact freeze exists")
        record = load_json(FREEZE_RECORD)
        if record.get("artifact_sha256") != digest(ARTIFACT):
            raise RuntimeError("existing R11 artifact freeze is invalid")
        print(f"PHASE6I_MR_ARTIFACT_ALREADY_FROZEN sha256={digest(ARTIFACT)}")
        return

    required = [
        TRANSLATION_POLICY, TRANSLATION_AUTHORIZATION, TRANSLATION_DECISION,
        REPLAY, SELECTION, CONFIG, MANIFEST, INTEGRITY,
        ROOT / "scripts/run_phase6i_mr_r11_validation.py",
        ROOT / "scripts/finalize_phase6i_mr.py",
        Path(__file__),
        ROOT / "rcias_clgri/search/csgni.py",
        ROOT / "rcias_clgri/search/alns.py",
        ROOT / "rcias_clgri/analysis/phase6h.py",
        ROOT / "rcias_clgri/ni/phase6i_live_inference.py",
        ROOT / "rcias_clgri/ni/phase6i_policy.py",
        ROOT / "rcias_clgri/data/phase6i_access.py",
    ]
    missing = [relative(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"cannot freeze; missing files: {missing}")

    policy = load_json(TRANSLATION_POLICY)
    authorization = load_json(TRANSLATION_AUTHORIZATION)
    decision = load_json(TRANSLATION_DECISION)
    config = load_json(CONFIG)
    integrity = load_json(INTEGRITY)
    checks = {
        "translation_passed": decision.get("status")
        == "PASS_AUTHORIZE_SELECTED_ARTIFACT_FREEZE",
        "translation_policy_matches": decision.get("translation_policy_sha256")
        == digest(TRANSLATION_POLICY),
        "translation_authorization_matches": decision.get(
            "translation_authorization_sha256"
        ) == digest(TRANSLATION_AUTHORIZATION),
        "selection_matches": decision.get("selection_decision_sha256")
        == digest(SELECTION),
        "all_translation_checks_passed": all(decision.get("checks", {}).values()),
        "r11_not_accessed": all(
            item.get("r11_accessed") is False
            for item in (policy, authorization, decision, load_json(SELECTION))
        ),
        "suite_integrity_passed": integrity.get("status") == "PASS"
        and all(integrity.get("checks", {}).values()),
    }
    if not all(checks.values()):
        raise RuntimeError({"artifact_freeze_preconditions": checks})

    code_paths = [path for path in required if path.suffix == ".py"]
    artifact = dict(policy)
    artifact.update({
        "schema": "phase6i-mr-selected-artifact-v1.2",
        "status": "FROZEN_BEFORE_R11",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "r11_unlock": True,
        "r11_accessed": False,
        "translation_evidence": {
            "decision_path": relative(TRANSLATION_DECISION),
            "decision_sha256": digest(TRANSLATION_DECISION),
            "directional_solver_improvement_observed": decision[
                "directional_solver_improvement_observed"
            ],
            "aggregate_revised_relative_to_alns": decision[
                "aggregate_revised_relative_to_alns"
            ],
        },
        "deterministic_replay_fixture": {
            "path": relative(REPLAY), "sha256": digest(REPLAY),
        },
        "training_and_manifest_hashes": {
            **policy["source_hashes"],
            "instance_manifest": digest(MANIFEST),
            "instance_integrity_audit": digest(INTEGRITY),
        },
        "accessed_split_ledger": {
            "R09": "ACCESSED_FOR_TRAINING_AND_CALIBRATION",
            "R10": "ACCESSED_ONCE_FOR_SELECTION_AND_TRANSLATION",
            "R11": "CONTENT_NOT_ACCESSED_BEFORE_THIS_FREEZE",
            "r10_refit_performed": False,
        },
        "r11_protocol": {
            "instances": 18,
            "seeds": config["r11_protocol"]["seeds"],
            "methods": config["r11_protocol"]["methods"],
            "iterative_budget_seconds": "2N",
            "execution": "SEQUENTIAL_ONE_WORKER_ONE_GPU_ONE_CPU_THREAD",
            "checkpoint_loading": "LOAD_ONCE_AND_REPORT_SEPARATELY_EXCLUDED_FROM_EACH_RUN_BUDGET",
            "forced_diagnostic_states_per_revised_run": 10,
            "forced_state_progress_targets": [
                (index + 0.5) / 10.0 for index in range(10)
            ],
            "forced_roles": ["TOP1", "TOP2", "ALNS_RELATED_FALLBACK", "DIVERSE"],
            "labels": "POST_TRAJECTORY_ONLY_NO_SEARCH_FEEDBACK",
            "paired_unit": "INSTANCE_MEAN_OVER_FIVE_MATCHED_SEEDS",
            "bootstrap_replicates": 10000,
            "bootstrap_seed": 688001,
            "wilcoxon": "EXACT_ONE_SIDED_GREATER_ON_18_INSTANCE_MEANS",
        },
        "frozen_gate_constants": {
            **config["promotion_gates"],
            "u0_material_pairwise_accuracy_gain": 0.01,
            "u0_material_ndcg_at_1_gain": 0.01,
            "target_relative_gaps": [0.05, 0.02, 0.01, 0.005],
        },
        "code_hashes": {relative(path): digest(path) for path in code_paths},
        "source_hashes": {
            **policy["source_hashes"],
            "config": digest(CONFIG),
            "command_manifest": digest(ROOT / "configs/phase6i_mr_command_manifest.json"),
            "translation_policy": digest(TRANSLATION_POLICY),
            "translation_authorization": digest(TRANSLATION_AUTHORIZATION),
            "translation_decision": digest(TRANSLATION_DECISION),
        },
    })
    atomic_json(artifact, ARTIFACT)
    artifact_hash = digest(ARTIFACT)
    record = {
        "schema": "phase6i-mr-artifact-freeze-v1",
        "status": "FROZEN_BEFORE_R11",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_path": relative(ARTIFACT),
        "artifact_sha256": artifact_hash,
        "r11_content_accessed": False,
        "preconditions": checks,
        "r11_execution_command": (
            "/home/liulei/miniconda3/envs/gnn311/bin/python "
            "scripts/run_phase6i_mr_r11_validation.py --device cuda"
        ),
    }
    atomic_json(record, FREEZE_RECORD)
    print(f"PHASE6I_MR_ARTIFACT_FROZEN sha256={artifact_hash}")


if __name__ == "__main__":
    main()
