#!/usr/bin/env python3
"""Freeze and validate the immutable Phase 6L L1 preregistration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/phase6l_legacy_score_decoupling_v1.json"
PROTOCOL = ROOT / "docs/reports/phase6l_legacy_score_decoupling_preregistered_protocol.md"
OUTPUT = ROOT / "outputs/phase6l_legacy_score_decoupling_v1/preregistration/preregistration.json"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    config = json.loads(CONFIG.read_text())
    if config["status"] != "PREREGISTERED_BEFORE_PHASE6L_DATASET_DERIVATION_OR_OPTIMIZER_STEP":
        raise RuntimeError("invalid Phase 6L preregistration status")
    locked = config["locked_inputs"]
    required = {
        "phase6j_grouped_labels": "phase6j_grouped_labels_sha256",
        "phase6j_raw_seed_labels": "phase6j_raw_seed_labels_sha256",
        "phase6j_collection_integrity": "phase6j_collection_integrity_sha256",
        "phase6j_tensor_cache_integrity": "phase6j_tensor_cache_integrity_sha256",
        "phase6j_tensor_manifest": "phase6j_tensor_manifest_sha256",
        "phase6f_base_checkpoint": "phase6f_base_checkpoint_sha256",
    }
    for path_key, hash_key in required.items():
        if digest(ROOT / locked[path_key]) != locked[hash_key]:
            raise RuntimeError(f"locked input changed: {path_key}")
    for forbidden in (
        ROOT / "outputs/phase6j_caur/r13_selection/access_ledger.json",
        ROOT / "outputs/phase6j_caur/r14_holdout/access_ledger.json",
    ):
        if forbidden.exists():
            raise RuntimeError(f"forbidden holdout access ledger exists: {forbidden}")
    sources = [
        CONFIG,
        PROTOCOL,
        ROOT / locked["phase6l_dependency_map"],
        ROOT / locked["phase6j_config"],
        ROOT / locked["phase6j_training_protocol"],
        ROOT / "rcias_clgri/analysis/phase6l_legacy_score.py",
        Path(__file__),
    ]
    record = {
        "schema": "phase6l-preregistration-freeze-v1",
        "status": "FROZEN_BEFORE_PHASE6L_DATASET_DERIVATION_OR_OPTIMIZER_STEP",
        "git_commit_before_l1_files": config["l0_commit"],
        "source_hashes": {
            str(path.resolve().relative_to(ROOT)): digest(path) for path in sources
        },
        "locked_input_hashes": {
            locked[path_key]: locked[hash_key] for path_key, hash_key in required.items()
        },
        "primary_model": config["primary_model"]["model_id"],
        "primary_seeds": config["training"]["seeds"],
        "optional_fallback_model": None,
        "r12_bundle_count": 1,
        "r13_accessed": False,
        "r14_accessed": False,
        "worktree_at_freeze": subprocess.check_output(
            ["git", "status", "--short"], cwd=ROOT, text=True
        ).splitlines(),
    }
    text = json.dumps(record, indent=2, sort_keys=True) + "\n"
    if OUTPUT.exists() and OUTPUT.read_text() != text:
        raise RuntimeError("refusing to replace an existing Phase 6L preregistration")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if not OUTPUT.exists():
        OUTPUT.write_text(text)
    print(json.dumps({
        "status": record["status"],
        "primary_model": record["primary_model"],
        "source_hashes": len(record["source_hashes"]),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
