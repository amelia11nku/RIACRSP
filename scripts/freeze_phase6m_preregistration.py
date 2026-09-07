#!/usr/bin/env python3
"""Freeze the Phase 6M M1 protocol before any new optimizer step."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
STARTING_COMMIT = "6f99e2da4e31a809fc00a7b646076c5790643f68"
CONFIG = ROOT / "configs/phase6m_selective_confidence_v1.json"
REPORT = ROOT / "docs/reports/phase6m_selective_confidence_preregistered_protocol.md"
AUDIT = ROOT / "outputs/phase6m_selective_confidence_v1/audit"
OUT = ROOT / "outputs/phase6m_selective_confidence_v1/preregistration/preregistration.json"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    config = json.loads(CONFIG.read_text())
    require(config["status"] == "PREREGISTERED_BEFORE_PHASE6M_OUTER_OOF_OR_OPTIMIZER_STEP",
            "invalid Phase 6M preregistration status")
    require(config["starting_commit"] == STARTING_COMMIT, "starting commit changed")
    require(subprocess.run(
        ["git", "merge-base", "--is-ancestor", STARTING_COMMIT, "HEAD"],
        cwd=ROOT, check=False,
    ).returncode == 0, "M0 commit is not an ancestor of HEAD")
    failure = json.loads((AUDIT / "failure_attribution.json").read_text())
    require(failure["status"] == "M0_COMPLETE", "M0 audit is incomplete")
    require(failure["scientific_attribution"]["raw_ranker_invalidated"] is False,
            "ranker architecture cannot be inherited")
    require(not (ROOT / "outputs/phase6j_caur/r13_selection/access_ledger.json").exists(),
            "R13 access detected")
    require(not (ROOT / "outputs/phase6j_caur/r14_holdout/access_ledger.json").exists(),
            "R14 access detected")
    forbidden_outputs = [
        ROOT / "outputs/phase6m_selective_confidence_v1/training/oof_predictions.parquet",
        ROOT / "outputs/phase6m_selective_confidence_v1/quality/development_quality.json",
    ]
    require(not any(path.exists() for path in forbidden_outputs),
            "Phase 6M qualification output exists before preregistration")
    inputs = [
        CONFIG,
        REPORT,
        Path(__file__),
        AUDIT / "failure_attribution.json",
        AUDIT / "gate_rejection_waterfall.csv",
        AUDIT / "confidence_diagnostics.csv",
        AUDIT / "support_rejection_breakdown.csv",
        AUDIT / "candidate_selection_uncertainty.csv",
        ROOT / "outputs/phase6l_legacy_score_decoupling_v1/training/training_protocol.json",
        ROOT / "outputs/phase6l_legacy_score_decoupling_v1/training/oof_predictions.parquet",
        ROOT / "outputs/phase6l_legacy_score_decoupling_v1/data/r12_score_free_seed_labels.parquet",
        ROOT / "outputs/phase6l_legacy_score_decoupling_v1/data/r12_score_free_grouped_labels.parquet",
    ]
    payload = {
        "schema": "phase6m-selective-confidence-preregistration-v1",
        "status": "FROZEN_BEFORE_PHASE6M_OUTER_OOF_OR_OPTIMIZER_STEP",
        "primary_family": config["primary_family"],
        "starting_commit": STARTING_COMMIT,
        "input_sha256": {
            str(path.relative_to(ROOT)): digest(path) for path in inputs
        },
        "protected_phase6l_manifest": {
            "path": str((AUDIT / "protected_phase6l_evidence.json").relative_to(ROOT)),
            "sha256": digest(AUDIT / "protected_phase6l_evidence.json"),
        },
        "optimizer_steps_started": False,
        "outer_oof_generated": False,
        "teacher_ablation": "NOT_RUN",
        "gurobi_run": False,
        "r13_accessed": False,
        "r14_accessed": False,
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        require(OUT.read_text() == text, "refusing to replace frozen Phase 6M preregistration")
    else:
        temporary = OUT.with_suffix(".json.tmp")
        temporary.write_text(text)
        temporary.replace(OUT)
    print(json.dumps({
        "status": payload["status"], "primary_family": payload["primary_family"],
        "input_files": len(payload["input_sha256"]),
        "preregistration_sha256": digest(OUT),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
