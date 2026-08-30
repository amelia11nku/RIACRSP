#!/usr/bin/env python3
"""Audit frozen inputs, required outputs, and temporary artifacts for Phase 6C."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/phase6c"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    frozen = json.loads((OUT / "environment/phase6b_freeze_record.json").read_text())
    expected_hashes = {
        **frozen["phase6b_raw_sha256"],
        **frozen["phase6b_summary_sha256"],
        "docs/reports/phase6b_counterfactual_data_pilot_report.md": frozen["phase6b_report_sha256"],
        "outputs/phase6b/audit/completion_gate.json": frozen["phase6b_completion_gate_sha256"],
        "outputs/phase6b/diagnostics/phase6c_recommendation.json": frozen["phase6b_recommendation_sha256"],
    }
    mismatches = {}
    for relative, expected in expected_hashes.items():
        path = ROOT / relative
        actual = digest(path) if path.exists() else None
        if actual != expected:
            mismatches[relative] = {"expected": expected, "actual": actual}
    required = (
        "outputs/phase6c/audit/counterfactual_integrity.json",
        "outputs/phase6c/audit/dataset_freeze_record.json",
        "outputs/phase6c/diagnostics/phase6d_recommendation.json",
        "outputs/phase6c/manifests/generation_manifest.json",
        "outputs/phase6c/manifests/shard_manifest.csv",
        "docs/reports/phase6c_ni_dataset_contract.md",
        "docs/reports/phase6c_scaled_counterfactual_dataset_report.md",
    )
    missing = [relative for relative in required if not (ROOT / relative).exists()]
    partial_files = sorted(
        str(path.relative_to(ROOT)) for path in OUT.rglob("*.partial.*")
    )
    cache_directories = sorted(
        str(path.relative_to(ROOT)) for path in ROOT.rglob("__pycache__")
        if ".git" not in path.parts
    )
    payload = {
        "schema": "phase6c-repository-audit-v1",
        "phase6a_phase6b_frozen_evidence_unchanged": not mismatches,
        "frozen_evidence_mismatches": mismatches,
        "required_phase6c_outputs_present": not missing,
        "missing_required_outputs": missing,
        "partial_files_absent": not partial_files,
        "partial_files": partial_files,
        "cache_directories_absent": not cache_directories,
        "cache_directories": cache_directories,
    }
    payload["audit_status"] = "PASS" if all((
        payload["phase6a_phase6b_frozen_evidence_unchanged"],
        payload["required_phase6c_outputs_present"],
        payload["partial_files_absent"],
        payload["cache_directories_absent"],
    )) else "FAIL"
    path = OUT / "audit/repository_audit.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if payload["audit_status"] != "PASS":
        raise RuntimeError(payload)
    print("PHASE6C_REPOSITORY_AUDIT_PASS")


if __name__ == "__main__":
    main()
