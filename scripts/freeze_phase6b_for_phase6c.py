#!/usr/bin/env python3
"""Verify and freeze the Phase 6A/6B evidence used by Phase 6C."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P6B = ROOT / "outputs/phase6b"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    completion = json.loads((P6B / "audit/completion_gate.json").read_text())
    recommendation = json.loads((P6B / "diagnostics/phase6c_recommendation.json").read_text())
    phase6a = json.loads((P6B / "environment/phase6a_freeze_record.json").read_text())
    train = json.loads((P6B / "audit/train_distribution_freeze.json").read_text())
    if not completion["PHASE6B_COMPLETE"]:
        raise RuntimeError("Phase 6B completion gate is not frozen TRUE")
    expected = {"COUNTERFACTUAL_STATE_COUNT": 13500, "PRIMARY_COUNTERFACTUAL_ARM_COUNT": 185643,
                "COUNTERFACTUAL_ARM_COUNT": 224021, "PHASE6C_RECOMMENDATION": "SCALE_WITH_REVISED_ARM_DESIGN"}
    if any(recommendation[key] != value for key, value in expected.items()):
        raise RuntimeError("Phase 6B recommendation does not match the Phase 6C contract")
    raw_paths = [
        P6B / "trajectory_reservoir/pilot_state_manifest.parquet",
        P6B / "counterfactual/counterfactual_arm_results.parquet",
        P6B / "counterfactual/counterfactual_target_rows.parquet",
        P6B / "marginal_target/marginal_swap_results.parquet",
    ]
    summary_paths = sorted((P6B / "summaries").glob("*.csv"))
    record = {
        "schema": "phase6b-freeze-for-phase6c-v1",
        "phase6a_freeze_hash": phase6a["freeze_hash"],
        "train_distribution_freeze_hash": train["freeze_hash"],
        "phase6b_report_sha256": digest(ROOT / "docs/reports/phase6b_counterfactual_data_pilot_report.md"),
        "phase6b_raw_sha256": {str(path.relative_to(ROOT)): digest(path) for path in raw_paths},
        "phase6b_summary_sha256": {str(path.relative_to(ROOT)): digest(path) for path in summary_paths},
        "phase6b_completion_gate_sha256": digest(P6B / "audit/completion_gate.json"),
        "phase6b_recommendation_sha256": digest(P6B / "diagnostics/phase6c_recommendation.json"),
        "frozen_conclusion": recommendation,
    }
    canonical = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    record["freeze_hash"] = hashlib.sha256(canonical).hexdigest()
    output = ROOT / "outputs/phase6c/environment/phase6b_freeze_record.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print("PHASE6A_PHASE6B_FROZEN", record["freeze_hash"])


if __name__ == "__main__":
    main()
