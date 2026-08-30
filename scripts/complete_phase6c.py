#!/usr/bin/env python3
"""Validate the final Phase 6C completion gate."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/phase6c"


def main():
    states = pd.read_csv(OUT / "manifests/state_manifest.csv")
    integrity = json.loads((OUT / "audit/counterfactual_integrity.json").read_text())
    freeze = json.loads((OUT / "audit/dataset_freeze_record.json").read_text())
    environment = json.loads((OUT / "environment/environment.json").read_text())
    repository = json.loads((OUT / "audit/repository_audit.json").read_text())
    validation = environment["final_validation"]
    figure_names = (
        "state_coverage_by_split", "structural_cell_coverage", "arm_family_positive_yield",
        "single_seed_vs_three_seed_rank_stability", "improvement_probability_distribution",
        "within_state_rank_diversity", "local_swap_near_neighbor_value", "predictability_by_split",
        "predictability_by_scale_cf", "predictability_by_ri_ti", "operation_pair_signal",
        "compute_storage_profile",
    )
    gates = {
        "PHASE6A_PHASE6B_FROZEN": (OUT / "environment/phase6b_freeze_record.json").exists(),
        "PRODUCTION_CONFIG_FROZEN": (OUT / "environment/production_config_freeze.json").exists(),
        "DISTINCT_STATE_COUNT_EXACT": len(states) == states.state_id.nunique() == 100000,
        "SPLIT_COUNTS_EXACT": states.training_split.value_counts().to_dict() == {
            "TRAIN": 60000, "TRAIN_VALIDATION": 20000, "TRAIN_INTERNAL_HOLDOUT": 20000,
        },
        "ALL_INTEGRITY_GATES_PASSED": integrity["COUNTERFACTUAL_INTEGRITY_PASSED"],
        "DATASET_FROZEN": freeze["status"] == "FROZEN",
        "NI_DATASET_CONTRACT_COMPLETE": (ROOT / "docs/reports/phase6c_ni_dataset_contract.md").exists(),
        "PHASE6C_REPORT_COMPLETE": (ROOT / "docs/reports/phase6c_scaled_counterfactual_dataset_report.md").exists(),
        "REQUIRED_FIGURES_PRESENT": all(
            (OUT / "figures" / f"Fig{index:02d}_{name}.png").exists() and
            (OUT / "figures" / f"Fig{index:02d}_{name}.pdf").exists()
            for index, name in enumerate(figure_names, 1)
        ),
        "FULL_TEST_SUITE_PASSED": validation["full_test_suite"].startswith("PASS_"),
        "FROZEN_BENCHMARKS_UNCHANGED": validation["canonical_regeneration"] == "PASS_130_OF_130" and
                                       validation["cb1_checksums"] == "PASS_113_OF_113" and
                                       validation["train_checksums"] == "PASS_405_OF_405",
        "REPOSITORY_AUDITED": repository["audit_status"] == "PASS",
    }
    gates = {name: bool(value) for name, value in gates.items()}
    gates["PHASE6C_COMPLETE"] = all(gates.values())
    path = OUT / "audit/completion_gate.json"
    path.write_text(json.dumps(gates, indent=2, sort_keys=True) + "\n")
    if not gates["PHASE6C_COMPLETE"]:
        raise RuntimeError(gates)
    print("PHASE6C_COMPLETE = TRUE")


if __name__ == "__main__":
    main()
