#!/usr/bin/env python3
"""Freeze the Phase 6C production configuration after Gates A and B."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/phase6c"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    audits = {
        name: json.loads((OUT / f"gates/{name}/gate_audit.json").read_text())
        for name in ("gate_a", "gate_b")
    }
    expected_states = {"gate_a": 1000, "gate_b": 5000}
    for name, audit in audits.items():
        if audit["counts"]["states"] != expected_states[name] or not all(audit["integrity"].values()):
            raise RuntimeError(f"{name} did not pass")
        if audit["production_config_change_justified"]:
            raise RuntimeError(f"{name} requests a production-config change")
    record = {
        "schema": "phase6c-production-freeze-v1",
        "label_generation_version": "phase6c-v1",
        "config_sha256": digest(ROOT / "configs/phase6c_counterfactual.json"),
        "counterfactual_evaluator_sha256": digest(ROOT / "rcias_clgri/search/counterfactual.py"),
        "revised_arm_design_sha256": digest(ROOT / "rcias_clgri/search/phase6c.py"),
        "reconstruction_contract_sha256": digest(ROOT / "rcias_clgri/data/phase6c.py"),
        "field_contract_sha256": digest(ROOT / "rcias_clgri/data/phase6c_contract.py"),
        "shard_io_sha256": digest(ROOT / "rcias_clgri/data/phase6c_io.py"),
        "reservoir_runner_sha256": digest(ROOT / "scripts/run_phase6c_reservoir.py"),
        "dataset_runner_sha256": digest(ROOT / "scripts/run_phase6c_dataset.py"),
        "gate_a_audit_sha256": digest(OUT / "gates/gate_a/gate_audit.json"),
        "gate_b_audit_sha256": digest(OUT / "gates/gate_b/gate_audit.json"),
        "state_targets": {"TRAIN": 60000, "TRAIN_VALIDATION": 20000, "TRAIN_INTERNAL_HOLDOUT": 20000},
        "repair_seed_count": 3,
        "repair_operator": "transport_aware",
        "destroy_fraction": 0.15,
        "requested_arm_rule_count": 24,
        "post_gate_data_quality_corrections": [
            "fixed_iteration_trajectory_budget_for_checksum_reproducibility",
            "all-state_stage_reservoir_plus_rare-bottleneck_supplements_for_minimum_coverage"
        ],
        "counterfactual_label_protocol_changed_after_gates": False,
        "status": "FROZEN",
    }
    record["freeze_hash"] = hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    path = OUT / "environment/production_config_freeze.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print("PHASE6C_PRODUCTION_CONFIG_FROZEN", record["freeze_hash"])


if __name__ == "__main__":
    main()
