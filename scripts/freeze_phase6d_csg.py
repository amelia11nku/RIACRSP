#!/usr/bin/env python3
"""Freeze CSG-1.0 only after every Phase 6D acceptance gate passes."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import subprocess
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/phase6d"
SCHEMA = ROOT / "configs/csg_v1_schema.json"
PHASE6C_HASH = "695307ac6193ecbbeb0f73e81a94ea20672ffd47aa9419bb37610be9d3161437"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    value.update(path.read_bytes())
    return value.hexdigest()


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def write_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    schema = json.loads(SCHEMA.read_text())
    validation = json.loads((OUT / "validation/csg_validation_summary.json").read_text())
    information = json.loads((OUT / "information_audit/information_audit_summary.json").read_text())
    examples = json.loads((OUT / "examples/examples_summary.json").read_text())
    complexity = json.loads((OUT / "profiling/complexity_model.json").read_text())
    repository = json.loads((OUT / "audit/repository_audit.json").read_text())
    regression = json.loads((OUT / "audit/regression_summary.json").read_text())
    permutation = pd.read_csv(OUT / "information_audit/permutation_invariance_summary.csv")
    temporal = pd.read_csv(OUT / "validation/temporal_consistency_summary.csv")
    checks_frame = pd.read_csv(OUT / "validation/structural_check_summary.csv")
    checks = dict(zip(checks_frame.check, checks_frame.passed.astype(bool)))
    relation_checks = (
        "precedence_exact", "eligibility_exact", "support_exact",
        "product_chain_exact", "island_chain_exact", "w_chain_exact", "f_chain_exact",
        "reconfiguration_nodes_exact", "w_event_nodes_exact", "f_event_nodes_exact",
        "w_synchronization_exact", "f_synchronization_exact",
        "reconfiguration_synchronization_exact", "temporal_semantics",
        "causal_subgraph_is_dag",
    )
    gates = {
        "CSG_SCHEMA_DEFINED": schema["version"] == "CSG-1.0" and len(schema["node_types"]) == 8 and len(schema["edge_types"]) == 20,
        "CSG_BUILDER_IMPLEMENTED": (ROOT / "rcias_clgri/csg/builder.py").exists(),
        "CSG_BUILD_DETERMINISTIC": bool(validation["deterministic_build_passed"]),
        "PHASE6C_STATE_RECONSTRUCTION_REUSED": bool(validation["phase6c_state_reconstruction_reused"]),
        "NO_FORBIDDEN_FUTURE_INFORMATION": bool(information["no_forbidden_future_information"]),
        "ALL_REQUIRED_INFORMATION_REPRESENTED": bool(information["all_required_information_represented"]),
        "ALL_TARGET_OPERATIONS_MAPPABLE": validation["mapped_target_operation_count"] == 4_297_000,
        "SAME_STATE_ACTION_GRAPH_INVARIANT": bool(validation["same_state_action_graph_invariance_passed"]),
        "PRECEDENCE_RELATIONS_VALID": checks.get("precedence_exact", False),
        "ELIGIBILITY_RELATIONS_VALID": checks.get("eligibility_exact", False),
        "CAPABILITY_RELATIONS_VALID": checks.get("support_exact", False),
        "PRODUCT_CHAINS_VALID": checks.get("product_chain_exact", False),
        "ISLAND_CHAINS_VALID": checks.get("island_chain_exact", False),
        "W_CHAINS_VALID": checks.get("w_chain_exact", False),
        "F_CHAINS_VALID": checks.get("f_chain_exact", False),
        "RECONFIG_EVENTS_VALID": checks.get("reconfiguration_nodes_exact", False),
        "W_EVENTS_VALID": checks.get("w_event_nodes_exact", False),
        "F_EVENTS_VALID": checks.get("f_event_nodes_exact", False),
        "SYNCHRONIZATION_RELATIONS_VALID": all(checks.get(name, False) for name in (
            "w_synchronization_exact", "f_synchronization_exact", "reconfiguration_synchronization_exact",
        )),
        "TEMPORAL_CONSISTENCY_PASSED": checks.get("temporal_semantics", False) and int(temporal.negative_gap_count.sum()) == 0,
        "CAUSAL_SUBGRAPH_VALID": checks.get("causal_subgraph_is_dag", False),
        "IDENTIFIER_PERMUTATION_TEST_PASSED": bool(permutation.passed.all()),
        "S_M_L_PROFILING_COMPLETE": complexity["profiled_state_count"] >= 30 and bool(complexity["all_validation_passed"]),
        "HISTORICAL_GRAPH_BEHAVIOR_UNCHANGED": bool(repository["checks"]["frozen_source_boundaries_unchanged"]),
        "FULL_TEST_SUITE_PASSED": bool(regression["all_passed"]),
    }
    if not all(gates.values()):
        raise RuntimeError({key: value for key, value in gates.items() if not value})

    csg_sources = sorted((ROOT / "rcias_clgri/csg").glob("*.py"))
    frozen_at = datetime.now(timezone.utc).isoformat()
    freeze = {
        "schema": "phase6d-csg-schema-freeze-v1",
        "status": "FROZEN",
        "frozen_at_utc": frozen_at,
        "csg_schema_version": "CSG-1.0",
        "csg_schema_sha256": digest(SCHEMA),
        "phase6c_dataset_freeze_hash": PHASE6C_HASH,
        "node_type_count": len(schema["node_types"]),
        "edge_type_count": len(schema["edge_types"]),
        "node_types": list(schema["node_types"]),
        "edge_types": [edge["key"] for edge in schema["edge_types"]],
        "csg_source_sha256": {str(path.relative_to(ROOT)): digest(path) for path in csg_sources},
        "formal_definition_sha256": digest(ROOT / "docs/reports/phase6d_csg_formal_definition.md"),
        "validation_report_sha256": digest(ROOT / "docs/reports/phase6d_csg_validation_report.md"),
        "validation_state_count": validation["completed_state_count"],
        "validation_graph_hash_policy": "SHA-256 of canonical typed graph payload including state identity and lookup mapping",
        "acceptance_gates": gates,
        "all_acceptance_gates_passed": True,
        "phase6e_recommendation": "PROCEED_TO_SUPERVISED_NI_MODEL",
    }
    write_json(freeze, OUT / "schema_freeze_record.json")
    write_json({
        **gates,
        "schema": "phase6d-completion-gate-v1",
        "PHASE6D_COMPLETE": True,
        "CSG_1_0_FROZEN": True,
    }, OUT / "audit/completion_gate.json")
    write_json({
        "schema": "phase6d-phase6e-recommendation-v1",
        "PHASE6D_COMPLETE": True,
        "CSG_SCHEMA_VERSION": "CSG-1.0",
        "PHASE6E_RECOMMENDATION": "PROCEED_TO_SUPERVISED_NI_MODEL",
        "recommended_target": "set-level target-action improvement classification/ranking",
        "repair_operator": "transport_aware",
        "destroy_size_fraction": 0.15,
        "neural_model_implemented_in_phase6d": False,
    }, OUT / "diagnostics/phase6e_recommendation.json")
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        capture_output=True, check=True,
    ).stdout.strip()
    write_json({
        "schema": "phase6d-environment-v1",
        "recorded_at_utc": frozen_at,
        "platform": platform.platform(),
        "python": sys.version,
        "git_head": git_head,
        "packages": {
            name: package_version(name)
            for name in ("numpy", "pandas", "pyarrow", "matplotlib", "pytest")
        },
        "phase6c_dataset_freeze_hash": PHASE6C_HASH,
        "validation_workers": 8,
        "selection_seed": validation["selection_seed"],
    }, OUT / "environment/environment.json")
    print("PHASE6D_COMPLETE CSG-1.0_FROZEN PHASE6E=PROCEED_TO_SUPERVISED_NI_MODEL")


if __name__ == "__main__":
    main()
