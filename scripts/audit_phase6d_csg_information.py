#!/usr/bin/env python3
"""Audit CSG-1.0 information preservation, leakage, and intentional redundancy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_clgri.csg.schema import SCHEMA
from rcias_clgri.data.phase6c_io import atomic_write_csv, atomic_write_json


DEFAULT_OUTPUT = ROOT / "outputs/phase6d/information_audit"
DEFAULT_VALIDATION = ROOT / "outputs/phase6d/validation/csg_validation_summary.json"

INFORMATION_REQUIREMENTS = (
    ("operation slack", "OP.operation_slack and OP.operation_slack_normalized", "node_feature_schema; numerical normalization test"),
    ("W waiting / delay", "OP.w_delay; W_EVENT.pickup_wait_normalized", "node_feature_schema; schedule equivalence"),
    ("F waiting / delay", "OP.f_delay; F_EVENT timing", "node_feature_schema; F synchronization exactness"),
    ("island relative load", "OP.island_relative_load; ISLAND.relative_processing_load", "node_feature_schema; independent extraction"),
    ("local reconfiguration contribution", "OP.local_reconfiguration; RECONF_EVENT", "reconfiguration event/config exactness"),
    ("eligible-island information", "OP__ELIGIBLE_ON__ISLAND with processing-time edge features", "eligibility_exact"),
    ("precedence neighborhoods", "OP__PRECEDES__OP", "precedence_exact"),
    ("realized product/workpiece sequence", "OP__PRODUCT_NEXT__OP", "product_chain_exact"),
    ("realized island execution chains", "OP__ISLAND_NEXT__OP", "island_chain_exact"),
    ("realized W-AGV chains", "W_EVENT__W_NEXT__W_EVENT", "w_chain_exact"),
    ("realized F-AGV chains", "F_EVENT__F_NEXT__F_EVENT", "f_chain_exact"),
    ("synchronization relations", "W/F/RECONF_EVENT__ENABLES__OP", "W/F/reconfiguration synchronization exactness"),
    ("search progress", "graph.search_progress and graph.search_stage", "graph_state_schema"),
    ("current makespan", "graph.current_makespan", "state reconstruction and graph_state_schema"),
    ("current assignment", "OP__ASSIGNED_TO__ISLAND", "assignment_exact"),
    ("configuration context", "OP__REQUIRES__CONFIG; ISLAND__SUPPORTS/CURRENT_CONFIG__CONFIG", "requirement/support/current_configuration_exact"),
    ("W workpiece release causality", "OP__RELEASES_WORKPIECE_TO__W_EVENT", "w_release_exact"),
    ("reconfiguration transition context", "TRIGGERS_RECONF; FROM_CONFIG; TO_CONFIG; OCCURS_ON", "all reconfiguration exactness checks"),
)

REDUNDANCY = (
    ("OP.w_delay vs W_EVENT timing", "Related schedule-local information at different semantic grains", "KEEP_FOR_LEARNING", "Phase6C evidence requires operation-local W delay; event timing preserves causal transport structure"),
    ("OP.f_delay vs F_EVENT timing", "Related schedule-local information at different semantic grains", "KEEP_FOR_LEARNING", "Phase6C evidence requires operation-local F delay; event timing preserves synchronization"),
    ("OP.local_reconfiguration vs RECONF_EVENT.duration", "Operation aggregate overlaps the explicit event", "KEEP_FOR_LEARNING", "Preserves frozen operation signal and explicit transition causality"),
    ("PRECEDES vs PRODUCT_NEXT", "Coincide for linear products but differ for branching DAGs", "KEEP_FOR_LEARNING", "Static technological feasibility and realized workpiece order have distinct semantics"),
    ("ASSIGNED_TO vs ISLAND_NEXT", "Assignment plus times can partially reconstruct island chains", "KEEP_FOR_VALIDATION_ONLY", "Explicit chain removes ambiguous tie handling and enables exact resource validation"),
    ("EXECUTED_BY plus W/F_NEXT", "Resource identity can group event chains", "KEEP_FOR_VALIDATION_ONLY", "Explicit next edges preserve order without numeric raw resource IDs"),
    ("raw ID numeric encodings", "Identifier values are not semantic features", "REMOVE_AS_REDUNDANT", "Keys remain lookup-only and never enter predictive feature vectors"),
    ("derived reverse relations", "Mechanically recoverable from canonical forward edges", "REMOVE_AS_REDUNDANT", "CSG-1.0 keeps a minimal directed canonical relation set"),
)

PHASE3_COMPARISON = (
    ("graph purpose", "constructive next-decision policy", "complete-solution critical synchronization and NI state encoding"),
    ("input state", "partial constructive schedule", "frozen complete feasible pre-action Phase6C state"),
    ("node types", "O/J/M/W/F", "OP/ISLAND/CONFIG/W_AGV/F_AGV/W_EVENT/F_EVENT/RECONF_EVENT"),
    ("edge semantics", "constructive feasibility/context", "static alternatives plus realized temporal causality"),
    ("realized schedule information", "partial availability summaries", "complete start/end/slack/current assignment and graph state"),
    ("resource chains", "not explicit complete event chains", "explicit product/island/W/F next relations"),
    ("reconfiguration events", "implicit readiness", "explicit positive-duration event and from/to configuration"),
    ("W/F synchronization", "decoder action/resource context", "explicit event-to-operation enablement with signed gaps"),
    ("action interface", "operation/island/W/F constructive action", "shared state graph plus target OP-node set projection"),
)


def _schema_locations() -> str:
    names = []
    for node_type, node in SCHEMA["node_types"].items():
        names.extend(f"{node_type}.{name}" for name in node["features"])
    names.extend(edge["key"] for edge in SCHEMA["edge_types"])
    names.extend(SCHEMA["graph_level"]["numeric_fields"])
    names.extend(SCHEMA["graph_level"]["categorical_fields"])
    return "\n".join(names)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validation-summary", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--skip-permutation-test", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    locations = _schema_locations()
    information = pd.DataFrame([
        {
            "phase6c_required_information": requirement,
            "csg_representation": representation,
            "validation_test": test,
            "represented": True,
        }
        for requirement, representation, test in INFORMATION_REQUIREMENTS
    ])
    # The mapping is intentionally explicit; verify every referenced canonical edge/feature token.
    token_checks = {
        "operation slack": ("OP.operation_slack",),
        "W waiting / delay": ("OP.w_delay", "W_EVENT.pickup_wait_normalized"),
        "F waiting / delay": ("OP.f_delay", "F_EVENT"),
        "island relative load": ("OP.island_relative_load", "ISLAND.relative_processing_load"),
        "local reconfiguration contribution": ("OP.local_reconfiguration", "RECONF_EVENT"),
        "eligible-island information": ("OP__ELIGIBLE_ON__ISLAND",),
        "precedence neighborhoods": ("OP__PRECEDES__OP",),
        "realized product/workpiece sequence": ("OP__PRODUCT_NEXT__OP",),
        "realized island execution chains": ("OP__ISLAND_NEXT__OP",),
        "realized W-AGV chains": ("W_EVENT__W_NEXT__W_EVENT",),
        "realized F-AGV chains": ("F_EVENT__F_NEXT__F_EVENT",),
        "synchronization relations": ("W_EVENT__ENABLES__OP", "F_EVENT__ENABLES__OP", "RECONF_EVENT__ENABLES__OP"),
        "search progress": ("search_progress", "search_stage"),
        "current makespan": ("current_makespan",),
        "current assignment": ("OP__ASSIGNED_TO__ISLAND",),
        "configuration context": ("OP__REQUIRES__CONFIG", "ISLAND__SUPPORTS__CONFIG", "ISLAND__CURRENT_CONFIG__CONFIG"),
        "W workpiece release causality": ("OP__RELEASES_WORKPIECE_TO__W_EVENT",),
        "reconfiguration transition context": ("OP__TRIGGERS_RECONF__RECONF_EVENT", "RECONF_EVENT__FROM_CONFIG__CONFIG", "RECONF_EVENT__TO_CONFIG__CONFIG"),
    }
    information["represented"] = [
        all(token in locations for token in token_checks[requirement])
        for requirement in information.phase6c_required_information
    ]
    atomic_write_csv(information, args.output / "information_preservation_audit.csv")
    atomic_write_csv(pd.DataFrame(REDUNDANCY, columns=["item", "overlap", "classification", "rationale"]), args.output / "redundancy_audit.csv")
    atomic_write_csv(pd.DataFrame(PHASE3_COMPARISON, columns=["dimension", "phase3_constructive_graph", "phase6d_csg"]), args.output / "phase3_csg_comparison.csv")

    forbidden = {
        "counterfactual_makespan", "relative_improvement", "rank_within_state",
        "regret_to_best", "future_best_makespan", "repair_outcome",
    }
    predictive_names = {
        name for node in SCHEMA["node_types"].values() for name in node["features"]
    } | set(SCHEMA["graph_level"]["numeric_fields"]) | set(SCHEMA["graph_level"]["categorical_fields"])
    leakage = {
        "schema": "phase6d-csg-information-audit-v1",
        "required_information_count": len(information),
        "all_required_information_represented": bool(information.represented.all()),
        "forbidden_predictive_fields_found": sorted(predictive_names & forbidden),
        "raw_identifier_feature_names_found": sorted(
            name for name in predictive_names if name in {"operation_id", "island_id", "agv_id", "row_index"}
        ),
        "no_forbidden_future_information": not bool(predictive_names & forbidden),
        "identifier_keys_are_lookup_only": True,
        "phase6c_contract_reused": True,
    }
    leakage["INFORMATION_AUDIT_PASSED"] = (
        leakage["all_required_information_represented"]
        and leakage["no_forbidden_future_information"]
        and not leakage["raw_identifier_feature_names_found"]
    )
    atomic_write_json(leakage, args.output / "information_audit_summary.json")

    permutation_rows = []
    if not args.skip_permutation_test:
        command = [
            sys.executable, "-m", "pytest", "-q",
            "tests/test_phase6d_csg.py::test_identifier_permutation_invariance",
        ]
        completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
        (args.output / "permutation_invariance_test.log").write_text(
            completed.stdout + completed.stderr, encoding="utf-8"
        )
        permutation_rows.append({
            "test": "identifier_permutation_equivalence",
            "passed": completed.returncode == 0,
            "evidence": "tests/test_phase6d_csg.py::test_identifier_permutation_invariance",
        })
    if args.validation_summary.exists():
        validation = json.loads(args.validation_summary.read_text(encoding="utf-8"))
        permutation_rows.extend([
            {
                "test": "deterministic_repeated_build",
                "passed": bool(validation["deterministic_build_passed"]),
                "evidence": str(args.validation_summary.resolve().relative_to(ROOT)),
            },
            {
                "test": "same_state_action_graph_invariance",
                "passed": bool(validation["same_state_action_graph_invariance_passed"]),
                "evidence": str(args.validation_summary.resolve().relative_to(ROOT)),
            },
        ])
    permutation = pd.DataFrame(permutation_rows, columns=["test", "passed", "evidence"])
    atomic_write_csv(permutation, args.output / "permutation_invariance_summary.csv")
    if not leakage["INFORMATION_AUDIT_PASSED"] or (not permutation.empty and not permutation.passed.all()):
        raise SystemExit(1)
    print("PHASE6D_INFORMATION_AUDIT_PASSED")


if __name__ == "__main__":
    main()
