#!/usr/bin/env python3
"""Fail-closed closure of the Phase 6K historical-input equivalence gate."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_phase6k_start import OUT, digest, write_once


def audit_preprocessing_report(report):
    rows = report["rows"]
    if len(rows) != 288 or len({row["state_id"] for row in rows}) != 288:
        raise ValueError("expected all 288 unique R12 states")
    if sum(row["candidate_rows"] for row in rows) != 6809:
        raise ValueError("expected all 6809 candidate rows")
    if report["r13_accessed"] is not False or report["r14_accessed"] is not False:
        raise ValueError("holdout boundary changed")
    keys = ("bank_identity_and_order_exact", "score_values_exact", "features_exact", "fallback_exact")
    for row in rows:
        if any(type(row[key]) is not bool for key in keys):
            raise ValueError("gate fields must be boolean")
        if row["features_exact"] != all(row["feature_checks"].values()):
            raise ValueError("derived feature summary disagrees with raw checks")
    checks = {key: all(row[key] for row in rows) for key in keys}
    counts = {key: sum(not row[key] for row in rows) for key in keys}
    if checks != report["checks"] or counts != report["mismatch_states"]:
        raise ValueError("summary does not match per-state evidence")
    return checks, counts


def main():
    report_path = OUT / "preprocessing_equivalence.json"
    report = json.loads(report_path.read_text())
    checks, counts = audit_preprocessing_report(report)
    if all(checks.values()):
        raise ValueError("a passing gate cannot be closed with preprocessing HOLD")
    config_path = ROOT / "configs/phase6k_runtime_v1.json"
    config = json.loads(config_path.read_text())
    if config["status"] != "HOLD_BEFORE_RUNTIME_PREREGISTRATION":
        raise ValueError("configuration is not at this stop boundary")
    protected_path = OUT / "protected_evidence.json"
    protected = json.loads(protected_path.read_text())
    if any(digest(ROOT / path) != record["sha256"] for path, record in protected.items()):
        raise ValueError("protected Phase 6I-MR/6J evidence changed")
    if digest(ROOT / "scripts/audit_phase6k_preprocessing.py") != report["script_sha256"]:
        raise ValueError("preprocessing audit implementation changed")
    files = [config_path, *sorted(OUT.glob("*.json")), Path(__file__),
             ROOT / "rcias_clgri/search/lghga_2o.py", ROOT / "configs/lghga_2o_baseline.json"]
    final = {"schema": "phase6k-preprocessing-hold-v1", "decision": "HOLD",
             "reason": "historical FP16 source-score/derived-feature reproduction fails exact frozen input contract",
             "stop_boundary": "BEFORE_RUNTIME_PREREGISTRATION_AND_E1_E2",
             "audit_status": "COMPLETE", "phase6k_success": False, "checks": checks,
             "mismatch_states": counts, "states": 288, "candidate_rows": 6809,
             "protected_evidence_unchanged": True, "runtime_equivalence": "NOT_RUN",
             "final_selected_action_equivalence": "NOT_RUN_AFTER_INPUT_EQUIVALENCE_FAILURE",
             "formal_runtime": "NOT_RUN", "solver_sanity": "NOT_RUN",
             "r13_accessed": False, "r14_accessed": False, "csg_ni_v1_frozen": False,
             "artifact_sha256": {str(p.relative_to(ROOT)): digest(p) for p in files},
             "next_step": "resolve the historical preprocessing reproducibility failure without altering the frozen input contract; do not start formal timing or unlock holdouts"}
    write_once(OUT.parent / "final/final_decision.json", final)
    print(json.dumps({k: v for k, v in final.items() if k != "artifact_sha256"}, indent=2))


if __name__ == "__main__":
    main()
