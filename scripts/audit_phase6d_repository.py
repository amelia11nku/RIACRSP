#!/usr/bin/env python3
"""Reproduce Phase 6D regressions and audit frozen repository boundaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/phase6d/audit"
PHASE6C_FREEZE_HASH = "695307ac6193ecbbeb0f73e81a94ea20672ffd47aa9419bb37610be9d3161437"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def run(name: str, arguments: list[str]) -> dict:
    started = time.perf_counter()
    environment = dict(os.environ)
    cache = Path("/tmp/ri_acrsp_matplotlib")
    cache.mkdir(parents=True, exist_ok=True)
    environment.setdefault("MPLCONFIGDIR", str(cache))
    completed = subprocess.run(
        [sys.executable, *arguments], cwd=ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        env=environment, check=False,
    )
    log = OUT / f"{name}.log"
    log.write_text(completed.stdout, encoding="utf-8")
    return {
        "name": name,
        "command": [sys.executable, *arguments],
        "return_code": completed.returncode,
        "runtime_seconds": time.perf_counter() - started,
        "log": str(log.relative_to(ROOT)),
        "passed": completed.returncode == 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-regression", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    commands = []
    if not args.skip_regression:
        commands = [
            run("compileall", ["-m", "compileall", "-q", "rcias_clgri", "scripts", "tests"]),
            run("pytest", ["-m", "pytest", "-q"]),
            run("canonical_benchmarks", ["scripts/generate_canonical_benchmarks.py", "--verify-only"]),
            run("small_validation", ["scripts/run_small_validation.py"]),
            run("train_benchmarks", ["scripts/generate_phase6b_train_distribution.py", "--verify-only"]),
        ]

    phase6c_freeze = json.loads((ROOT / "outputs/phase6c/audit/dataset_freeze_record.json").read_text())
    phase6c_gate = json.loads((ROOT / "outputs/phase6c/audit/completion_gate.json").read_text())
    phase6c_integrity = json.loads((ROOT / "outputs/phase6c/audit/counterfactual_integrity.json").read_text())
    phase6a_regression = json.loads((ROOT / "outputs/phase6a/diagnostics/instrumentation_regression.json").read_text())
    controlled = json.loads((ROOT / "outputs/phase5c/controlled_benchmark_audit/coverage_diagnostics.json").read_text())
    validation = json.loads((ROOT / "outputs/phase6d/validation/csg_validation_summary.json").read_text())
    information = json.loads((ROOT / "outputs/phase6d/information_audit/information_audit_summary.json").read_text())
    examples = json.loads((ROOT / "outputs/phase6d/examples/examples_summary.json").read_text())
    complexity = json.loads((ROOT / "outputs/phase6d/profiling/complexity_model.json").read_text())

    required = [
        ROOT / "configs/csg_v1_schema.json",
        ROOT / "outputs/phase6d/validation/csg_validation_summary.json",
        ROOT / "outputs/phase6d/validation/node_type_summary.csv",
        ROOT / "outputs/phase6d/validation/edge_type_summary.csv",
        ROOT / "outputs/phase6d/validation/temporal_consistency_summary.csv",
        ROOT / "outputs/phase6d/validation/resource_chain_summary.csv",
        ROOT / "outputs/phase6d/information_audit/information_preservation_audit.csv",
        ROOT / "outputs/phase6d/information_audit/redundancy_audit.csv",
        ROOT / "outputs/phase6d/information_audit/permutation_invariance_summary.csv",
        ROOT / "outputs/phase6d/profiling/profiling_summary.csv",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    partials = [str(path.relative_to(ROOT)) for path in (ROOT / "outputs/phase6d").rglob("*.partial.*")]
    status = subprocess.run(
        ["git", "status", "--short"], cwd=ROOT, text=True,
        capture_output=True, check=True,
    ).stdout.splitlines()
    frozen_source_prefixes = (
        "rcias_clgri/graph/", "rcias_clgri/env/", "rcias_clgri/search/",
        "rcias_clgri/nn/", "rcias_clgri/training/", "instances/",
        "outputs/phase6a/", "outputs/phase6b/", "outputs/phase6c/",
    )
    changed_paths = [line[3:] for line in status if len(line) >= 4]
    frozen_changes = sorted(
        path for path in changed_paths if path.startswith(frozen_source_prefixes)
    )
    csg_sources = sorted((ROOT / "rcias_clgri/csg").glob("*.py"))
    forbidden_imports = {
        str(path.relative_to(ROOT)): token
        for path in csg_sources
        for token in ("torch", "torch_geometric", "pyg", "hgt")
        if token in path.read_text(encoding="utf-8").lower()
    }
    checks = {
        "regression_commands_passed": all(command["passed"] for command in commands),
        "phase6c_freeze_hash_matches": phase6c_freeze["freeze_hash"] == PHASE6C_FREEZE_HASH,
        "phase6c_complete": bool(phase6c_gate["PHASE6C_COMPLETE"]),
        "phase6c_integrity_passed": bool(phase6c_integrity["COUNTERFACTUAL_INTEGRITY_PASSED"]),
        "phase6a_instrumentation_unchanged": not bool(phase6a_regression["INSTRUMENTATION_CHANGES_SEARCH_BEHAVIOR"]),
        "controlled_108_feasible": bool(controlled["all_108_feasible"]),
        "phase6d_10000_state_validation_passed": validation["completed_state_count"] == 10_000 and bool(validation["all_structural_checks_passed"]),
        "phase6d_information_audit_passed": bool(information["INFORMATION_AUDIT_PASSED"]),
        "phase6d_examples_passed": examples["example_count"] == 7 and bool(examples["all_validation_passed"]),
        "phase6d_profiling_passed": complexity["profiled_state_count"] >= 30 and bool(complexity["all_validation_passed"]),
        "required_outputs_present": not missing,
        "partial_files_absent": not partials,
        "frozen_source_boundaries_unchanged": not frozen_changes,
        "no_neural_framework_in_csg": not forbidden_imports,
    }
    payload = {
        "schema": "phase6d-repository-audit-v1",
        "checks": checks,
        "regression_commands": commands,
        "phase6c_dataset_freeze_hash": phase6c_freeze["freeze_hash"],
        "missing_required_outputs": missing,
        "partial_files": partials,
        "frozen_source_changes": frozen_changes,
        "preexisting_or_in_scope_worktree_changes": changed_paths,
        "forbidden_neural_imports": forbidden_imports,
        "csg_source_sha256": {
            str(path.relative_to(ROOT)): digest(path) for path in csg_sources
        },
        "audit_status": "PASS" if all(checks.values()) else "FAIL",
    }
    (OUT / "regression_summary.json").write_text(
        json.dumps({
            "schema": "phase6d-regression-summary-v1",
            "commands": commands,
            "all_passed": all(command["passed"] for command in commands),
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (OUT / "repository_audit.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if payload["audit_status"] != "PASS":
        raise RuntimeError(payload)
    print("PHASE6D_REPOSITORY_AUDIT_PASS")


if __name__ == "__main__":
    main()
