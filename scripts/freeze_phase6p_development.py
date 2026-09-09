#!/usr/bin/env python3
"""Freeze the exact Phase 6P P3 matched-budget execution contract."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/phase6p_adaptive_portfolio_v1/development/protocol.json"
REGISTRY_ROOT = ROOT / "outputs/frozen_2o_baselines"
METHODS = [
    "P1_CSG_ADAPTIVE_PORTFOLIO",
    "PHASE6N_DETERMINISTIC_TOP1",
    "ALNS",
    "PHASE6H",
    "LG_HGA_2O",
]


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    if OUT.exists():
        raise RuntimeError("Phase 6P development protocol already exists")
    if subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip():
        raise RuntimeError("freeze Phase 6P development only from a clean committed worktree")
    registry = load_json(REGISTRY_ROOT / "registry.json")
    audit_path = ROOT / "outputs/phase6p_adaptive_portfolio_v1/audit/comparator_reuse_audit.json"
    audit = load_json(audit_path)
    if (
        registry["status"] != "INITIALIZED_ALL_REQUIRED_RUNS_PENDING"
        or audit["status"] != "COMPLETE_BEFORE_COMPARATOR_EXECUTION"
        or not audit["comparators_may_start"]
        or any(entry["status"] != "RERUN_REQUIRED" for entry in registry["entries"])
    ):
        raise RuntimeError("canonical comparator audit/registry is not ready")
    instance_manifest_path = REGISTRY_ROOT / "instance_manifest.json"
    instance_manifest = load_json(instance_manifest_path)
    if len(instance_manifest["instances"]) != 18:
        raise RuntimeError("Phase 6P P3 requires 18 R12 CAUR-FIT instances")
    if any((
        (ROOT / "outputs/phase6j_caur/r13_selection/access_ledger.json").exists(),
        (ROOT / "outputs/phase6j_caur/r14_holdout/access_ledger.json").exists(),
    )):
        raise RuntimeError("R13/R14 must remain locked")
    source_paths = [
        "scripts/run_phase6p_development.py",
        "rcias_clgri/search/phase6p_adaptive.py",
        "rcias_clgri/ni/phase6p_live_inference.py",
        "rcias_clgri/search/alns.py",
        "rcias_clgri/search/csgni.py",
        "rcias_clgri/ni/live_inference.py",
        "rcias_clgri/search/lghga.py",
        "rcias_clgri/search/lghga_2o.py",
        "rcias_clgri/search/lghga_v2.py",
        "rcias_clgri/search/lghga_learning.py",
        "rcias_clgri/search/lghga_neighborhoods.py",
        "rcias_clgri/search/lghga_neighborhoods_v2.py",
        "rcias_clgri/search/common.py",
        "rcias_clgri/env/insertion_decoder.py",
        "rcias_clgri/env/feasibility.py",
        "rcias_clgri/heuristic/dispatching.py",
    ]
    artifact_paths = [
        "configs/phase6p_adaptive_portfolio_v1.json",
        "configs/phase5c_alns.json",
        "configs/phase6h_live_calibration.json",
        "configs/lghga_2o_baseline.json",
        "outputs/phase6p_adaptive_portfolio_v1/preregistration/preregistration.json",
        "outputs/phase6p_adaptive_portfolio_v1/preregistration/phase6n_checkpoint_manifest.json",
        "outputs/phase6p_adaptive_portfolio_v1/audit/comparator_reuse_audit.json",
        "outputs/phase6h_calibration/frozen/phase6h_policy.json",
        "outputs/phase6h_calibration/frozen/freeze_record.json",
        "outputs/phase6f/audit/experiment_freeze.json",
        "outputs/baselines/lghga_kb_v2/models/model_manifest.json",
    ]
    seeds = [746101, 746102, 746103]
    nominal = 5 * len(seeds) * sum(
        2 * int(row["num_operations"]) for row in instance_manifest["instances"]
    )
    protocol = {
        "schema": "phase6p-development-protocol-v1",
        "status": "FROZEN_BEFORE_P3_SOLVER_OUTCOMES",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "methods": METHODS,
        "execution_order": "method-major, then frozen instance-manifest order, then seed order",
        "development_seeds": seeds,
        "instances": instance_manifest["instances"],
        "task_count": 270,
        "budget_formula": "2 * instance.num_operations seconds",
        "nominal_total_budget_seconds": nominal,
        "nominal_total_budget_hours": nominal / 3600.0,
        "initialization_accounting": "H1 or canonical LG_HGA initializer and all instance-specific preprocessing inside each run budget",
        "model_residency": "one-time checkpoint/model loading outside per-run budget and reported separately",
        "execution_concurrency": 1,
        "concurrency_reason": "avoid CPU/GPU contention in matched wall-clock comparisons",
        "phase6h_semantics": {
            "policy": "CALIBRATED_PROBABILITY_UTILITY frozen Phase 6H deployment",
            "intervention_rate_percent": 100,
            "proposal_namespace": 670102,
            "ni_repair_namespace": 670103,
            "acceptance_namespace": 670104,
        },
        "phase6n_top1_semantics": "same Phase 6P 20% schedule and search mechanics; deterministic neural rank 1 replaces top-6 roulette",
        "primary_semantics": "P1_CSG_ADAPTIVE_PORTFOLIO preregistered top-6 inverse-rank by mean-origin ALNS destroy weight",
        "lghga_semantics": "frozen LG_HGA-2O initializer/search/DTRs; only outer generation stop removed",
        "result_provenance": "NEW_RUN",
        "resume_semantics": "validate every atomic per-run payload against this protocol hash and skip only complete valid runs",
        "instance_manifest_sha256": digest(instance_manifest_path),
        "initial_registry_sha256": digest(REGISTRY_ROOT / "registry.json"),
        "source_hashes": {path: digest(ROOT / path) for path in source_paths},
        "artifact_hashes": {path: digest(ROOT / path) for path in artifact_paths},
        "command": [
            "/home/liulei/miniconda3/envs/gnn311/bin/python",
            "-u",
            "scripts/run_phase6p_development.py",
            "--device",
            "cuda",
        ],
        "gurobi": False,
        "r13_accessed": False,
        "r14_accessed": False,
    }
    atomic_json(OUT, protocol)
    print(json.dumps({
        "status": protocol["status"],
        "task_count": protocol["task_count"],
        "nominal_hours": protocol["nominal_total_budget_hours"],
        "path": str(OUT.relative_to(ROOT)),
    }))


if __name__ == "__main__":
    main()
