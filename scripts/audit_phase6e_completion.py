#!/usr/bin/env python3
"""Run Phase 6E regressions and audit the offline-validation completion gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
PHASE = ROOT / "outputs" / "phase6e"
AUDIT = PHASE / "audit"
REPORT = ROOT / "docs" / "reports" / "phase6e_supervised_ni_validation_report.md"
PHASE6C_FREEZE_HASH = "695307ac6193ecbbeb0f73e81a94ea20672ffd47aa9419bb37610be9d3161437"

FIGURE_STEMS = [
    "Fig01_training_validation_curves",
    "Fig02_internal_holdout_ranking_comparison",
    "Fig03_selected_action_improvement",
    "Fig04_selected_action_regret",
    "Fig05_full_csg_vs_flat_set",
    "Fig06_full_csg_vs_static_csg",
    "Fig07_performance_by_scale_cf",
    "Fig08_performance_by_ri_ti",
    "Fig09_performance_by_search_stage",
    "Fig10_inference_latency_by_scale",
]

REPORT_SECTIONS = [
    "## 1. Executive conclusion",
    "## 2. Frozen Phase 6C/6D boundary",
    "## 3. Tensorization and data pipeline",
    "## 4. Model architecture",
    "## 5. Target-set action encoder",
    "## 6. Training objective",
    "## 7. Baselines",
    "## 8. Sanity/leakage tests",
    "## 9. Validation model selection",
    "## 10. Internal-holdout predictive results",
    "## 11. Selected-action utility",
    "## 12. Full CSG vs flat representation",
    "## 13. Synchronization-topology ablation",
    "## 14. Structural-regime robustness",
    "## 15. Seed stability",
    "## 16. Runtime and memory",
    "## 17. Failure cases",
    "## 18. Scientific interpretation",
    "## 19. Phase 6F recommendation",
    "## 20. Reproducibility checklist",
]

CONCLUSION_KEYS = [
    "TENSORIZER_VALIDATED",
    "TARGET_SET_SCORER_TRAINED",
    "THREE_MODEL_SEEDS_COMPLETE",
    "NO_LABEL_LEAKAGE",
    "FULL_CSG_BEATS_RANDOM",
    "FULL_CSG_BEATS_RELATED",
    "FULL_CSG_BEATS_BEST_FIXED_OPERATOR",
    "FULL_CSG_BEATS_TABULAR_BASELINE",
    "FULL_CSG_BEATS_FLAT_SET_MODEL",
    "REALIZED_SYNCHRONIZATION_TOPOLOGY_ADDS_VALUE",
    "EDGE_FEATURES_ADD_VALUE",
    "MODEL_STABLE_ACROSS_SEEDS",
    "MODEL_STABLE_ACROSS_STRUCTURAL_REGIMES",
    "INFERENCE_COST_ACCEPTABLE_FOR_SOLVER_INTEGRATION",
    "PHASE6F_RECOMMENDATION",
]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def run(name: str, arguments: list[str]) -> dict:
    started = time.perf_counter()
    environment = dict(os.environ)
    matplotlib_cache = Path("/tmp/ri_acrsp_matplotlib")
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    environment.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    completed = subprocess.run(
        [sys.executable, *arguments],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=environment,
        check=False,
    )
    log = AUDIT / f"{name}.log"
    log.write_text(completed.stdout, encoding="utf-8")
    return {
        "name": name,
        "command": [sys.executable, *arguments],
        "return_code": completed.returncode,
        "runtime_seconds": time.perf_counter() - started,
        "log": str(log.relative_to(ROOT)),
        "passed": completed.returncode == 0,
    }


def exact_record_hash(record: dict) -> str:
    payload = dict(record)
    expected = payload.pop("freeze_hash")
    actual = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return expected == actual


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-regressions", action="store_true")
    args = parser.parse_args()
    AUDIT.mkdir(parents=True, exist_ok=True)

    commands = []
    if not args.skip_regressions:
        commands = [
            run("compileall", ["-m", "compileall", "-q", "rcias_clgri", "scripts", "tests"]),
            run("pytest", ["-m", "pytest", "-q"]),
            run("canonical_benchmarks", ["scripts/generate_canonical_benchmarks.py", "--verify-only"]),
            run("small_validation", ["scripts/run_small_validation.py"]),
            run(
                "phase6b_train_benchmarks",
                ["scripts/generate_phase6b_train_distribution.py", "--verify-only"],
            ),
        ]

    evaluation_path = PHASE / "evaluation_v2" / "evaluation_completion.json"
    access_path = PHASE / "evaluation_v2" / "holdout_access_record.json"
    evaluation = load_json(evaluation_path)
    access = load_json(access_path)
    conclusions = load_json(AUDIT / "scientific_conclusions.json")
    sanity = load_json(PHASE / "sanity" / "mandatory_sanity.json")
    tensor_cache = load_json(AUDIT / "tensor_cache_audit.json")
    config_study = load_json(AUDIT / "config_study_audit.json")
    final_training = load_json(AUDIT / "final_training_audit.json")
    ablation_training = load_json(AUDIT / "ablation_training_audit.json")
    profile = load_json(PHASE / "profiling" / "inference_profile_summary.json")

    phase6a_gate = load_json(ROOT / "outputs" / "phase6a" / "diagnostics" / "completion_gate.json")
    phase6a_regression = load_json(
        ROOT / "outputs" / "phase6a" / "diagnostics" / "instrumentation_regression.json"
    )
    phase6b_gate = load_json(ROOT / "outputs" / "phase6b" / "audit" / "completion_gate.json")
    phase6b_integrity = load_json(
        ROOT / "outputs" / "phase6b" / "audit" / "counterfactual_integrity_audit.json"
    )
    phase6c_gate = load_json(ROOT / "outputs" / "phase6c" / "audit" / "completion_gate.json")
    phase6c_freeze = load_json(
        ROOT / "outputs" / "phase6c" / "audit" / "dataset_freeze_record.json"
    )
    phase6d_gate = load_json(ROOT / "outputs" / "phase6d" / "audit" / "completion_gate.json")
    phase6d_freeze = load_json(ROOT / "outputs" / "phase6d" / "schema_freeze_record.json")

    holdout_hashes_exact = all(
        (ROOT / relative).exists() and digest(ROOT / relative) == expected
        for relative, expected in evaluation["outputs"].items()
    )
    final_manifest = load_json(PHASE / "training" / "final_seeds" / "checkpoint_manifest.json")
    final_checkpoints_exact = all(
        Path(row["checkpoint_path"]).exists()
        and digest(Path(row["checkpoint_path"])) == row["checkpoint_sha256"]
        for row in final_manifest["checkpoints"]
    )

    phase6c_files_exact = all((
        (
            digest(ROOT / "configs" / "phase6c_counterfactual.json")
            == phase6c_freeze["production_config_sha256"]
        ),
        (
            digest(ROOT / "outputs" / "phase6c" / "environment" / "production_config_freeze.json")
            == phase6c_freeze["production_config_freeze_sha256"]
        ),
        (
            digest(ROOT / "outputs" / "phase6c" / "manifests" / "state_manifest.csv")
            == phase6c_freeze["state_manifest_sha256"]
        ),
        (
            digest(ROOT / "outputs" / "phase6c" / "manifests" / "shard_manifest.csv")
            == phase6c_freeze["shard_manifest_sha256"]
        ),
    ))
    phase6d_sources_exact = phase6d_freeze["csg_source_sha256"] == {
        str(path.relative_to(ROOT)): digest(path)
        for path in sorted((ROOT / "rcias_clgri" / "csg").glob("*.py"))
    }

    report_text = REPORT.read_text(encoding="utf-8")
    figure_paths = [
        PHASE / "figures" / f"{stem}.{suffix}"
        for stem in FIGURE_STEMS
        for suffix in ("png", "pdf")
    ]
    figures_complete = all(path.exists() and path.stat().st_size > 10_000 for path in figure_paths)
    report_sections_complete = all(section in report_text for section in REPORT_SECTIONS)
    report_conclusions_complete = all(f"{key} =" in report_text for key in CONCLUSION_KEYS)

    required_outputs = [
        PHASE / "training" / "final_seeds" / "model_seed_summary.csv",
        PHASE / "training" / "final_seeds" / "validation_metrics.csv",
        PHASE / "evaluation_v2" / "internal_holdout_metrics.csv",
        PHASE / "evaluation_v2" / "selected_action_utility.csv",
        PHASE / "evaluation_v2" / "structural_regime_metrics.csv",
        PHASE / "ablations" / "ablation_summary.csv",
        PHASE / "statistics" / "pairwise_statistics.csv",
        PHASE / "profiling" / "inference_profile.csv",
        PHASE / "training" / "final_seeds" / "checkpoint_manifest.json",
        AUDIT / "experiment_freeze.json",
        AUDIT / "scientific_conclusions.json",
        REPORT,
        *[
            PHASE / "training" / "final_seeds" / f"seed_{seed}" / "training_history.csv"
            for seed in (660201, 660202, 660203)
        ],
    ]
    partial_files = sorted(
        str(path.relative_to(ROOT))
        for pattern in ("*.partial*", "*.tmp")
        for path in PHASE.rglob(pattern)
    )
    smoke_paths = sorted(
        str(path.relative_to(ROOT))
        for path in (PHASE / "training").glob("smoke_*")
    )

    status_lines = subprocess.run(
        ["git", "status", "--short"], cwd=ROOT, text=True, capture_output=True, check=True
    ).stdout.splitlines()
    changed_paths = [line[3:] for line in status_lines if len(line) >= 4]
    frozen_semantic_prefixes = (
        "rcias_clgri/search/",
        "rcias_clgri/env/",
        "rcias_clgri/exact/",
        "rcias_clgri/heuristic/",
        "rcias_clgri/graph/",
        "rcias_clgri/csg/",
    )
    frozen_semantic_changes = sorted(
        path for path in changed_paths if path.startswith(frozen_semantic_prefixes)
    )

    c = conclusions["conclusions"]
    checks = {
        "regression_commands_passed": bool(commands) and all(row["passed"] for row in commands),
        "phase6e_v2_holdout_complete": (
            evaluation["status"] == "COMPLETE"
            and evaluation["experiment_version"] == "phase6e-holdout-v2"
            and evaluation["state_count"] == 20_000
            and evaluation["action_count"] == 472_452
        ),
        "holdout_opened_after_freeze": (
            access["status"] == "COMPLETE" and not access["checkpoint_selection_after_open"]
        ),
        "holdout_completion_hash_exact": access["evaluation_completion_sha256"] == digest(evaluation_path),
        "holdout_output_hashes_exact": holdout_hashes_exact,
        "v1_invalidation_retained": (PHASE / "evaluation" / "holdout_v1_invalidation.json").exists(),
        "tensor_cache_audit_passed": (
            tensor_cache["status"] == "PASS" and all(tensor_cache["checks"].values())
        ),
        "mandatory_sanity_passed": sanity["status"] == "PASS" and all(sanity["checks"].values()),
        "config_study_audit_passed": config_study["status"] == "PASS",
        "final_training_audit_passed": final_training["status"] == "PASS",
        "ablation_training_audit_passed": ablation_training["status"] == "PASS",
        "three_final_checkpoints_exact": len(final_manifest["checkpoints"]) == 3 and final_checkpoints_exact,
        "inference_profile_complete": set(profile["checks"]) == {
            "p90_ensemble_latency_within_budget",
            "projected_1000_decisions_within_budget",
            "gpu_memory_within_budget",
        },
        "inference_failure_reported_honestly": (
            profile["status"] == "FAIL"
            and c["INFERENCE_COST_ACCEPTABLE_FOR_SOLVER_INTEGRATION"] == "FALSE"
            and c["PHASE6F_RECOMMENDATION"] == "REVISE_MODEL"
        ),
        "required_machine_outputs_present": all(
            path.exists() and path.stat().st_size > 0 for path in required_outputs
        ),
        "required_figures_png_pdf_present": figures_complete,
        "report_20_sections_complete": report_sections_complete,
        "explicit_conclusions_complete": report_conclusions_complete,
        "phase6a_instrumentation_regression_preserved": (
            phase6a_gate["PHASE6A_COMPLETE"]
            and phase6a_gate["INSTRUMENTATION_REGRESSION_PASSED"]
            and not phase6a_regression["INSTRUMENTATION_CHANGES_SEARCH_BEHAVIOR"]
        ),
        "phase6b_counterfactual_regression_preserved": (
            phase6b_gate["PHASE6B_COMPLETE"]
            and phase6b_gate["COUNTERFACTUAL_EVALUATOR_IMMUTABLE"]
            and all(value is True for key, value in phase6b_integrity.items() if key != "evidence")
        ),
        "phase6c_dataset_checksums_preserved": (
            phase6c_gate["PHASE6C_COMPLETE"]
            and phase6c_freeze["freeze_hash"] == PHASE6C_FREEZE_HASH
            and exact_record_hash(phase6c_freeze)
            and phase6c_files_exact
            and tensor_cache["checks"]["source_hashes_exact"]
        ),
        "phase6d_schema_and_sources_preserved": (
            phase6d_gate["PHASE6D_COMPLETE"]
            and phase6d_freeze["all_acceptance_gates_passed"]
            and phase6d_freeze["csg_schema_sha256"]
            == digest(ROOT / "configs" / "csg_v1_schema.json")
            and phase6d_sources_exact
        ),
        "historical_phase3_graph_behavior_preserved": phase6d_gate[
            "HISTORICAL_GRAPH_BEHAVIOR_UNCHANGED"
        ],
        "frozen_solver_semantics_untouched": not frozen_semantic_changes,
        "partial_files_absent": not partial_files,
        "abandoned_smoke_artifacts_absent": not smoke_paths,
    }
    audit_status = "PASS" if all(checks.values()) else "FAIL"

    artifact_paths = sorted(set(required_outputs + figure_paths + [evaluation_path, access_path]))
    artifact_manifest = {
        "schema": "phase6e-final-artifact-manifest-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": {
            str(path.relative_to(ROOT)): {
                "bytes": path.stat().st_size,
                "sha256": digest(path),
            }
            for path in artifact_paths
            if path.exists()
        },
    }
    (AUDIT / "artifact_manifest.json").write_text(
        json.dumps(artifact_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    regression = {
        "schema": "phase6e-regression-summary-v1",
        "commands": commands,
        "all_passed": bool(commands) and all(row["passed"] for row in commands),
    }
    (AUDIT / "regression_summary.json").write_text(
        json.dumps(regression, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    repository_audit = {
        "schema": "phase6e-repository-audit-v1",
        "audit_status": audit_status,
        "checks": checks,
        "missing_required_outputs": [
            str(path.relative_to(ROOT)) for path in required_outputs if not path.exists()
        ],
        "partial_files": partial_files,
        "abandoned_smoke_artifacts": smoke_paths,
        "frozen_semantic_changes": frozen_semantic_changes,
        "worktree_paths": changed_paths,
    }
    (AUDIT / "repository_audit.json").write_text(
        json.dumps(repository_audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    completion_gate = {
        "schema": "phase6e-completion-gate-v1",
        "PHASE6E_COMPLETE": audit_status == "PASS",
        "OFFLINE_HOLDOUT_V2_COMPLETE": checks["phase6e_v2_holdout_complete"],
        "TENSORIZER_VALIDATED": c["TENSORIZER_VALIDATED"] == "TRUE",
        "TARGET_SET_SCORER_TRAINED": c["TARGET_SET_SCORER_TRAINED"] == "TRUE",
        "THREE_MODEL_SEEDS_COMPLETE": c["THREE_MODEL_SEEDS_COMPLETE"] == "TRUE",
        "NO_LABEL_LEAKAGE_DETECTED": c["NO_LABEL_LEAKAGE"] == "TRUE",
        "REQUIRED_FIGURES_COMPLETE": figures_complete,
        "FINAL_REPORT_COMPLETE": report_sections_complete and report_conclusions_complete,
        "FULL_REGRESSION_SUITE_PASSED": checks["regression_commands_passed"],
        "REPOSITORY_AUDIT_PASSED": audit_status == "PASS",
        "INFERENCE_PROFILE_COMPLETE": checks["inference_profile_complete"],
        "INFERENCE_COST_ACCEPTABLE_FOR_SOLVER_INTEGRATION": False,
        "PHASE6F_LIVE_INTEGRATION_APPROVED": False,
        "PHASE6F_RECOMMENDATION": c["PHASE6F_RECOMMENDATION"],
        "STOP_CONDITION_ENFORCED": True,
    }
    (AUDIT / "completion_gate.json").write_text(
        json.dumps(completion_gate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    if audit_status != "PASS":
        failed = {key: value for key, value in checks.items() if not value}
        raise RuntimeError(f"Phase 6E completion audit failed: {failed}")
    print("PHASE6E_COMPLETION_AUDIT_PASS")


if __name__ == "__main__":
    main()
