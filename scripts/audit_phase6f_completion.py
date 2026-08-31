#!/usr/bin/env python3
"""Run Phase 6F regressions and produce the final repository completion gate."""

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
PHASE = ROOT / "outputs/phase6f"
AUDIT = PHASE / "audit"
REPORT = ROOT / "docs/reports/phase6f_utility_aware_model_revision_report.md"

FIGURES = [
    "Fig01_utility_objective_comparison",
    "Fig02_compact_model_quality_latency_frontier",
    "Fig03_calibration_reliability",
    "Fig04_intervention_coverage_utility_tradeoff",
    "Fig05_revision_holdout_selected_utility",
    "Fig06_revision_holdout_regret",
    "Fig07_old_vs_revised_model",
    "Fig08_revision_performance_by_scale_cf",
    "Fig09_revision_performance_by_ri_ti",
    "Fig10_deployment_latency",
]
REPORT_SECTIONS = [f"## {index}." for index in range(1, 20)]
CONCLUSIONS = [
    "REVISION_HOLDOUT_CREATED",
    "REVISION_HOLDOUT_STATE_COUNT",
    "REVISION_HOLDOUT_UNTOUCHED_UNTIL_MODEL_FREEZE",
    "UTILITY_AWARE_OBJECTIVE_ADDS_VALUE",
    "CALIBRATION_VALIDATED",
    "SELECTIVE_INTERVENTION_VALIDATED",
    "DISTILLATION_USED",
    "COMPACT_SINGLE_MODEL_READY",
    "COMPACT_MODEL_BEATS_PHASE6E_ENSEMBLE_UTILITY",
    "COMPACT_MODEL_PRESERVES_PHASE6E_UTILITY",
    "MODEL_DECISION_P90_S_MS",
    "MODEL_DECISION_P90_M_MS",
    "MODEL_DECISION_P90_L_MS",
    "LATENCY_GATE_PASSED",
    "MODEL_STABLE_ACROSS_SEEDS",
    "MODEL_STABLE_ACROSS_STRUCTURAL_REGIMES",
    "PHASE6G_RECOMMENDATION",
]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def canonical_hash(record: dict[str, object], key: str) -> str:
    payload = {name: value for name, value in record.items() if name != key}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def run(name: str, arguments: list[str]) -> dict[str, object]:
    started = time.perf_counter()
    environment = dict(os.environ)
    cache = Path("/tmp/phase6f_audit_matplotlib")
    cache.mkdir(parents=True, exist_ok=True)
    environment["MPLCONFIGDIR"] = str(cache)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-regressions", action="store_true")
    args = parser.parse_args()
    AUDIT.mkdir(parents=True, exist_ok=True)
    regression_path = AUDIT / "regression_summary.json"
    if args.skip_regressions:
        regression = load(regression_path)
        commands = regression["commands"]
    else:
        commands = [
            run("compileall", ["-m", "compileall", "-q", "rcias_clgri", "scripts", "tests"]),
            run("pytest", ["-m", "pytest", "-q"]),
            run("canonical_benchmarks", ["scripts/generate_canonical_benchmarks.py", "--verify-only"]),
            run("small_validation", ["scripts/run_small_validation.py"]),
            run("phase6b_train_benchmarks", ["scripts/generate_phase6b_train_distribution.py", "--verify-only"]),
        ]
        regression = {
            "schema": "phase6f-regression-summary-v1",
            "commands": commands,
            "all_passed": all(bool(row["passed"]) for row in commands),
        }
        atomic_json(regression_path, regression)

    environment = load(PHASE / "environment/freeze_record.json")
    state = load(PHASE / "revision_holdout/state_audit.json")
    label_open = load(AUDIT / "revision_holdout_label_open.json")
    cache = load(AUDIT / "revision_holdout_cache_audit.json")
    features = load(AUDIT / "r06_tabular_feature_audit.json")
    sanity = load(AUDIT / "mandatory_sanity.json")
    model_freeze = load(AUDIT / "experiment_freeze.json")
    evaluation_path = PHASE / "evaluation/revision_holdout_evaluation_completion.json"
    evaluation = load(evaluation_path)
    latency = load(PHASE / "profiling/latency_profile_summary.json")
    gate = load(AUDIT / "deployment_gate.json")
    checkpoint_manifest = load(PHASE / "training/final_seeds/checkpoint_manifest.json")

    output_hashes_exact = all(
        (ROOT / relative).exists() and digest(ROOT / relative) == expected
        for relative, expected in evaluation["outputs"].items()
    )
    checkpoints_exact = all(
        Path(str(row["checkpoint_path"])).exists()
        and digest(Path(str(row["checkpoint_path"]))) == row["checkpoint_sha256"]
        for row in checkpoint_manifest["checkpoints"]
    )
    frozen_phase6e_exact = all([
        digest(ROOT / "outputs/phase6e/audit/artifact_manifest.json")
        == environment["phase6e_artifact_manifest_sha256"],
        digest(ROOT / "outputs/phase6e/audit/completion_gate.json")
        == environment["phase6e_completion_gate_sha256"],
        digest(ROOT / "outputs/phase6e/audit/repository_audit.json")
        == environment["phase6e_repository_audit_sha256"],
        digest(ROOT / "outputs/phase6e/training/final_seeds/checkpoint_manifest.json")
        == environment["phase6e_checkpoint_manifest_sha256"],
    ])

    required = [
        PHASE / "objectives/objective_validation_summary.csv",
        PHASE / "compact_models/compact_model_validation_summary.csv",
        PHASE / "calibration/calibration_summary.csv",
        PHASE / "calibration/selective_policy_summary.csv",
        PHASE / "distillation/distillation_summary.csv",
        PHASE / "evaluation/revision_holdout_metrics.csv",
        PHASE / "statistics/revision_holdout_pairwise_statistics.csv",
        PHASE / "evaluation/revision_holdout_structural_summary.csv",
        PHASE / "profiling/latency_profile.csv",
        PHASE / "training/final_seeds/checkpoint_manifest.json",
        AUDIT / "experiment_freeze.json",
        AUDIT / "deployment_gate.json",
        REPORT,
    ]
    figure_paths = [
        PHASE / "figures" / f"{name}.{suffix}"
        for name in FIGURES
        for suffix in ("png", "pdf")
    ]
    report_text = REPORT.read_text(encoding="utf-8")
    partial_files = sorted(
        str(path.relative_to(ROOT))
        for pattern in ("*.partial", "*.tmp")
        for path in PHASE.rglob(pattern)
    )
    checks = {
        "regression_commands_passed": bool(commands) and all(
            bool(row["passed"]) for row in commands
        ),
        "phase6a_to_phase6e_evidence_frozen": bool(
            environment["status"] == "FROZEN_BEFORE_R06_GENERATION"
            and all(environment["checks"].values())
            and environment["freeze_hash"] == canonical_hash(environment, "freeze_hash")
            and frozen_phase6e_exact
        ),
        "fresh_r06_state_protocol_passed": state["status"] == "PASS",
        "r06_opened_only_after_model_freeze": bool(
            label_open["revision_holdout_labels_opened_after_model_freeze"]
            and label_open["model_freeze_hash"] == model_freeze["freeze_hash"]
        ),
        "experiment_freeze_hash_valid": model_freeze["freeze_hash"]
        == canonical_hash(model_freeze, "freeze_hash"),
        "r06_cache_audit_passed": cache["status"] == "PASS"
        and all(cache["checks"].values()),
        "r06_tabular_feature_audit_passed": features["status"] == "PASS"
        and all(features["checks"].values()),
        "mandatory_sanity_passed": sanity["status"] == "PASS"
        and all(sanity["checks"].values()),
        "three_final_checkpoints_exact": len(checkpoint_manifest["checkpoints"]) == 3
        and checkpoints_exact,
        "r06_evaluation_complete": bool(
            evaluation["status"] == "COMPLETE"
            and evaluation["state_count"] == 8_100
            and evaluation["action_count"] == 191_416
            and evaluation["deployment_checkpoint_selected_before_r06"]
        ),
        "r06_output_hashes_exact": output_hashes_exact,
        "latency_profile_complete_and_passed": bool(
            latency["status"] == "COMPLETE" and latency["latency_gate_passed"]
        ),
        "deployment_gate_passed": gate["status"] == "PASS",
        "required_machine_outputs_present": all(
            path.exists() and path.stat().st_size > 0 for path in required
        ),
        "required_figures_png_pdf_present": all(
            path.exists() and path.stat().st_size > 5_000 for path in figure_paths
        ),
        "report_19_sections_complete": all(value in report_text for value in REPORT_SECTIONS),
        "explicit_conclusions_complete": all(
            f"{key} =" in report_text for key in CONCLUSIONS
        ),
        "partial_files_absent": not partial_files,
        "live_solver_integration_not_implemented": not any(
            path.exists() for path in (
                PHASE / "live_integration",
                PHASE / "solver_integration",
            )
        ),
    }
    status = "PASS" if all(checks.values()) else "FAIL"

    artifact_paths = sorted(set(required + figure_paths + [evaluation_path, regression_path]))
    atomic_json(AUDIT / "artifact_manifest.json", {
        "schema": "phase6f-final-artifact-manifest-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": {
            str(path.relative_to(ROOT)): {
                "bytes": path.stat().st_size,
                "sha256": digest(path),
            }
            for path in artifact_paths if path.exists()
        },
    })
    atomic_json(AUDIT / "repository_audit.json", {
        "schema": "phase6f-repository-audit-v1",
        "audit_status": status,
        "checks": checks,
        "missing_required_outputs": [
            str(path.relative_to(ROOT)) for path in required if not path.exists()
        ],
        "partial_files": partial_files,
    })
    conclusion = gate["explicit_conclusions"]
    atomic_json(AUDIT / "completion_gate.json", {
        "schema": "phase6f-completion-gate-v1",
        "PHASE6F_COMPLETE": status == "PASS",
        "REVISION_HOLDOUT_COMPLETE": checks["r06_evaluation_complete"],
        "MODEL_FROZEN_BEFORE_R06_OPEN": checks["r06_opened_only_after_model_freeze"],
        "THREE_MODEL_SEEDS_COMPLETE": checks["three_final_checkpoints_exact"],
        "NO_LABEL_LEAKAGE_DETECTED": checks["mandatory_sanity_passed"],
        "QUALITY_GATE_U1_PASSED": gate["gate_interpretation"]["U1"] == "PASS",
        "QUALITY_GATE_U2_PASSED": gate["gate_interpretation"]["U2"] == "PASS",
        "LATENCY_GATE_PASSED": conclusion["LATENCY_GATE_PASSED"],
        "REQUIRED_FIGURES_COMPLETE": checks["required_figures_png_pdf_present"],
        "FINAL_REPORT_COMPLETE": checks["report_19_sections_complete"]
        and checks["explicit_conclusions_complete"],
        "FULL_REGRESSION_SUITE_PASSED": checks["regression_commands_passed"],
        "REPOSITORY_AUDIT_PASSED": status == "PASS",
        "PHASE6G_LIVE_INTEGRATION_RECOMMENDED": conclusion["PHASE6G_RECOMMENDATION"]
        == "PROCEED_TO_LIVE_NI_SOLVER_INTEGRATION",
        "LIVE_SOLVER_INTEGRATION_IMPLEMENTED": False,
        "STOP_CONDITION_ENFORCED": True,
        "PHASE6G_RECOMMENDATION": conclusion["PHASE6G_RECOMMENDATION"],
    })
    if status != "PASS":
        failed = {key: value for key, value in checks.items() if not value}
        raise RuntimeError(f"Phase 6F completion audit failed: {failed}")
    print("PHASE6F_COMPLETION_AUDIT_PASS")


if __name__ == "__main__":
    main()
