#!/usr/bin/env python3
"""Phase 6J stage boundary validator and frozen command entrypoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_clgri.analysis.phase6j_caur import select_shortest_adequate_horizon
from rcias_clgri.data.phase6c_io import atomic_write_json


CONFIG_PATH = ROOT / "configs/phase6j_caur.json"
COMMANDS_PATH = ROOT / "configs/phase6j_caur_command_manifest.json"
INSTANCE_AUDIT = (
    ROOT
    / "instances/controlled/RCIAS-CB1-TRAIN-CAUR-R12R14/manifests/phase6j_integrity_audit.json"
)
PILOT_OUT = ROOT / "outputs/phase6j_caur/r12_pilot"
R12_COLLECTION_SCRIPT = ROOT / "scripts/run_phase6j_caur_collection.py"
R12_FREEZE = ROOT / "outputs/phase6j_caur/frozen/r12_horizon_freeze.json"
IMPLEMENTED_STAGES = {"preflight", "pilot-cost", "freeze-horizon", "r12-collect"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_and_validate() -> tuple[dict, dict]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    commands = json.loads(COMMANDS_PATH.read_text(encoding="utf-8"))
    instance_audit = json.loads(INSTANCE_AUDIT.read_text(encoding="utf-8"))
    manifest_path = ROOT / config["instance_suite"]["manifest"]
    checks = {
        "config_preregistered": config["status"]
        == "PREREGISTERED_BEFORE_R12_PILOT_OR_R13_R14_CONTENT_ACCESS",
        "config_hash_frozen": commands["config_sha256"] == digest(CONFIG_PATH),
        "instance_manifest_hash_frozen": config["instance_suite"]["manifest_sha256"]
        == digest(manifest_path),
        "instance_integrity_pass": instance_audit["status"] == "PASS"
        and all(instance_audit["checks"].values()),
        "r13_locked": not (ROOT / "outputs/phase6j_caur/r13_selection/access_ledger.json").exists(),
        "r14_locked": not (ROOT / "outputs/phase6j_caur/r14_holdout/access_ledger.json").exists(),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Phase 6J preflight failed: {checks}")
    return config, checks


def pilot_cost(config: dict) -> dict:
    pilot = config["r12_pilot"]
    states = int(pilot["expected_states"])
    expected_candidates = states * float(pilot["estimated_unique_candidates_per_state"])
    maximum_candidates = states * int(pilot["maximum_unique_candidates_per_state"])
    seeds = len(config["rng"]["continuation_crn_seeds"])
    horizon = max(int(value) for value in pilot["horizons"])
    trials = int(config["candidate_bank"]["candidate_trials_per_target"])
    expected_decoder_evaluations = expected_candidates * trials * (1 + seeds * horizon)
    maximum_decoder_evaluations = maximum_candidates * trials * (1 + seeds * horizon)
    reference_seconds = 160.3596199430176
    reference_continuation_decoder_evaluations = 216 * 12 * 8
    decoder_ratio = (expected_candidates * seeds * horizon * trials) / reference_continuation_decoder_evaluations
    continuation_seconds = reference_seconds * decoder_ratio
    return {
        "schema": "phase6j-caur-r12-pilot-cost-v1",
        "states": states,
        "expected_unique_candidate_rows": expected_candidates,
        "maximum_unique_candidate_rows": maximum_candidates,
        "continuation_seeds": seeds,
        "horizon_prefixes": pilot["horizons"],
        "maximum_horizon_iterations": horizon,
        "repair_decoder_trials_per_target": trials,
        "expected_total_decoder_evaluations": expected_decoder_evaluations,
        "maximum_total_decoder_evaluations": maximum_decoder_evaluations,
        "reference": {
            "phase6i_mr_states": 27,
            "phase6i_mr_action_continuations": 216,
            "phase6i_mr_h12_decoder_evaluations": reference_continuation_decoder_evaluations,
            "phase6i_mr_observed_continuation_seconds": reference_seconds,
        },
        "estimated_continuation_seconds": continuation_seconds,
        "conservative_end_to_end_eta_minutes": [20, 35],
        "eta_caveat": "historical local timing; refresh after the first completed R12 state",
        "gpu_blocker": "current torch CUDA unavailable; do not launch until device policy is explicitly chosen",
    }


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _r12_pilot_artifacts() -> list[Path]:
    return [
        PILOT_OUT / "progress.json",
        PILOT_OUT / "worker_status.json",
        PILOT_OUT / "full_bank_cost_completeness_audit.json",
        PILOT_OUT / "horizon_decision.json",
        PILOT_OUT / "horizon_comparison.csv",
        PILOT_OUT / "horizon_comparison_by_state.csv",
        PILOT_OUT / "pilot_seed_horizon_labels.parquet",
        PILOT_OUT / "pilot_grouped_horizon_labels.parquet",
        PILOT_OUT / "candidate_source_bias_audit.csv",
        PILOT_OUT / "continuation_vs_immediate_target_report.json",
    ]


def freeze_horizon(config: dict) -> dict:
    """Validate the completed pilot and freeze the only authorized R12 scope."""
    required = _r12_pilot_artifacts()
    if not all(path.is_file() for path in required):
        raise RuntimeError("R12 horizon freeze requires every pilot deliverable")
    if not R12_COLLECTION_SCRIPT.is_file():
        raise RuntimeError("R12 collection implementation is missing")

    progress = _load_json(PILOT_OUT / "progress.json")
    worker = _load_json(PILOT_OUT / "worker_status.json")
    audit = _load_json(PILOT_OUT / "full_bank_cost_completeness_audit.json")
    decision = _load_json(PILOT_OUT / "horizon_decision.json")
    selected_horizon = select_shortest_adequate_horizon({
        int(row["horizon"]): row for row in decision["metrics"]
    })
    integrity_checks = {
        "pilot_progress_complete": progress.get("status") == "COMPLETE"
        and progress.get("states_complete") == progress.get("states_expected") == 27,
        "worker_exit_zero": worker.get("status") == "COMPLETE"
        and worker.get("exit_code") == 0,
        "pilot_completeness_pass": audit.get("status") == "PASS"
        and all(audit.get("checks", {}).values()),
        "horizon_rule_reproduced": decision.get("selected_horizon") == selected_horizon,
        "solver_outcomes_not_used": decision.get("solver_performance_used") is False,
        "r13_r14_unopened": decision.get("r13_accessed") is False
        and decision.get("r14_accessed") is False
        and not (ROOT / "outputs/phase6j_caur/r13_selection/access_ledger.json").exists()
        and not (ROOT / "outputs/phase6j_caur/r14_holdout/access_ledger.json").exists(),
    }
    if not all(integrity_checks.values()):
        raise RuntimeError(f"R12 pilot cannot be frozen: {integrity_checks}")

    raw = pd.read_parquet(PILOT_OUT / "pilot_seed_horizon_labels.parquet")
    selected = raw[raw.horizon.eq(selected_horizon)]
    reference = raw[raw.horizon.eq(max(config["r12_pilot"]["horizons"]))]
    statuses = [
        _load_json(path) for path in sorted((PILOT_OUT / "state_status").glob("*.json"))
    ]
    sources = [
        _load_json(path) for path in sorted((PILOT_OUT / "source_runs").glob("*.json"))
    ]
    expected_states = int(config["r12_collection"]["expected_states"])
    expected_sources = (
        int(config["r12_collection"]["instances"])
        * int(config["r12_collection"]["source_trajectories_per_instance"])
    )
    state_multiplier = expected_states / int(audit["states"])
    source_multiplier = expected_sources / len(sources)
    selected_continuation_seconds = float(selected.candidate_continuation_seconds.sum())
    reference_continuation_seconds = float(reference.candidate_continuation_seconds.sum())
    forced_seconds = sum(float(row["forced_decode_seconds"]) for row in statuses)
    observed_state_seconds = sum(float(row["elapsed_seconds"]) for row in statuses)
    state_overhead_seconds = max(
        0.0, observed_state_seconds - reference_continuation_seconds - forced_seconds
    )
    source_seconds = sum(float(row["runtime_seconds"]) for row in sources)
    projected_elapsed_seconds = (
        (selected_continuation_seconds + forced_seconds + state_overhead_seconds)
        * state_multiplier
        + source_seconds * source_multiplier
    )
    conservative_elapsed_seconds = (
        observed_state_seconds * state_multiplier + source_seconds * source_multiplier
    )
    average_candidates = float(audit["unique_candidate_state_pairs"]) / int(audit["states"])
    seeds = len(config["rng"]["continuation_crn_seeds"])
    trials = int(config["candidate_bank"]["candidate_trials_per_target"])
    projected_decoder_evaluations = round(
        expected_states * average_candidates * trials * (1 + seeds * selected_horizon)
    )
    maximum_decoder_evaluations = (
        expected_states
        * int(config["r12_pilot"]["maximum_unique_candidates_per_state"])
        * trials
        * (1 + seeds * selected_horizon)
    )
    fallback = config["r12_collection"]["fallback_if_cost_excessive"]
    cost_fallback_activated = (
        projected_decoder_evaluations > int(fallback["decoder_evaluation_cap"])
        or conservative_elapsed_seconds / 3600.0 > float(fallback["elapsed_time_cap_hours"])
    )
    if cost_fallback_activated:
        raise RuntimeError(
            "the preregistered cost fallback activated; sampled collection is not implemented"
        )

    artifact_hashes = {
        str(path.relative_to(ROOT)): digest(path) for path in required
    }
    payload = {
        "schema": "phase6j-caur-r12-horizon-freeze-v1",
        "status": "FROZEN_BEFORE_R12_COLLECTION",
        "selected_horizon": selected_horizon,
        "horizon_rule": decision["rule"],
        "collection_scope": "TRUE_FULL_DEDUPLICATED_24_RULE_BANK",
        "cost_fallback_activated": False,
        "r13_accessed": False,
        "r14_accessed": False,
        "config_sha256": digest(CONFIG_PATH),
        "command_manifest_sha256": digest(COMMANDS_PATH),
        "collection_script_sha256": digest(R12_COLLECTION_SCRIPT),
        "freeze_implementation_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip(),
        "pilot_source_implementation_commits": sorted({
            str(row["implementation_commit"]) for row in sources
        }),
        "pilot_label_implementation_commits": sorted({
            str(row["implementation_commit"]) for row in statuses
        }),
        "integrity_checks": integrity_checks,
        "pilot_artifact_sha256": artifact_hashes,
        "observed_pilot": {
            "states": int(audit["states"]),
            "unique_candidate_state_pairs": int(audit["unique_candidate_state_pairs"]),
            "average_unique_candidates_per_state": average_candidates,
            "selected_horizon_continuation_seconds": selected_continuation_seconds,
            "reference_horizon_continuation_seconds": reference_continuation_seconds,
            "forced_decode_seconds": forced_seconds,
            "source_trajectory_seconds": source_seconds,
        },
        "full_collection_projection": {
            "states": expected_states,
            "source_trajectories": expected_sources,
            "projected_decoder_evaluations": projected_decoder_evaluations,
            "maximum_decoder_evaluations": maximum_decoder_evaluations,
            "decoder_evaluation_cap": int(fallback["decoder_evaluation_cap"]),
            "projected_elapsed_seconds": projected_elapsed_seconds,
            "projected_elapsed_hours": projected_elapsed_seconds / 3600.0,
            "conservative_h12_elapsed_seconds": conservative_elapsed_seconds,
            "conservative_h12_elapsed_hours": conservative_elapsed_seconds / 3600.0,
            "elapsed_time_cap_hours": float(fallback["elapsed_time_cap_hours"]),
        },
    }
    if R12_FREEZE.exists():
        existing = _load_json(R12_FREEZE)
        if existing != payload:
            raise RuntimeError("existing R12 horizon freeze differs from recomputed freeze")
    else:
        R12_FREEZE.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(payload, R12_FREEZE)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config, checks = load_and_validate()
    if args.stage == "preflight":
        print(json.dumps({"status": "PASS", "checks": checks}, indent=2, sort_keys=True))
        return
    if args.stage == "pilot-cost":
        print(json.dumps(pilot_cost(config), indent=2, sort_keys=True))
        return
    if args.stage == "freeze-horizon":
        if not args.execute:
            raise RuntimeError("freeze-horizon requires --execute")
        print(json.dumps(freeze_horizon(config), indent=2, sort_keys=True))
        return
    if args.stage == "r12-collect":
        if not args.execute:
            raise RuntimeError("r12-collect requires --execute")
        command = [
            "/home/liulei/miniconda3/envs/gnn311/bin/python",
            "scripts/run_phase6j_caur_collection.py",
            "--device",
            os.environ.get("PHASE6J_DEVICE", "cuda"),
        ]
        raise SystemExit(subprocess.run(command, cwd=ROOT, check=False).returncode)
    if args.stage not in IMPLEMENTED_STAGES:
        mode = "execution" if args.execute else "preflight"
        raise RuntimeError(
            f"{args.stage} {mode} is locked until its implementation and stage-specific tests pass"
        )


if __name__ == "__main__":
    sys.exit(main())
