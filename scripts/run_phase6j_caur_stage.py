#!/usr/bin/env python3
"""Phase 6J stage boundary validator and frozen command entrypoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/phase6j_caur.json"
COMMANDS_PATH = ROOT / "configs/phase6j_caur_command_manifest.json"
INSTANCE_AUDIT = (
    ROOT
    / "instances/controlled/RCIAS-CB1-TRAIN-CAUR-R12R14/manifests/phase6j_integrity_audit.json"
)
IMPLEMENTED_STAGES = {"preflight", "pilot-cost"}


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
    if args.stage not in IMPLEMENTED_STAGES:
        mode = "execution" if args.execute else "preflight"
        raise RuntimeError(
            f"{args.stage} {mode} is locked until its implementation and stage-specific tests pass"
        )


if __name__ == "__main__":
    sys.exit(main())
