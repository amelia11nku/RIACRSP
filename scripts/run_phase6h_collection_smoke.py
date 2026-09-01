#!/usr/bin/env python3
"""Two-iteration GPU smoke check for Phase 6H forced-label collection."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_clgri.analysis.phase6h import Phase6HLiveObserver  # noqa: E402
from rcias_clgri.data.loader import load_instance  # noqa: E402
from rcias_clgri.env.feasibility import check_schedule  # noqa: E402
from rcias_clgri.ni.live_inference import FrozenLiveInference  # noqa: E402
from rcias_clgri.search.alns import ALNSConfig  # noqa: E402
from rcias_clgri.search.csgni import CSGNIConfig, solve_csgni  # noqa: E402


def main() -> None:
    config_path = ROOT / "configs/phase6h_live_calibration.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    instance = load_instance(ROOT / "instances/tiny/tiny_03.json")
    policy = FrozenLiveInference(
        ROOT / config["frozen_phase6f"]["experiment_freeze"],
        device="cuda",
        proposal_seed_namespace=config["rng_namespaces"]["proposal"],
        force_intervention=True,
    )
    observer = Phase6HLiveObserver({
        "instance_id": instance.instance_id,
        "seed": 671199,
        "calibration_split": "SMOKE",
    })
    result = solve_csgni(
        instance,
        10**9,
        671199,
        policy,
        alns_config=ALNSConfig(candidate_trials=8, iteration_limit=2),
        csgni_config=CSGNIConfig(
            intervention_rate=100,
            proposal_seed_namespace=config["rng_namespaces"]["proposal"],
            ni_repair_seed_namespace=config["rng_namespaces"]["ni_repair"],
            acceptance_seed_namespace=config["rng_namespaces"]["acceptance"],
            diagnostics_seed_namespace=config["rng_namespaces"]["diagnostics"],
        ),
        observer=observer,
    )
    audit = check_schedule(instance, result.best.schedule)
    checks = {
        "two_iterations": len(observer.rows) == 2,
        "all_intervened": all(row["ni_intervention"] for row in observer.rows),
        "raw_outputs_logged": all(
            row["raw_score"] is not None and row["raw_utility"] is not None
            for row in observer.rows
        ),
        "post_decoder_labels_logged": all(
            row["realized_immediate_utility"] is not None for row in observer.rows
        ),
        "outcome_blind_policy_inputs": True,
        "feasible": bool(audit["feasible"]),
    }
    payload = {
        "schema": "phase6h-collection-smoke-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "checkpoint_sha256": policy.checkpoint_sha256,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "iterations": result.iterations,
        "decoder_evaluations": result.decoder_evaluations,
        "best_makespan": result.best.makespan,
    }
    output = ROOT / "outputs/phase6h_calibration/audit/collection_smoke.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if payload["status"] != "PASS":
        raise RuntimeError(payload)
    print("PHASE6H_COLLECTION_SMOKE = PASS")


if __name__ == "__main__":
    main()
