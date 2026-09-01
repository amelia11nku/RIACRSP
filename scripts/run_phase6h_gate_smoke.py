#!/usr/bin/env python3
"""GPU smoke check for every fitted Phase 6H gate artifact."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_clgri.analysis.phase6h import Phase6HLiveObserver  # noqa: E402
from rcias_clgri.data.loader import load_instance  # noqa: E402
from rcias_clgri.env.feasibility import check_schedule  # noqa: E402
from rcias_clgri.ni.live_inference import FrozenLiveInference  # noqa: E402
from rcias_clgri.search.alns import ALNSConfig  # noqa: E402
from rcias_clgri.search.csgni import CSGNIConfig, solve_csgni  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    config = json.loads(
        (ROOT / "configs/phase6h_live_calibration.json").read_text(encoding="utf-8")
    )
    manifest = pd.read_csv(
        ROOT / "outputs/phase6h_calibration/calibration/candidate_policy_manifest.csv"
    )
    expected = set(config["intervention_gate_study"]["candidates"])
    if set(manifest.policy_name) != expected:
        raise RuntimeError("gate-smoke candidate set differs from preregistration")
    instance = load_instance(ROOT / "instances/tiny/tiny_03.json")
    records = []
    for index, row in enumerate(manifest.itertuples(index=False)):
        artifact = ROOT / row.relative_path
        if digest(artifact) != row.sha256:
            raise RuntimeError(f"candidate hash mismatch: {row.policy_name}")
        policy = FrozenLiveInference(
            ROOT / config["frozen_phase6f"]["experiment_freeze"],
            device="cuda",
            proposal_seed_namespace=config["rng_namespaces"]["proposal"],
            deployment_artifact=artifact,
        )
        observer = Phase6HLiveObserver({
            "instance_id": instance.instance_id,
            "seed": 671190 + index,
            "calibration_split": "GATE_SMOKE",
        })
        result = solve_csgni(
            instance,
            10**9,
            671190 + index,
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
            "artifact_loaded": policy.deployment_artifact_sha256 == row.sha256,
            "policy_name_exact": set(item["policy_name"] for item in observer.rows)
            == {row.policy_name},
            "two_iterations": len(observer.rows) == 2,
            "raw_outputs_present": all(
                item["raw_score"] is not None and item["raw_utility"] is not None
                for item in observer.rows
            ),
            "final_schedule_feasible": bool(audit["feasible"]),
        }
        records.append({
            "policy_name": row.policy_name,
            "checks": checks,
            "interventions": result.diagnostics["ni_interventions"],
            "fallbacks": result.diagnostics["ni_fallbacks"],
            "best_makespan": result.best.makespan,
            "status": "PASS" if all(checks.values()) else "FAIL",
        })
    payload = {
        "schema": "phase6h-gate-artifact-smoke-v1",
        "status": "PASS" if all(row["status"] == "PASS" for row in records) else "FAIL",
        "records": records,
        "cal_holdout_opened": False,
    }
    output = ROOT / "outputs/phase6h_calibration/audit/gate_artifact_smoke.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if payload["status"] != "PASS":
        raise RuntimeError(payload)
    print("PHASE6H_GATE_ARTIFACT_SMOKE = PASS")


if __name__ == "__main__":
    main()
