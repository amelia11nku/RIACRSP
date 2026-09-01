#!/usr/bin/env python3
"""Short production-equivalent smoke test for frozen live CSG-NI."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_clgri.analysis.phase6g import Phase6GLiveObserver  # noqa: E402
from rcias_clgri.data.loader import load_instance  # noqa: E402
from rcias_clgri.env.feasibility import check_schedule  # noqa: E402
from rcias_clgri.ni.live_inference import FrozenLiveInference  # noqa: E402
from rcias_clgri.search.alns import ALNSConfig  # noqa: E402
from rcias_clgri.search.csgni import CSGNIConfig, solve_csgni  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument(
        "--instance",
        default="instances/controlled/RCIAS-CB1/dev/CB1_DEV_S_CF1_R01.json",
    )
    args = parser.parse_args()
    config = json.loads((ROOT / "configs/phase6g_live_solver.json").read_text())
    load_started = time.perf_counter()
    policy = FrozenLiveInference(
        ROOT / config["frozen_phase6f"]["experiment_freeze"],
        device=args.device,
        proposal_seed_namespace=config["rng_namespaces"]["proposal"],
    )
    model_load_seconds = time.perf_counter() - load_started
    instance = load_instance(ROOT / args.instance)
    observer = Phase6GLiveObserver({
        "instance_id": instance.instance_id,
        "seed": 670199,
        "scale": instance.metadata.get("scale", "UNKNOWN"),
        "CF_level": instance.metadata.get("CF_level", "UNKNOWN"),
        "split": "SMOKE",
        "intervention_rate": "R100",
    })
    alns = ALNSConfig(
        initial_temperature=0.05,
        cooling_rate=0.995,
        destroy_fraction=0.15,
        reaction_factor=0.2,
        candidate_trials=8,
        iteration_limit=args.iterations,
    )
    result = solve_csgni(
        instance,
        10**9,
        670199,
        policy,
        alns_config=alns,
        csgni_config=CSGNIConfig(
            intervention_rate=100,
            proposal_seed_namespace=config["rng_namespaces"]["proposal"],
            ni_repair_seed_namespace=config["rng_namespaces"]["ni_repair"],
            acceptance_seed_namespace=config["rng_namespaces"]["acceptance"],
            diagnostics_seed_namespace=config["rng_namespaces"]["diagnostics"],
        ),
        observer=observer,
    )
    feasibility = check_schedule(instance, result.best.schedule)
    if not feasibility["feasible"]:
        raise RuntimeError(feasibility["violations"])
    if not observer.rows or any(row["candidate_bank_size"] < 18 for row in observer.rows):
        raise RuntimeError("live proposal bank did not reach the required 18-target minimum")
    output_root = ROOT / "outputs/phase6g/integration_regression"
    output_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(observer.rows).to_parquet(output_root / "live_smoke_iterations.parquet", index=False)
    payload = {
        "schema": "phase6g-live-smoke-v1",
        "status": "PASS",
        "instance_id": instance.instance_id,
        "device": args.device,
        "model_load_seconds": model_load_seconds,
        "checkpoint_sha256": policy.checkpoint_sha256,
        "iterations": result.iterations,
        "decoder_evaluations": result.decoder_evaluations,
        "initial_makespan": observer.rows[0]["current_makespan"],
        "best_makespan": result.best.makespan,
        "feasible": feasibility["feasible"],
        "diagnostics": result.diagnostics,
        "bank_sizes": [row["candidate_bank_size"] for row in observer.rows],
        "interventions": sum(row["ni_intervention"] for row in observer.rows),
        "fallbacks": sum(row["fallback"] for row in observer.rows),
        "mean_ni_overhead_ms": sum(row["ni_overhead_ms"] for row in observer.rows) / len(observer.rows),
        "best_solution": result.best.schedule.to_dict(),
        "convergence_trace": [asdict(point) for point in result.convergence_trace],
    }
    (output_root / "live_smoke.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({key: payload[key] for key in (
        "status", "instance_id", "model_load_seconds", "iterations",
        "decoder_evaluations", "bank_sizes", "interventions", "fallbacks",
        "mean_ni_overhead_ms", "best_makespan", "feasible",
    )}, indent=2))


if __name__ == "__main__":
    main()
