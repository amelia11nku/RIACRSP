#!/usr/bin/env python3
"""One-run tiny comparison without misrepresenting unsupported Gurobi profiles."""

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
from rcias_clgri.exact.general_gurobi import solve_general_gurobi  # noqa: E402
from rcias_clgri.exact.tiny_exact_solver import solve_tiny_exact  # noqa: E402
from rcias_clgri.ni.live_inference import FrozenLiveInference  # noqa: E402
from rcias_clgri.search.alns import ALNSConfig, solve_alns  # noqa: E402
from rcias_clgri.search.csgni import CSGNIConfig, solve_csgni  # noqa: E402
from rcias_clgri.search.ga import GAConfig, solve_ga  # noqa: E402


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def frozen_config(name: str, cls):
    raw = load_json(ROOT / "configs" / name)
    return cls(**{
        key: value for key, value in raw.items() if key in cls.__dataclass_fields__
    })


def gurobi_probe(instance, time_limit: float) -> dict:
    started = time.perf_counter()
    record = {
        "method": "Gurobi MILP",
        "backend": "gurobi-general-rcias-milp",
        "status": "UNKNOWN",
        "runtime_seconds": None,
        "makespan": None,
        "best_bound": None,
        "gap": None,
        "feasible": None,
        "note": None,
    }
    try:
        import gurobipy as gp
        from gurobipy import GRB

        license_model = gp.Model("phase6g_license_probe")
        license_model.Params.OutputFlag = 0
        value = license_model.addVar(lb=0.0, name="value")
        license_model.setObjective(value, GRB.MINIMIZE)
        license_model.optimize()
        record["gurobi_version"] = ".".join(map(str, gp.gurobi.version()))
        record["license_probe_status"] = int(license_model.Status)
        result = solve_general_gurobi(
            instance, time_limit_seconds=time_limit, seed=670401
        )
        record.update({
            "status": result.status,
            "runtime_seconds": result.runtime_seconds,
            "total_runtime_seconds": result.total_runtime_seconds,
            "makespan": result.replay_makespan,
            "best_bound": result.best_bound,
            "gap": result.gap,
            "feasible": result.replay_feasible,
            "optimality_proven": result.optimality_proven,
            "variable_count": result.variable_count,
            "constraint_count": result.constraint_count,
        })
    except ValueError as error:
        record.update({
            "status": "UNSUPPORTED_PROFILE",
            "runtime_seconds": time.perf_counter() - started,
            "note": str(error),
        })
    except Exception as error:  # license/environment errors are evidence, not algorithm failure
        record.update({
            "status": "ENVIRONMENT_ERROR",
            "runtime_seconds": time.perf_counter() - started,
            "note": f"{type(error).__name__}: {error}",
        })
    return record


def stochastic_record(instance, method: str, result, optimum: float) -> dict:
    audit = check_schedule(instance, result.best.schedule)
    if not audit["feasible"]:
        raise RuntimeError(f"{method} produced an infeasible schedule: {audit['violations']}")
    return {
        "method": method,
        "backend": result.method,
        "status": "FEASIBLE",
        "runtime_seconds": result.runtime,
        "makespan": result.best.makespan,
        "best_bound": optimum,
        "gap": (result.best.makespan - optimum) / optimum,
        "feasible": True,
        "decoder_evaluations": result.decoder_evaluations,
        "iterations": result.iterations,
        "best_found_time": result.best_found_time,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--search-budget", type=float, default=10.0)
    parser.add_argument("--gurobi-time-limit", type=float, default=600.0)
    parser.add_argument("--seed", type=int, default=670401)
    parser.add_argument("--intervention-rate", type=int, default=20)
    args = parser.parse_args()
    instance = load_instance(ROOT / "instances/tiny/tiny_01.json")
    output = ROOT / "outputs/phase6g/exact_validation/tiny_01_comparison"
    output.mkdir(parents=True, exist_ok=True)

    gurobi = gurobi_probe(instance, args.gurobi_time_limit)
    exact = solve_tiny_exact(
        instance,
        time_limit_seconds=args.gurobi_time_limit,
        node_limit=1_000_000,
    )
    if exact.status != "OPTIMAL":
        raise RuntimeError(f"tiny_01 BnB did not prove optimality: {exact.status}")
    exact_audit = check_schedule(instance, exact.schedule)
    if not exact_audit["feasible"]:
        raise RuntimeError("exact tiny_01 schedule failed independent replay")
    optimum = exact.best_value
    exact_record = {
        "method": "Exact BnB",
        "backend": exact.backend,
        "status": exact.status,
        "runtime_seconds": exact.runtime_seconds,
        "makespan": optimum,
        "best_bound": optimum,
        "gap": 0.0,
        "feasible": True,
        "explored_nodes": exact.explored_nodes,
        "note": "Valid exact reference because the existing Gurobi profile does not support tiny_01.",
    }

    alns = solve_alns(
        instance,
        args.search_budget,
        args.seed,
        frozen_config("phase5c_alns.json", ALNSConfig),
    )
    ga = solve_ga(
        instance,
        args.search_budget,
        args.seed,
        frozen_config("phase5c_ga.json", GAConfig),
    )

    phase6g = load_json(ROOT / "configs/phase6g_live_solver.json")
    load_started = time.perf_counter()
    policy = FrozenLiveInference(
        ROOT / phase6g["frozen_phase6f"]["experiment_freeze"],
        device=args.device,
        proposal_seed_namespace=phase6g["rng_namespaces"]["proposal"],
    )
    model_load_seconds = time.perf_counter() - load_started
    observer = Phase6GLiveObserver({
        "instance_id": instance.instance_id,
        "seed": args.seed,
        "scale": "TINY",
        "CF_level": "NA",
        "split": "TINY_DIAGNOSTIC",
        "method": f"CSG-NI-R{args.intervention_rate}",
        "intervention_rate": args.intervention_rate,
    })
    csgni = solve_csgni(
        instance,
        args.search_budget,
        args.seed,
        policy,
        alns_config=frozen_config("phase5c_alns.json", ALNSConfig),
        csgni_config=CSGNIConfig(
            intervention_rate=args.intervention_rate,
            proposal_seed_namespace=phase6g["rng_namespaces"]["proposal"],
            ni_repair_seed_namespace=phase6g["rng_namespaces"]["ni_repair"],
            acceptance_seed_namespace=phase6g["rng_namespaces"]["acceptance"],
            diagnostics_seed_namespace=phase6g["rng_namespaces"]["diagnostics"],
        ),
        observer=observer,
    )
    pd.DataFrame(observer.rows).to_parquet(output / "csgni_live_iterations.parquet", index=False)

    records = [
        gurobi,
        exact_record,
        stochastic_record(instance, "ALNS", alns, optimum),
        stochastic_record(instance, "GA", ga, optimum),
        stochastic_record(instance, f"CSG-NI-R{args.intervention_rate}", csgni, optimum),
    ]
    frame = pd.DataFrame(records)
    frame.to_csv(output / "comparison.csv", index=False)
    payload = {
        "schema": "phase6g-tiny01-one-run-comparison-v1",
        "instance_id": instance.instance_id,
        "stochastic_seed": args.seed,
        "stochastic_search_budget_seconds": args.search_budget,
        "gurobi_time_limit_seconds": args.gurobi_time_limit,
        "csgni_rate": f"R{args.intervention_rate}",
        "csgni_rate_status": "DIAGNOSTIC_PRE_RATE_FREEZE",
        "model_load_seconds_excluded_from_search_budget": model_load_seconds,
        "exact_reference_makespan": optimum,
        "results": records,
        "csgni_diagnostics": csgni.diagnostics,
        "csgni_convergence_trace": [asdict(point) for point in csgni.convergence_trace],
        "interpretation_limit": "One stochastic seed is a diagnostic comparison, not a statistical ranking.",
    }
    (output / "comparison.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(frame[["method", "status", "runtime_seconds", "makespan", "gap", "feasible"]].to_string(index=False))
    print(f"model_load_seconds={model_load_seconds:.6f}")
    print(f"output={output}")


if __name__ == "__main__":
    main()
