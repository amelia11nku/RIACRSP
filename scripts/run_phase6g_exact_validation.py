#!/usr/bin/env python3
"""Resumable selected-policy tiny and DEV-Small exact validation for Phase 6G."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
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
from rcias_clgri.heuristic.dispatching import solve_dispatching  # noqa: E402
from rcias_clgri.ni.live_inference import FrozenLiveInference  # noqa: E402
from rcias_clgri.search.alns import ALNSConfig, solve_alns  # noqa: E402
from rcias_clgri.search.csgni import CSGNIConfig, solve_csgni  # noqa: E402
from rcias_clgri.search.ga import GAConfig, solve_ga  # noqa: E402


PHASE = ROOT / "outputs/phase6g"
OUT = PHASE / "exact_validation"
GUROBI = PHASE / "gurobi"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def frozen_config(name: str, cls):
    raw = load_json(ROOT / "configs" / name)
    return cls(**{
        key: value for key, value in raw.items() if key in cls.__dataclass_fields__
    })


def save_gurobi(instance, protocol: dict, suite: str) -> dict:
    path = GUROBI / "runs" / suite / f"{instance.instance_id}.json"
    if path.exists():
        return load_json(path)
    record = {
        "schema": "phase6g-gurobi-run-v1",
        "suite": suite,
        "instance_id": instance.instance_id,
        "solver": "gurobi-general-rcias-milp",
        "time_limit_seconds": protocol["gurobi_time_limit_seconds"],
        "status": "ENVIRONMENT_ERROR",
        "runtime_seconds": None,
        "total_runtime_seconds": None,
        "incumbent": None,
        "lower_bound": None,
        "mip_gap": None,
        "optimality_proven": False,
        "replay_feasible": False,
    }
    started = time.perf_counter()
    try:
        result = solve_general_gurobi(
            instance,
            time_limit_seconds=float(protocol["gurobi_time_limit_seconds"]),
            seed=int(protocol["gurobi_seed"]),
            threads=int(protocol["gurobi_threads"]),
        )
        record.update({
            "status": result.status,
            "solver_version": result.solver_version,
            "runtime_seconds": result.runtime_seconds,
            "total_runtime_seconds": result.total_runtime_seconds,
            "incumbent": result.replay_makespan,
            "native_incumbent": result.solver_makespan,
            "lower_bound": result.best_bound,
            "mip_gap": result.gap,
            "optimality_proven": result.optimality_proven,
            "replay_feasible": result.replay_feasible,
            "variable_count": result.variable_count,
            "constraint_count": result.constraint_count,
            "h1_upper_bound": result.h1_upper_bound,
            "result": result.to_dict(),
        })
    except Exception as error:
        record.update({
            "total_runtime_seconds": time.perf_counter() - started,
            "error": f"{type(error).__name__}: {error}",
        })
    atomic_json(record, path)
    print(
        f"GUROBI {instance.instance_id} status={record['status']} "
        f"incumbent={record['incumbent']} bound={record['lower_bound']}", flush=True,
    )
    return record


def run_tiny(protocol: dict) -> None:
    phase_config = load_json(ROOT / "configs/phase6g_live_solver.json")
    selected = load_json(PHASE / "frequency_study/selected_rate_freeze.json")
    rate = int(selected["selected_rate"])
    if rate != int(protocol["selected_intervention_rate"]):
        raise RuntimeError("exact protocol disagrees with selected-rate freeze")
    alns_config = frozen_config("phase5c_alns.json", ALNSConfig)
    ga_config = frozen_config("phase5c_ga.json", GAConfig)
    policy = None
    for instance_id in protocol["tiny_instances"]:
        instance = load_instance(ROOT / "instances/tiny" / f"{instance_id}.json")
        h1_path = OUT / "runs/tiny" / instance_id / "H1/result.json"
        if not h1_path.exists():
            h1 = solve_dispatching(instance, "H1")
            audit = check_schedule(instance, h1.schedule)
            if not audit["feasible"]:
                raise RuntimeError(f"H1 infeasible on {instance_id}")
            atomic_json({
                "schema": "phase6g-exact-heuristic-run-v1",
                "instance_id": instance_id,
                "method": "H1",
                "seed": None,
                "runtime_seconds": h1.runtime_seconds,
                "makespan": h1.objective.makespan,
                "feasible": True,
            }, h1_path)
        save_gurobi(instance, protocol, "tiny")
        if instance_id == "tiny_01":
            exact_path = OUT / "runs/tiny" / instance_id / "Exact_BnB/result.json"
            if not exact_path.exists():
                exact = solve_tiny_exact(
                    instance,
                    time_limit_seconds=float(protocol["gurobi_time_limit_seconds"]),
                    node_limit=1_000_000,
                )
                audit = check_schedule(instance, exact.schedule)
                if exact.status != "OPTIMAL" or not audit["feasible"]:
                    raise RuntimeError("tiny_01 exact BnB reference failed")
                atomic_json({
                    "schema": "phase6g-exact-reference-run-v1",
                    "instance_id": instance_id,
                    "method": "Exact BnB",
                    "status": exact.status,
                    "runtime_seconds": exact.runtime_seconds,
                    "incumbent": exact.best_value,
                    "lower_bound": exact.best_value,
                    "mip_gap": 0.0,
                    "optimality_proven": True,
                    "replay_feasible": True,
                    "explored_nodes": exact.explored_nodes,
                }, exact_path)
        for method in ("ALNS", "GA", "CSGNI"):
            for seed in protocol["tiny_stochastic_seeds"]:
                path = OUT / "runs/tiny" / instance_id / method / f"seed_{seed}.json"
                if path.exists():
                    continue
                budget = float(protocol["tiny_stochastic_budget_seconds"])
                if method == "ALNS":
                    result = solve_alns(instance, budget, seed, alns_config)
                    observer = None
                elif method == "GA":
                    result = solve_ga(instance, budget, seed, ga_config)
                    observer = None
                else:
                    if policy is None:
                        policy = FrozenLiveInference(
                            ROOT / phase_config["frozen_phase6f"]["experiment_freeze"],
                            device="cuda",
                            proposal_seed_namespace=phase_config["rng_namespaces"]["proposal"],
                        )
                    observer = Phase6GLiveObserver({
                        "instance_id": instance_id,
                        "seed": seed,
                        "scale": "TINY",
                        "CF_level": "NA",
                        "split": "TINY_EXACT_VALIDATION",
                        "method": "CSGNI",
                        "intervention_rate": rate,
                    })
                    result = solve_csgni(
                        instance, budget, seed, policy,
                        alns_config=alns_config,
                        csgni_config=CSGNIConfig(
                            intervention_rate=rate,
                            proposal_seed_namespace=phase_config["rng_namespaces"]["proposal"],
                            ni_repair_seed_namespace=phase_config["rng_namespaces"]["ni_repair"],
                            acceptance_seed_namespace=phase_config["rng_namespaces"]["acceptance"],
                            diagnostics_seed_namespace=phase_config["rng_namespaces"]["diagnostics"],
                        ), observer=observer,
                    )
                audit = check_schedule(instance, result.best.schedule)
                if not audit["feasible"]:
                    raise RuntimeError(f"{method} infeasible on {instance_id}, seed {seed}")
                atomic_json({
                    "schema": "phase6g-exact-heuristic-run-v1",
                    "instance_id": instance_id,
                    "method": method,
                    "seed": seed,
                    "time_limit_seconds": budget,
                    "runtime_seconds": result.runtime,
                    "makespan": result.best.makespan,
                    "feasible": True,
                    "decoder_evaluations": result.decoder_evaluations,
                    "iterations": result.iterations,
                    "convergence_trace": [asdict(point) for point in result.convergence_trace],
                    "diagnostics": result.diagnostics,
                }, path)
                if observer is not None:
                    log_path = PHASE / "live_logs/exact_validation" / instance_id / f"seed_{seed}.parquet"
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    pd.DataFrame(observer.rows).to_parquet(log_path, index=False)
                print(
                    f"TINY {instance_id} {method} seed={seed} "
                    f"makespan={result.best.makespan:g}", flush=True,
                )


def run_small(protocol: dict) -> None:
    for case in protocol["additional_dev_small_cases"]:
        instance = load_instance(ROOT / "instances/controlled/RCIAS-CB1" / case["relative_path"])
        if instance.num_operations != int(case["number_of_operations"]):
            raise RuntimeError(f"operation count mismatch for {instance.instance_id}")
        save_gurobi(instance, protocol, "dev_small")


def summarize(protocol: dict) -> None:
    gurobi_records = []
    for path in sorted((GUROBI / "runs").rglob("*.json")):
        item = load_json(path)
        gurobi_records.append({key: item.get(key) for key in (
            "suite", "instance_id", "solver", "status", "time_limit_seconds",
            "runtime_seconds", "total_runtime_seconds", "incumbent", "lower_bound",
            "mip_gap", "optimality_proven", "replay_feasible", "variable_count",
            "constraint_count", "h1_upper_bound", "error",
        )})
    gurobi_frame = pd.DataFrame(gurobi_records)
    atomic_csv(gurobi_frame, GUROBI / "gurobi_results.csv")

    references = {}
    for row in gurobi_records:
        if row["optimality_proven"]:
            references[row["instance_id"]] = float(row["incumbent"])
    bnb_path = OUT / "runs/tiny/tiny_01/Exact_BnB/result.json"
    if bnb_path.exists():
        references["tiny_01"] = float(load_json(bnb_path)["incumbent"])

    rows = []
    for path in sorted((OUT / "runs/tiny").rglob("*.json")):
        item = load_json(path)
        if item.get("method") not in {"H1", "ALNS", "GA", "CSGNI"}:
            continue
        rows.append({
            "suite": "tiny",
            "instance_id": item["instance_id"],
            "method": item["method"],
            "seed": item.get("seed"),
            "status": "FEASIBLE",
            "runtime_seconds": item["runtime_seconds"],
            "feasible_makespan": item["makespan"],
            "lower_bound": references.get(item["instance_id"]),
            "optimality_proven": item["instance_id"] in references,
        })
    holdout = pd.read_csv(PHASE / "dev_holdout/dev_holdout_instance_summary.csv")
    for case in protocol["additional_dev_small_cases"]:
        part = holdout[holdout.instance_id == case["instance_id"]]
        for row in part.itertuples(index=False):
            rows.append({
                "suite": "dev_small",
                "instance_id": row.instance_id,
                "method": row.method,
                "seed": None,
                "status": "FEASIBLE_BEST_OF_REGISTERED_RUNS",
                "runtime_seconds": row.mean_runtime,
                "feasible_makespan": row.best,
                "lower_bound": references.get(row.instance_id),
                "optimality_proven": row.instance_id in references,
            })
    comparison = pd.DataFrame(rows)
    comparison["gap_to_optimum"] = comparison.apply(
        lambda row: (
            (row.feasible_makespan - row.lower_bound) / row.lower_bound
            if row.optimality_proven and pd.notna(row.lower_bound) else None
        ), axis=1,
    )
    incumbent_by_instance = {
        row["instance_id"]: row["incumbent"] for row in gurobi_records
        if row["incumbent"] is not None
    }
    lower_by_instance = {
        row["instance_id"]: row["lower_bound"] for row in gurobi_records
        if row["lower_bound"] is not None
    }
    comparison["gap_to_gurobi_incumbent"] = comparison.apply(
        lambda row: (
            (row.feasible_makespan - incumbent_by_instance[row.instance_id])
            / incumbent_by_instance[row.instance_id]
            if row.instance_id in incumbent_by_instance else None
        ), axis=1,
    )
    comparison["gap_to_gurobi_lower_bound"] = comparison.apply(
        lambda row: (
            (row.feasible_makespan - lower_by_instance[row.instance_id])
            / lower_by_instance[row.instance_id]
            if row.instance_id in lower_by_instance else None
        ), axis=1,
    )
    atomic_csv(comparison, OUT / "exact_solver_comparison.csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("tiny", "small", "summarize", "all"), default="all")
    args = parser.parse_args()
    protocol = load_json(OUT / "exact_protocol_freeze.json")
    if protocol.get("status") != "FROZEN_BEFORE_SELECTED_POLICY_EXACT_VALIDATION":
        raise RuntimeError("exact validation protocol is not frozen")
    if args.stage in {"tiny", "all"}:
        run_tiny(protocol)
    if args.stage in {"small", "all"}:
        run_small(protocol)
    summarize(protocol)
    print(f"PHASE6G_EXACT_VALIDATION_{args.stage.upper()}_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
