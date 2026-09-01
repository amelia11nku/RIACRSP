#!/usr/bin/env python3
"""Resumable Phase 6G DEV-HOLDOUT evaluation on the frozen R02 split."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
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
from rcias_clgri.heuristic.dispatching import solve_dispatching  # noqa: E402
from rcias_clgri.ni.live_inference import FrozenLiveInference  # noqa: E402
from rcias_clgri.search.alns import ALNSConfig, solve_alns  # noqa: E402
from rcias_clgri.search.csgni import CSGNIConfig, solve_csgni  # noqa: E402
from rcias_clgri.search.ga import GAConfig, solve_ga  # noqa: E402


OUT = ROOT / "outputs/phase6g/dev_holdout"
LOG_ROOT = ROOT / "outputs/phase6g/live_logs/dev_holdout"
METHODS = ("H1", "ALNS", "GA", "CSGNI")


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_parquet(temporary, index=False, engine="pyarrow")
    temporary.replace(path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def read_search_config(name: str, cls):
    raw = load_json(ROOT / "configs" / name)
    return cls(**{
        key: value for key, value in raw.items() if key in cls.__dataclass_fields__
    })


def load_selected_rate(config: dict) -> tuple[int, dict]:
    freeze_path = ROOT / "outputs/phase6g/frequency_study/selected_rate_freeze.json"
    if not freeze_path.exists():
        raise RuntimeError("selected intervention rate must be frozen before DEV-HOLDOUT")
    freeze = load_json(freeze_path)
    if freeze.get("status") != "FROZEN_BEFORE_DEV_HOLDOUT":
        raise RuntimeError(f"invalid selected-rate freeze status: {freeze.get('status')}")
    rate = int(freeze["selected_rate"])
    if rate not in config["search"]["intervention_rates"]:
        raise RuntimeError(f"frozen intervention rate is not preregistered: {rate}")
    return rate, freeze


def run_path(method: str, instance_id: str, seed: int | None) -> Path:
    filename = "result.json" if seed is None else f"seed_{seed}.json"
    return OUT / "runs" / method / instance_id / filename


def build_tasks(config: dict, split: pd.DataFrame, selected_rate: int) -> list[dict]:
    tasks: list[dict] = []
    for method in METHODS:
        seeds = (None,) if method == "H1" else tuple(config["seeds"]["DEV_HOLDOUT"])
        for row in split.itertuples(index=False):
            for seed in seeds:
                tasks.append({
                    "method": method,
                    "intervention_rate": selected_rate if method == "CSGNI" else 0,
                    "instance_id": row.instance_id,
                    "instance_relative_path": row.relative_path,
                    "scale": row.scale,
                    "CF_level": row.CF_level,
                    "seed": seed,
                })
    return tasks


def search_payload(
    result,
    *,
    task: dict,
    budget: float,
    feasibility: dict,
    model_load_seconds: float | None,
) -> dict:
    return {
        "schema": "phase6g-dev-holdout-run-v1",
        "status": "COMPLETE",
        **task,
        "time_limit_seconds": budget,
        "best_makespan": result.best.makespan,
        "best_found_time": result.best_found_time,
        "runtime": result.runtime,
        "decoder_evaluations": result.decoder_evaluations,
        "iterations": result.iterations,
        "generations_if_applicable": result.generations_if_applicable,
        "feasible": bool(feasibility["feasible"]),
        "model_load_seconds": model_load_seconds,
        "diagnostics": result.diagnostics,
        "convergence_trace": [asdict(point) for point in result.convergence_trace],
        "best_solution": result.best.schedule.to_dict(),
        "best_actions": [asdict(action) for action in result.best.actions],
    }


def h1_payload(result, *, task: dict, feasibility: dict) -> dict:
    return {
        "schema": "phase6g-dev-holdout-run-v1",
        "status": "COMPLETE",
        **task,
        "time_limit_seconds": None,
        "best_makespan": float(result.objective.makespan),
        "best_found_time": result.runtime_seconds,
        "runtime": result.runtime_seconds,
        "decoder_evaluations": 1,
        "iterations": 0,
        "generations_if_applicable": None,
        "feasible": bool(feasibility["feasible"]),
        "model_load_seconds": None,
        "diagnostics": {},
        "convergence_trace": [{
            "elapsed_time": result.runtime_seconds,
            "decoder_evaluations": 1,
            "current_best_makespan": float(result.objective.makespan),
        }],
        "best_solution": result.schedule.to_dict(),
        "best_actions": [asdict(action) for action in result.actions],
    }


def completed_count(all_tasks: list[dict]) -> int:
    return sum(
        run_path(task["method"], task["instance_id"], task["seed"]).exists()
        for task in all_tasks
    )


def write_progress(all_tasks: list[dict], process_started: float) -> None:
    completed = completed_count(all_tasks)
    elapsed = time.perf_counter() - process_started
    process_rate = completed / elapsed if elapsed > 0 else 0.0
    atomic_json({
        "schema": "phase6g-dev-holdout-progress-v1",
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "completed_runs": completed,
        "total_runs": len(all_tasks),
        "h1_runs": sum(task["method"] == "H1" for task in all_tasks),
        "stochastic_runs": sum(task["method"] != "H1" for task in all_tasks),
        "current_process_naive_throughput_per_hour": process_rate * 3600.0,
        "status": "COMPLETE" if completed == len(all_tasks) else "RUNNING",
    }, OUT / "progress.json")


def summarize(all_tasks: list[dict], selected_rate: int) -> None:
    records = []
    for task in all_tasks:
        path = run_path(task["method"], task["instance_id"], task["seed"])
        if not path.exists():
            continue
        payload = load_json(path)
        records.append({
            **{key: payload[key] for key in (
                "method", "intervention_rate", "instance_id", "scale", "CF_level",
                "seed", "time_limit_seconds", "best_makespan", "best_found_time",
                "runtime", "decoder_evaluations", "iterations", "feasible",
                "model_load_seconds",
            )},
            "ni_interventions": payload["diagnostics"].get("ni_interventions", 0),
            "ni_fallbacks": payload["diagnostics"].get("ni_fallbacks", 0),
            "ni_eligible_iterations": payload["diagnostics"].get("ni_eligible_iterations", 0),
        })
    runs = pd.DataFrame(records)
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_csv(runs, OUT / "dev_holdout_run_results.csv")
    if len(runs) != len(all_tasks):
        return

    rows = []
    for (method, instance_id), part in runs.groupby(["method", "instance_id"], sort=False):
        rows.append({
            "method": method,
            "intervention_rate": selected_rate if method == "CSGNI" else 0,
            "instance_id": instance_id,
            "scale": part.scale.iloc[0],
            "CF_level": part.CF_level.iloc[0],
            "run_count": len(part),
            "best": float(part.best_makespan.min()),
            "mean": float(part.best_makespan.mean()),
            "median": float(part.best_makespan.median()),
            "std": float(part.best_makespan.std(ddof=0)),
            "worst": float(part.best_makespan.max()),
            "mean_runtime": float(part.runtime.mean()),
            "mean_decoder_evaluations": float(part.decoder_evaluations.mean()),
            "feasibility_rate": float(part.feasible.mean()),
        })
    instance_summary = pd.DataFrame(rows)
    atomic_csv(instance_summary, OUT / "dev_holdout_instance_summary.csv")

    means = instance_summary.pivot(index="instance_id", columns="method", values="mean")
    method_rows = []
    for method, part in instance_summary.groupby("method", sort=False):
        method_rows.append({
            "method": method,
            "intervention_rate": selected_rate if method == "CSGNI" else 0,
            "instance_count": len(part),
            "mean_of_instance_means": float(part["mean"].mean()),
            "mean_of_instance_best": float(part["best"].mean()),
            "mean_runtime": float(part.mean_runtime.mean()),
            "mean_decoder_evaluations": float(part.mean_decoder_evaluations.mean()),
            "feasibility_rate": float(part.feasibility_rate.mean()),
            "mean_improvement_over_h1": float(
                ((means["H1"] - means[method]) / means["H1"]).mean()
            ),
            "mean_improvement_over_alns": (
                None if method == "H1" else float(
                    ((means["ALNS"] - means[method]) / means["ALNS"]).mean()
                )
            ),
            "mean_improvement_over_ga": (
                None if method == "H1" else float(
                    ((means["GA"] - means[method]) / means["GA"]).mean()
                )
            ),
        })
    atomic_csv(pd.DataFrame(method_rows), OUT / "dev_holdout_method_summary.csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit-runs", type=int)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()

    config = load_json(ROOT / "configs/phase6g_live_solver.json")
    selected_rate, _ = load_selected_rate(config)
    split = pd.read_csv(ROOT / "outputs/phase6g/environment/dev_split.csv")
    split = split[split.split == "DEV_HOLDOUT"]
    if len(split) != 9 or set(split.replicate) != {"R02"}:
        raise RuntimeError("frozen DEV-HOLDOUT must contain exactly the nine R02 instances")
    all_tasks = build_tasks(config, split, selected_rate)
    selected_tasks = [task for task in all_tasks if task["method"] in args.methods]
    pending = [
        task for task in selected_tasks
        if not run_path(task["method"], task["instance_id"], task["seed"]).exists()
    ]
    if args.limit_runs is not None:
        pending = pending[:args.limit_runs]

    print(
        f"PHASE6G_DEV_HOLDOUT_START pending={len(pending)} total={len(all_tasks)} "
        f"methods={','.join(args.methods)} selected_rate=R{selected_rate}",
        flush=True,
    )
    process_started = time.perf_counter()
    if args.summarize_only:
        summarize(all_tasks, selected_rate)
        write_progress(all_tasks, process_started)
        print("PHASE6G_DEV_HOLDOUT_SUMMARY_RETURNED", flush=True)
        return

    alns_config = read_search_config("phase5c_alns.json", ALNSConfig)
    ga_config = read_search_config("phase5c_ga.json", GAConfig)
    policy = None
    model_load_seconds = None
    for index, task in enumerate(pending, 1):
        instance = load_instance(
            ROOT / "instances/controlled/RCIAS-CB1" / task["instance_relative_path"]
        )
        if task["method"] == "H1":
            result = solve_dispatching(instance, "H1")
            feasibility = check_schedule(instance, result.schedule)
            payload = h1_payload(result, task=task, feasibility=feasibility)
            makespan = result.objective.makespan
            evaluations = 1
            runtime = result.runtime_seconds
        else:
            budget = config["search"]["wall_clock_seconds_per_operation"] * instance.num_operations
            if task["method"] == "ALNS":
                result = solve_alns(instance, budget, task["seed"], alns_config)
            elif task["method"] == "GA":
                result = solve_ga(instance, budget, task["seed"], ga_config)
            else:
                if policy is None:
                    load_started = time.perf_counter()
                    policy = FrozenLiveInference(
                        ROOT / config["frozen_phase6f"]["experiment_freeze"],
                        device=args.device,
                        proposal_seed_namespace=config["rng_namespaces"]["proposal"],
                    )
                    model_load_seconds = time.perf_counter() - load_started
                observer = Phase6GLiveObserver({
                    **{key: task[key] for key in (
                        "instance_id", "seed", "scale", "CF_level", "intervention_rate",
                    )},
                    "split": "DEV_HOLDOUT",
                    "method": "CSGNI",
                })
                result = solve_csgni(
                    instance, budget, task["seed"], policy,
                    alns_config=alns_config,
                    csgni_config=CSGNIConfig(
                        intervention_rate=selected_rate,
                        proposal_seed_namespace=config["rng_namespaces"]["proposal"],
                        ni_repair_seed_namespace=config["rng_namespaces"]["ni_repair"],
                        acceptance_seed_namespace=config["rng_namespaces"]["acceptance"],
                        diagnostics_seed_namespace=config["rng_namespaces"]["diagnostics"],
                    ),
                    observer=observer,
                )
                atomic_parquet(
                    pd.DataFrame(observer.rows),
                    LOG_ROOT / task["instance_id"] / f"seed_{task['seed']}.parquet",
                )
            feasibility = check_schedule(instance, result.best.schedule)
            payload = search_payload(
                result,
                task=task,
                budget=budget,
                feasibility=feasibility,
                model_load_seconds=model_load_seconds if task["method"] == "CSGNI" else None,
            )
            makespan = result.best.makespan
            evaluations = result.decoder_evaluations
            runtime = result.runtime
        if not feasibility["feasible"]:
            raise RuntimeError({"task": task, "violations": feasibility["violations"]})
        atomic_json(payload, run_path(task["method"], task["instance_id"], task["seed"]))
        write_progress(all_tasks, process_started)
        print(
            f"[{index}/{len(pending)}] {task['method']} {task['instance_id']} "
            f"seed={task['seed']} makespan={makespan:g} evals={evaluations} "
            f"runtime={runtime:.2f}s",
            flush=True,
        )

    summarize(all_tasks, selected_rate)
    write_progress(all_tasks, process_started)
    print("PHASE6G_DEV_HOLDOUT_BATCH_RETURNED", flush=True)


if __name__ == "__main__":
    main()
