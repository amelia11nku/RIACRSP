#!/usr/bin/env python3
"""Resumable DEV-TUNE study for the pre-registered R20/R50/R100 rates."""

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
from rcias_clgri.ni.live_inference import FrozenLiveInference  # noqa: E402
from rcias_clgri.search.alns import ALNSConfig, solve_alns  # noqa: E402
from rcias_clgri.search.csgni import CSGNIConfig, solve_csgni  # noqa: E402


OUT = ROOT / "outputs/phase6g/frequency_study"
LOG_ROOT = ROOT / "outputs/phase6g/live_logs/dev_tune"


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


def read_alns_config(config: dict) -> ALNSConfig:
    raw = load_json(ROOT / "configs/phase5c_alns.json")
    return ALNSConfig(**{
        key: value for key, value in raw.items() if key in ALNSConfig.__dataclass_fields__
    })


def run_path(method: str, instance_id: str, seed: int) -> Path:
    return OUT / "runs" / method / instance_id / f"seed_{seed}.json"


def result_payload(result, *, task: dict, budget: float, feasibility: dict, model_load_seconds: float | None) -> dict:
    return {
        "schema": "phase6g-frequency-run-v1",
        "status": "COMPLETE",
        **task,
        "time_limit_seconds": budget,
        "best_makespan": result.best.makespan,
        "best_found_time": result.best_found_time,
        "runtime": result.runtime,
        "decoder_evaluations": result.decoder_evaluations,
        "iterations": result.iterations,
        "feasible": bool(feasibility["feasible"]),
        "model_load_seconds": model_load_seconds,
        "diagnostics": result.diagnostics,
        "convergence_trace": [asdict(point) for point in result.convergence_trace],
        "best_solution": result.best.schedule.to_dict(),
    }


def tasks(config: dict, split: pd.DataFrame) -> list[dict]:
    methods = [("ALNS", 0), *[(f"R{rate}", rate) for rate in config["search"]["intervention_rates"]]]
    return [
        {
            "method": method,
            "intervention_rate": rate,
            "instance_id": row.instance_id,
            "instance_relative_path": row.relative_path,
            "scale": row.scale,
            "CF_level": row.CF_level,
            "seed": int(seed),
        }
        for method, rate in methods
        for row in split.itertuples(index=False)
        for seed in config["seeds"]["DEV_TUNE"]
    ]


def write_progress(all_tasks: list[dict], started: float) -> None:
    completed = sum(
        run_path(task["method"], task["instance_id"], task["seed"]).exists()
        for task in all_tasks
    )
    elapsed = time.perf_counter() - started
    rate = completed / elapsed if elapsed > 0 else 0.0
    remaining_seconds = (len(all_tasks) - completed) / rate if rate > 0 else None
    atomic_json({
        "schema": "phase6g-frequency-progress-v1",
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "completed_runs": completed,
        "total_runs": len(all_tasks),
        "run_throughput_per_hour": rate * 3600.0,
        "estimated_remaining_seconds_from_current_process": remaining_seconds,
        "status": "COMPLETE" if completed == len(all_tasks) else "RUNNING",
    }, OUT / "progress.json")


def summarize(all_tasks: list[dict], config: dict) -> None:
    records = []
    for task in all_tasks:
        path = run_path(task["method"], task["instance_id"], task["seed"])
        if path.exists():
            payload = load_json(path)
            records.append({
                **{key: payload[key] for key in (
                    "method", "intervention_rate", "instance_id", "scale",
                    "CF_level", "seed", "time_limit_seconds", "best_makespan",
                    "best_found_time", "runtime", "decoder_evaluations", "iterations",
                    "feasible",
                )},
                "ni_interventions": payload["diagnostics"].get("ni_interventions", 0),
                "ni_fallbacks": payload["diagnostics"].get("ni_fallbacks", 0),
                "ni_eligible_iterations": payload["diagnostics"].get("ni_eligible_iterations", 0),
            })
    runs = pd.DataFrame(records)
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_csv(runs, OUT / "frequency_study_run_results.csv")
    if len(runs) != len(all_tasks):
        return
    baseline = runs[runs.method == "ALNS"][["instance_id", "seed", "best_makespan"]].rename(
        columns={"best_makespan": "alns_makespan"}
    )
    rows = []
    for method, part in runs[runs.method != "ALNS"].groupby("method", sort=False):
        paired = part.merge(baseline, on=["instance_id", "seed"], validate="one_to_one")
        rows.append({
            "method": method,
            "intervention_rate": int(part.intervention_rate.iloc[0]),
            "run_count": len(part),
            "mean_final_makespan": float(part.best_makespan.mean()),
            "median_final_makespan": float(part.best_makespan.median()),
            "seed_std_makespan": float(part.best_makespan.std(ddof=0)),
            "mean_improvement_over_alns": float(
                ((paired.alns_makespan - paired.best_makespan) / paired.alns_makespan).mean()
            ),
            "mean_decoder_evaluations": float(part.decoder_evaluations.mean()),
            "mean_runtime": float(part.runtime.mean()),
            "feasibility_rate": float(part.feasible.mean()),
            "total_ni_interventions": int(part.ni_interventions.sum()),
            "total_ni_fallbacks": int(part.ni_fallbacks.sum()),
            "intervention_coverage": float(
                part.ni_interventions.sum() / max(part.ni_eligible_iterations.sum(), 1)
            ),
        })
    summary = pd.DataFrame(rows).sort_values([
        "mean_final_makespan", "mean_improvement_over_alns",
        "mean_decoder_evaluations", "seed_std_makespan",
    ], ascending=[True, False, True, True]).reset_index(drop=True)
    summary["selected"] = False
    summary.loc[0, "selected"] = True
    atomic_csv(summary, OUT / "frequency_study_summary.csv")
    selected = summary.iloc[0]
    freeze = load_json(ROOT / "outputs/phase6g/environment/phase6g_environment_freeze.json")
    atomic_json({
        "schema": "phase6g-intervention-rate-freeze-v1",
        "status": "FROZEN_BEFORE_DEV_HOLDOUT",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment_freeze_hash": freeze["freeze_hash"],
        "selection_split": "DEV_TUNE",
        "selection_rule": config["frequency_selection"],
        "selected_rate": int(selected.intervention_rate),
        "selected_method": str(selected.method),
        "selected_mean_final_makespan": float(selected.mean_final_makespan),
        "selected_mean_improvement_over_alns": float(selected.mean_improvement_over_alns),
    }, OUT / "selected_rate_freeze.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit-runs", type=int)
    parser.add_argument(
        "--methods", nargs="+", choices=("ALNS", "R20", "R50", "R100"),
        default=("ALNS", "R20", "R50", "R100"),
    )
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard index must be in [0, shard count)")
    config = load_json(ROOT / "configs/phase6g_live_solver.json")
    split = pd.read_csv(ROOT / "outputs/phase6g/environment/dev_split.csv")
    split = split[split.split == "DEV_TUNE"]
    all_tasks = tasks(config, split)
    selected_tasks = [
        task for index, task in enumerate(all_tasks)
        if task["method"] in args.methods and index % args.shard_count == args.shard_index
    ]
    pending = [
        task for task in selected_tasks
        if not run_path(task["method"], task["instance_id"], task["seed"]).exists()
    ]
    if args.limit_runs is not None:
        pending = pending[:args.limit_runs]
    print(
        f"PHASE6G_FREQUENCY_START pending={len(pending)} total={len(all_tasks)} "
        f"methods={','.join(args.methods)} shard={args.shard_index}/{args.shard_count}",
        flush=True,
    )
    started = time.perf_counter()
    if args.summarize_only:
        summarize(all_tasks, config)
        write_progress(all_tasks, started)
        print("PHASE6G_FREQUENCY_SUMMARY_RETURNED", flush=True)
        return
    policy = None
    model_load_seconds = None
    alns_config = read_alns_config(config)
    for index, task in enumerate(pending, 1):
        instance = load_instance(
            ROOT / "instances/controlled/RCIAS-CB1" / task["instance_relative_path"]
        )
        budget = config["search"]["wall_clock_seconds_per_operation"] * instance.num_operations
        if task["method"] == "ALNS":
            result = solve_alns(instance, budget, task["seed"], alns_config)
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
                "split": "DEV_TUNE",
                "method": task["method"],
            })
            result = solve_csgni(
                instance, budget, task["seed"], policy,
                alns_config=alns_config,
                csgni_config=CSGNIConfig(
                    intervention_rate=task["intervention_rate"],
                    proposal_seed_namespace=config["rng_namespaces"]["proposal"],
                    ni_repair_seed_namespace=config["rng_namespaces"]["ni_repair"],
                    acceptance_seed_namespace=config["rng_namespaces"]["acceptance"],
                    diagnostics_seed_namespace=config["rng_namespaces"]["diagnostics"],
                ),
                observer=observer,
            )
            atomic_parquet(
                pd.DataFrame(observer.rows),
                LOG_ROOT / task["method"] / task["instance_id"] / f"seed_{task['seed']}.parquet",
            )
        feasibility = check_schedule(instance, result.best.schedule)
        if not feasibility["feasible"]:
            raise RuntimeError({"task": task, "violations": feasibility["violations"]})
        atomic_json(
            result_payload(
                result, task=task, budget=budget, feasibility=feasibility,
                model_load_seconds=model_load_seconds if task["method"] != "ALNS" else None,
            ),
            run_path(task["method"], task["instance_id"], task["seed"]),
        )
        write_progress(all_tasks, started)
        print(
            f"[{index}/{len(pending)}] {task['method']} {task['instance_id']} "
            f"seed={task['seed']} makespan={result.best.makespan:g} "
            f"evals={result.decoder_evaluations} runtime={result.runtime:.2f}s",
            flush=True,
        )
    summarize(all_tasks, config)
    write_progress(all_tasks, started)
    print("PHASE6G_FREQUENCY_BATCH_RETURNED", flush=True)


if __name__ == "__main__":
    main()
