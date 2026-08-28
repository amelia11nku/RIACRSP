#!/usr/bin/env python3
"""Resumable Phase 6A ALNS trajectory collection on the frozen CB1-DEV set."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys

import pandas as pd
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_clgri.analysis.phase6a import Phase6AObserver
from rcias_clgri.data.loader import load_instance
from rcias_clgri.search.alns import ALNSConfig, solve_alns

CB1 = ROOT / "instances/controlled/RCIAS-CB1"
OUT = ROOT / "outputs/phase6a"
SEEDS = tuple(range(610001, 610011))


def _config(iteration_limit=None):
    raw = json.loads((ROOT / "configs/phase5c_alns.json").read_text())
    allowed = ALNSConfig.__dataclass_fields__
    values = {key: value for key, value in raw.items() if key in allowed}
    values["iteration_limit"] = iteration_limit
    return ALNSConfig(**values)


def _run(task):
    path, metadata, seed, budget_scale, iteration_limit, logging = task
    os.environ.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
    instance = load_instance(path)
    run_id = f"{instance.instance_id}_seed_{seed}"
    run_metadata = {
        "run_id": run_id, "instance_id": instance.instance_id, "suite": "DEV",
        "scale": metadata["scale"], "CF_level": metadata["CF_level"], "seed": seed,
    }
    observer = Phase6AObserver(instance, run_metadata) if logging else None
    process = psutil.Process()
    rss_before = process.memory_info().rss
    time_limit = 2.0 * instance.num_operations * budget_scale
    result = solve_alns(instance, time_limit, seed, _config(iteration_limit), observer)
    rss_after = process.memory_info().rss
    shard = OUT / "raw_logs/shards" / run_id
    shard.mkdir(parents=True, exist_ok=True)
    if observer:
        pd.DataFrame(observer.transitions).to_parquet(shard / "transitions.parquet", index=False)
        pd.DataFrame(observer.targets).to_parquet(shard / "targets.parquet", index=False)
    summary = {
        **run_metadata, "time_limit": time_limit, "budget_scale": budget_scale,
        "logging_enabled": logging, "best_makespan": result.best.makespan,
        "best_found_time": result.best_found_time, "runtime": result.runtime,
        "decoder_evaluations": result.decoder_evaluations, "iterations": result.iterations,
        "feasible": result.best.feasible, "rss_before_bytes": rss_before,
        "rss_after_bytes": rss_after, "rss_delta_bytes": rss_after - rss_before,
        "convergence_trace": json.dumps([asdict(point) for point in result.convergence_trace]),
        "diagnostics": json.dumps(result.diagnostics, sort_keys=True),
    }
    (shard / f"summary_{'on' if logging else 'off'}.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return run_id, result.best.makespan, result.iterations


def _tasks(args):
    with (CB1 / "manifests/dev_manifest.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))[:args.limit_instances]
    seeds = tuple(args.seeds or SEEDS)
    tasks = []
    for row in rows:
        for seed in seeds:
            summary = OUT / "raw_logs/shards" / f"{row['instance_id']}_seed_{seed}" / "summary_on.json"
            if summary.exists() and not args.disable_logging:
                previous = json.loads(summary.read_text())
                if previous["budget_scale"] == args.budget_scale and args.iteration_limit is None:
                    continue
            tasks.append((str(CB1 / row["relative_path"]), row, seed, args.budget_scale,
                          args.iteration_limit, not args.disable_logging))
    return tasks


def combine():
    raw = OUT / "raw_logs"
    transition_paths = sorted(raw.glob("shards/*/transitions.parquet"))
    target_paths = sorted(raw.glob("shards/*/targets.parquet"))
    summaries = [json.loads(path.read_text()) for path in sorted(raw.glob("shards/*/summary_on.json"))]
    if transition_paths:
        pd.concat([pd.read_parquet(path) for path in transition_paths], ignore_index=True).to_parquet(
            raw / "transition_log.parquet", index=False
        )
    if target_paths:
        pd.concat([pd.read_parquet(path) for path in target_paths], ignore_index=True).to_parquet(
            raw / "destroy_target_log.parquet", index=False
        )
    if summaries:
        pd.DataFrame(summaries).to_parquet(raw / "run_summary.parquet", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--budget-scale", type=float, default=1.0)
    parser.add_argument("--limit-instances", type=int)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--iteration-limit", type=int)
    parser.add_argument("--disable-logging", action="store_true")
    parser.add_argument("--combine-only", action="store_true")
    args = parser.parse_args()
    if args.combine_only:
        combine(); return
    tasks = _tasks(args)
    print(f"PHASE6A_START runs={len(tasks)} workers={args.workers}", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(_run, task) for task in tasks]
        for index, future in enumerate(as_completed(futures), 1):
            run_id, makespan, iterations = future.result()
            print(f"[{index}/{len(tasks)}] {run_id} best={makespan:g} iterations={iterations}", flush=True)
    if not args.disable_logging:
        combine()
    print("PHASE6A_RUNS_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
