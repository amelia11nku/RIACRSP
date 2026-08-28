#!/usr/bin/env python3
"""Resumable, externally parallel Phase 5C search experiment runner."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rcias_clgri.data.loader import load_instance
from rcias_clgri.search.alns import ALNSConfig, solve_alns
from rcias_clgri.search.dcga import DCGAConfig, solve_dcga
from rcias_clgri.search.ga import GAConfig, solve_ga


ALGORITHMS = ("GA", "Adapted DCGA", "ALNS-H1")


def _paths():
    return sorted(
        path for path in (ROOT / "instances/canonical/RCIAS-2.0").rglob("*.json")
        if path.name not in {"manifest.json", "generation_config.json"}
    )


def _read_config(name, cls):
    raw = json.loads((ROOT / "configs" / name).read_text())
    allowed = cls.__dataclass_fields__
    return cls(**{key: value for key, value in raw.items() if key in allowed})


def _output_path(algorithm, instance_id, seed, budget_scale, suite="legacy"):
    label = algorithm.lower().replace("-", "_").replace(" ", "_")
    regime = "formal" if budget_scale == 1.0 else f"pilot_{budget_scale:g}x"
    base = ROOT / "outputs/phase5c/search"
    if suite != "legacy":
        base = base / suite
    return base / regime / label / instance_id / f"seed_{seed}.json"


def _run(task):
    algorithm, path_string, seed, budget_scale, taxonomy_hash, suite, metadata = task
    os.environ.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
    instance = load_instance(path_string)
    time_limit = 2.0 * instance.num_operations * budget_scale
    if algorithm == "GA":
        result = solve_ga(instance, time_limit, seed, _read_config("phase5c_ga.json", GAConfig))
    elif algorithm == "Adapted DCGA":
        result = solve_dcga(instance, time_limit, seed, _read_config("phase5c_dcga.json", DCGAConfig))
    else:
        result = solve_alns(instance, time_limit, seed, _read_config("phase5c_alns.json", ALNSConfig))
    output = _output_path(algorithm, instance.instance_id, seed, budget_scale, suite)
    payload = {
        "schema": "phase5c-search-run-v1",
        "algorithm": result.method,
        "suite": suite,
        "scale": metadata.get("scale"),
        "CF_level": metadata.get("CF_level"),
        "RI_level": metadata.get("RI_level"),
        "TI_level": metadata.get("TI_level"),
        "instance_id": instance.instance_id,
        "instance_path": str(Path(path_string).relative_to(ROOT)),
        "seed": seed,
        "time_limit_seconds": time_limit,
        "budget_scale": budget_scale,
        "taxonomy_hash": taxonomy_hash,
        "best_makespan": result.best.makespan,
        "best_found_time": result.best_found_time,
        "runtime": result.runtime,
        "decoder_evaluations": result.decoder_evaluations,
        "iterations": result.iterations,
        "generations_if_applicable": result.generations_if_applicable,
        "feasible": result.best.feasible,
        "compute": {"cpu_threads": 1, "gpu_usage": False, "process_count": 1},
        "best_solution": result.best.schedule.to_dict(),
        "best_actions": [asdict(action) for action in result.best.actions],
        "convergence_trace": [asdict(point) for point in result.convergence_trace],
        "diagnostics": result.diagnostics,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    return str(output.relative_to(ROOT)), result.best.makespan, result.decoder_evaluations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithms", nargs="+", choices=ALGORITHMS, default=list(ALGORITHMS))
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--budget-scale", type=float, default=1.0)
    parser.add_argument("--limit-instances", type=int)
    parser.add_argument("--suite", choices=("legacy", "cb1_core", "cb1_sensitivity"), default="legacy")
    args = parser.parse_args()
    if args.budget_scale <= 0:
        raise ValueError("budget scale must be positive")
    seeds = args.seeds or json.loads((ROOT / "configs/phase5c_baseline_seeds.json").read_text())["seeds"]
    if args.suite == "legacy":
        taxonomy_hash = json.loads((ROOT / "outputs/phase5c/benchmark_audit/taxonomy_freeze.json").read_text())["taxonomy_hash"]
    else:
        taxonomy_hash = json.loads((ROOT / "outputs/phase5c/controlled_benchmark_audit/freeze_record.json").read_text())["freeze_hash"]
    if args.suite == "legacy":
        path_records = [(path, {}) for path in _paths()]
    else:
        import csv
        manifest_name = "core_manifest.csv" if args.suite == "cb1_core" else "sensitivity_manifest.csv"
        with (ROOT / "instances/controlled/RCIAS-CB1/manifests" / manifest_name).open(newline="") as handle:
            manifest_rows = list(csv.DictReader(handle))
        path_records = [(ROOT / "instances/controlled/RCIAS-CB1" / row["relative_path"], row) for row in manifest_rows]
    path_records = path_records[:args.limit_instances]
    tasks = []
    for algorithm in args.algorithms:
        for path, metadata in path_records:
            for seed in seeds:
                output = _output_path(algorithm, path.stem, seed, args.budget_scale, args.suite)
                if not output.exists():
                    tasks.append((algorithm, str(path), seed, args.budget_scale, taxonomy_hash, args.suite, metadata))
    print(f"PHASE5C_SEARCH_START pending={len(tasks)} workers={args.workers} budget_scale={args.budget_scale}", flush=True)
    completed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(_run, task) for task in tasks]
        for future in as_completed(futures):
            path, makespan, evaluations = future.result()
            completed += 1
            print(f"[{completed}/{len(tasks)}] {path} makespan={makespan:g} evals={evaluations}", flush=True)
    print(f"PHASE5C_SEARCH_COMPLETE completed={completed}")


if __name__ == "__main__":
    main()
