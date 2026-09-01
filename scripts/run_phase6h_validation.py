#!/usr/bin/env python3
"""Resumable unbiased CAL-HOLDOUT solver comparison for Phase 6H."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_clgri.analysis.phase6h import (  # noqa: E402
    Phase6HLiveObserver,
    sample_incumbent_trace,
    validate_incumbent_trace,
)
from rcias_clgri.data.loader import load_instance  # noqa: E402
from rcias_clgri.env.feasibility import check_schedule  # noqa: E402
from rcias_clgri.heuristic.dispatching import solve_dispatching  # noqa: E402
from rcias_clgri.ni.live_inference import FrozenLiveInference  # noqa: E402
from rcias_clgri.search.alns import ALNSConfig, solve_alns  # noqa: E402
from rcias_clgri.search.csgni import CSGNIConfig, solve_csgni  # noqa: E402
from rcias_clgri.search.dcga import DCGAConfig, solve_dcga  # noqa: E402
from rcias_clgri.search.ga import GAConfig, solve_ga  # noqa: E402


CONFIG_PATH = ROOT / "configs/phase6h_live_calibration.json"
OUT = ROOT / "outputs/phase6h_validation"
RUNS = OUT / "runs"
LIVE_LOGS = OUT / "live_logs"
FROZEN_POLICY = ROOT / "outputs/phase6h_calibration/frozen/phase6h_policy.json"
FREEZE_RECORD = ROOT / "outputs/phase6h_calibration/frozen/freeze_record.json"
METHODS = ("H1", "ALNS", "GA", "DCGA", "PHASE6G_CSGNI", "PHASE6H_CSGNI")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_parquet(temporary, index=False, engine="pyarrow")
    temporary.replace(path)


def read_search_config(name: str, cls):
    raw = load_json(ROOT / "configs" / name)
    return cls(**{
        key: value for key, value in raw.items() if key in cls.__dataclass_fields__
    })


def verify_holdout_unlock(config: dict) -> dict:
    if not FREEZE_RECORD.exists() or not FROZEN_POLICY.exists():
        raise RuntimeError("Phase 6H policy must be frozen before CAL-HOLDOUT access")
    record = load_json(FREEZE_RECORD)
    policy = load_json(FROZEN_POLICY)
    if (
        record.get("status") != "FROZEN_BEFORE_CAL_HOLDOUT"
        or policy.get("status") != "FROZEN_BEFORE_CAL_HOLDOUT"
        or record.get("cal_holdout_opened") is not False
        or digest(FROZEN_POLICY) != record.get("policy_sha256")
        or policy.get("checkpoint_sha256")
        != config["frozen_phase6f"]["checkpoint_sha256"]
    ):
        raise RuntimeError("invalid Phase 6H policy freeze record")
    return record


def build_tasks(config: dict, manifest: pd.DataFrame) -> list[dict]:
    holdout = manifest[manifest.calibration_split == "CAL_HOLDOUT"]
    if len(holdout) != 9 or set(holdout.replicate) != {"R08"}:
        raise RuntimeError("CAL-HOLDOUT must be exactly the nine frozen R08 instances")
    tasks = []
    for method in METHODS:
        seeds = (None,) if method == "H1" else config["seeds"]["CAL_HOLDOUT"]
        for row in holdout.itertuples(index=False):
            for seed in seeds:
                tasks.append({
                    "method": method,
                    "instance_id": row.instance_id,
                    "instance_relative_path": row.relative_path,
                    "scale": row.scale,
                    "CF_level": row.CF_level,
                    "RI_level": row.RI_level,
                    "TI_level": row.TI_level,
                    "seed": None if seed is None else int(seed),
                })
    return tasks


def result_path(task: dict) -> Path:
    filename = "result.json" if task["seed"] is None else f"seed_{task['seed']}.json"
    return RUNS / task["method"] / task["instance_id"] / filename


def live_log_path(task: dict) -> Path:
    return (
        LIVE_LOGS / task["method"] / task["instance_id"]
        / f"seed_{task['seed']}.parquet"
    )


def valid_result(task: dict, policy_hash: str) -> dict | None:
    path = result_path(task)
    if not path.exists():
        return None
    try:
        payload = load_json(path)
        if payload.get("status") != "COMPLETE":
            return None
        if task["method"] == "PHASE6H_CSGNI" and payload.get("policy_sha256") != policy_hash:
            return None
        if task["method"] in {"PHASE6G_CSGNI", "PHASE6H_CSGNI"}:
            log = live_log_path(task)
            if not log.exists() or payload.get("live_log_sha256") != digest(log):
                return None
        return payload
    except (json.JSONDecodeError, KeyError, OSError):
        return None


def write_progress(tasks: list[dict], process_started: float, policy_hash: str) -> None:
    completed = sum(valid_result(task, policy_hash) is not None for task in tasks)
    elapsed = time.perf_counter() - process_started
    rate = completed / elapsed if elapsed else 0.0
    atomic_json({
        "schema": "phase6h-cal-holdout-progress-v1",
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "completed_runs": completed,
        "total_runs": len(tasks),
        "completed_by_method": {
            method: sum(
                task["method"] == method and valid_result(task, policy_hash) is not None
                for task in tasks
            )
            for method in METHODS
        },
        "current_process_naive_remaining_seconds": (
            (len(tasks) - completed) / rate if rate else None
        ),
        "status": "COMPLETE" if completed == len(tasks) else "RUNNING",
    }, OUT / "progress.json")


def summarize(tasks: list[dict], policy_hash: str) -> None:
    rows = []
    for task in tasks:
        payload = valid_result(task, policy_hash)
        if payload is None:
            continue
        rows.append({
            **task,
            "time_limit_seconds": payload["time_limit_seconds"],
            "final_best": payload["final_best"],
            "time_to_best": payload["time_to_best"],
            "evals_to_best": payload["evals_to_best"],
            "total_runtime": payload["total_runtime"],
            "total_decoder_evals": payload["total_decoder_evals"],
            "initialization_seconds": payload["initialization_seconds"],
            "iterations": payload["iterations"],
            "feasible": payload["feasible"],
            "model_load_seconds": payload.get("model_load_seconds"),
            "policy_sha256": payload.get("policy_sha256"),
        })
    atomic_csv(pd.DataFrame(rows), OUT / "validation_run_summary.csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit-runs", type=int)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard selection")

    config = load_json(CONFIG_PATH)
    freeze = verify_holdout_unlock(config)
    policy_hash = str(freeze["policy_sha256"])
    manifest = pd.read_csv(ROOT / config["calibration_instances"]["manifest"])
    all_tasks = build_tasks(config, manifest)
    selected = [task for task in all_tasks if task["method"] in args.methods]
    selected = [
        task for index, task in enumerate(selected)
        if index % args.shard_count == args.shard_index
    ]
    process_started = time.perf_counter()
    if args.summarize_only:
        summarize(all_tasks, policy_hash)
        write_progress(all_tasks, process_started, policy_hash)
        print("PHASE6H_VALIDATION_SUMMARY_RETURNED", flush=True)
        return
    pending = [task for task in selected if valid_result(task, policy_hash) is None]
    if args.limit_runs is not None:
        pending = pending[:args.limit_runs]
    print(
        f"PHASE6H_VALIDATION_START pending={len(pending)} total={len(all_tasks)} "
        f"methods={','.join(args.methods)} shard={args.shard_index}/{args.shard_count}",
        flush=True,
    )

    instance_root = ROOT / config["calibration_instances"]["root"]
    alns_config = read_search_config("phase5c_alns.json", ALNSConfig)
    ga_config = read_search_config("phase5c_ga.json", GAConfig)
    dcga_config = read_search_config("phase5c_dcga.json", DCGAConfig)
    csgni_config = CSGNIConfig(
        intervention_rate=int(config["search"]["intervention_rate"]),
        proposal_seed_namespace=int(config["rng_namespaces"]["proposal"]),
        ni_repair_seed_namespace=int(config["rng_namespaces"]["ni_repair"]),
        acceptance_seed_namespace=int(config["rng_namespaces"]["acceptance"]),
        diagnostics_seed_namespace=int(config["rng_namespaces"]["diagnostics"]),
    )
    policies: dict[str, FrozenLiveInference] = {}
    model_load_seconds: dict[str, float] = {}

    for index, task in enumerate(pending, 1):
        instance = load_instance(instance_root / task["instance_relative_path"])
        method = task["method"]
        observer = None
        if method == "H1":
            result = solve_dispatching(instance, "H1")
            final_best = float(result.objective.makespan)
            trace = [{
                "elapsed_time": float(result.runtime_seconds),
                "decoder_evaluations": 1,
                "current_best_makespan": final_best,
            }]
            total_runtime = float(result.runtime_seconds)
            total_evals = 1
            iterations = 0
            initialization_seconds = total_runtime
            best_schedule = result.schedule
            best_actions = result.actions
            diagnostics = {}
            budget = None
        else:
            budget = float(config["search"]["wall_clock_seconds_per_operation"]) * instance.num_operations
            if method == "ALNS":
                result = solve_alns(instance, budget, task["seed"], alns_config)
            elif method == "GA":
                result = solve_ga(instance, budget, task["seed"], ga_config)
            elif method == "DCGA":
                result = solve_dcga(instance, budget, task["seed"], dcga_config)
            else:
                if method not in policies:
                    load_started = time.perf_counter()
                    policies[method] = FrozenLiveInference(
                        ROOT / config["frozen_phase6f"]["experiment_freeze"],
                        device=args.device,
                        proposal_seed_namespace=config["rng_namespaces"]["proposal"],
                        deployment_artifact=(
                            None if method == "PHASE6G_CSGNI" else FROZEN_POLICY
                        ),
                    )
                    model_load_seconds[method] = time.perf_counter() - load_started
                observer = Phase6HLiveObserver({
                    **task,
                    "calibration_split": "CAL_HOLDOUT",
                    "policy_name": policies[method].policy_name,
                })
                result = solve_csgni(
                    instance,
                    budget,
                    task["seed"],
                    policies[method],
                    alns_config=alns_config,
                    csgni_config=csgni_config,
                    observer=observer,
                )
            final_best = result.best.makespan
            trace = validate_incumbent_trace(
                result.convergence_trace, final_best=final_best
            )
            total_runtime = result.runtime
            total_evals = result.decoder_evaluations
            iterations = result.iterations
            initialization_seconds = float(
                result.diagnostics.get("initialization_seconds", trace[0]["elapsed_time"])
            )
            best_schedule = result.best.schedule
            best_actions = result.best.actions
            diagnostics = result.diagnostics
        feasibility = check_schedule(instance, best_schedule)
        if not feasibility["feasible"]:
            raise RuntimeError({"task": task, "violations": feasibility["violations"]})
        if observer is not None:
            log = live_log_path(task)
            atomic_parquet(pd.DataFrame(observer.rows), log)
            live_log_sha256 = digest(log)
        else:
            live_log_sha256 = None
        evals_to_best = int(trace[-1]["decoder_evaluations"])
        payload = {
            "schema": "phase6h-cal-holdout-run-v1",
            "status": "COMPLETE",
            **task,
            "calibration_split": "CAL_HOLDOUT",
            "time_limit_seconds": budget,
            "final_best": final_best,
            "time_to_best": float(trace[-1]["elapsed_time"]),
            "evals_to_best": evals_to_best,
            "total_runtime": total_runtime,
            "total_decoder_evals": total_evals,
            "initialization_seconds": initialization_seconds,
            "iterations": iterations,
            "feasible": True,
            "model_load_seconds": model_load_seconds.get(method),
            "policy_sha256": policy_hash if method == "PHASE6H_CSGNI" else None,
            "checkpoint_sha256": (
                config["frozen_phase6f"]["checkpoint_sha256"]
                if method in {"PHASE6G_CSGNI", "PHASE6H_CSGNI"} else None
            ),
            "live_log_sha256": live_log_sha256,
            "incumbent_trace": trace,
            "normalized_budget_checkpoints": (
                [] if budget is None else sample_incumbent_trace(
                    trace,
                    budget=budget,
                    fractions=config["anytime"]["normalized_budget_fractions"],
                )
            ),
            "diagnostics": diagnostics,
            "best_solution": best_schedule.to_dict(),
            "best_actions": [asdict(action) for action in best_actions],
        }
        atomic_json(payload, result_path(task))
        write_progress(all_tasks, process_started, policy_hash)
        print(
            f"[{index}/{len(pending)}] {method} {task['instance_id']} "
            f"seed={task['seed']} makespan={final_best:g} evals={total_evals} "
            f"runtime={total_runtime:.2f}s",
            flush=True,
        )

    summarize(all_tasks, policy_hash)
    write_progress(all_tasks, process_started, policy_hash)
    print("PHASE6H_VALIDATION_BATCH_RETURNED", flush=True)


if __name__ == "__main__":
    main()
