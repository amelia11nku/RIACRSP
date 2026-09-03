#!/usr/bin/env python3
"""Resumable Core45 runner for the provisional Phase 6H CSG-NI."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_clgri.analysis.phase6h import (  # noqa: E402
    Phase6HLiveObserver,
    sample_incumbent_trace,
    validate_incumbent_trace,
)
from rcias_clgri.data.loader import load_instance  # noqa: E402
from rcias_clgri.env.feasibility import check_schedule  # noqa: E402
from rcias_clgri.ni.live_inference import FrozenLiveInference  # noqa: E402
from rcias_clgri.search.alns import ALNSConfig  # noqa: E402
from rcias_clgri.search.csgni import CSGNIConfig, solve_csgni  # noqa: E402


ALGORITHM = "CSG_NI_PROVISIONAL_PHASE6H"
CONFIG_PATH = ROOT / "paper_experiments/configs/main_core/phase6h_provisional.json"
PROTOCOL_PATH = ROOT / "paper_experiments/configs/main_core/protocol.json"
IMPLEMENTATION_MANIFEST = (
    ROOT / "paper_experiments/configs/main_core/phase6h_provisional_implementation.json"
)
CORE_ROOT = ROOT / "instances/controlled/RCIAS-CB1"
OUTPUT_ROOT = ROOT / "paper_experiments/raw_results/core45/CSG_NI_PROVISIONAL_PHASE6H"
RUNS = OUTPUT_ROOT / "runs"
LIVE_LOGS = OUTPUT_ROOT / "live_logs"
PROGRESS = OUTPUT_ROOT / "progress.json"
PROGRESS_LOCK = OUTPUT_ROOT / ".progress.lock"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_parquet(temporary, index=False, engine="pyarrow")
    temporary.replace(path)


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        text=True, capture_output=True,
    ).stdout.strip()


def cpu_model() -> str:
    for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
        if line.lower().startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor() or "UNKNOWN"


def verify_file(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"missing {label}: {path}")
    observed = digest(path)
    if observed != expected:
        raise RuntimeError(f"{label} hash mismatch: expected={expected} observed={observed}")


def load_and_verify_protocol() -> tuple[dict, dict, str, str]:
    config = load_json(CONFIG_PATH)
    protocol = load_json(PROTOCOL_PATH)
    config_hash = digest(CONFIG_PATH)
    protocol_hash = digest(PROTOCOL_PATH)
    if config.get("status") != "FROZEN_BEFORE_PROVISIONAL_PHASE6H_CORE":
        raise RuntimeError("provisional Phase 6H Core config is not frozen")
    if config.get("algorithm") != ALGORITHM or config.get("experiment_status") != "PROVISIONAL":
        raise RuntimeError("provisional algorithm identity changed")
    if protocol_hash != config["core_protocol_sha256"]:
        raise RuntimeError("Core protocol hash mismatch")
    if protocol.get("primary_seeds") != config["seeds"] or protocol.get("supplementary_seeds"):
        raise RuntimeError("Core seed protocol mismatch")

    for key, label in (
        ("instance_manifest", "Core instance manifest"),
        ("instance_checksums", "Core checksum manifest"),
        ("seed_manifest", "seed manifest"),
        ("phase6h_policy", "Phase 6H policy"),
        ("phase6h_freeze_record", "Phase 6H freeze record"),
        ("phase6f_experiment_freeze", "Phase 6F experiment freeze"),
        ("phase6h_config", "Phase 6H config"),
        ("alns_config", "ALNS config"),
    ):
        verify_file(ROOT / config[key], config[f"{key}_sha256"], label)
    verify_file(
        IMPLEMENTATION_MANIFEST,
        config["implementation_manifest_sha256"],
        "provisional implementation manifest",
    )
    implementation = load_json(IMPLEMENTATION_MANIFEST)
    if implementation.get("status") != "FROZEN_BEFORE_PROVISIONAL_PHASE6H_CORE":
        raise RuntimeError("implementation manifest is not frozen")
    for record in implementation["files"]:
        verify_file(ROOT / record["path"], record["sha256"], record["path"])

    seed_manifest = load_json(ROOT / config["seed_manifest"])
    preregistered_seeds = seed_manifest.get("seeds")
    selected_count = len(config["seeds"])
    if preregistered_seeds[:selected_count] != config["seeds"]:
        raise RuntimeError("selected seeds are not the frozen preregistered prefix")
    if protocol.get("excluded_available_seeds") != preregistered_seeds[selected_count:]:
        raise RuntimeError("excluded preregistered seed suffix changed")
    freeze = load_json(ROOT / config["phase6h_freeze_record"])
    policy = load_json(ROOT / config["phase6h_policy"])
    experiment_freeze = load_json(ROOT / config["phase6f_experiment_freeze"])
    if not all([
        freeze.get("status") == "FROZEN_BEFORE_CAL_HOLDOUT",
        freeze.get("policy_sha256") == config["phase6h_policy_sha256"],
        policy.get("status") == "FROZEN_BEFORE_CAL_HOLDOUT",
        policy.get("checkpoint_sha256") == config["checkpoint_sha256"],
        experiment_freeze.get("selected_checkpoint_sha256") == config["checkpoint_sha256"],
    ]):
        raise RuntimeError("Phase 6H frozen artifact chain is invalid")
    return config, protocol, config_hash, digest(IMPLEMENTATION_MANIFEST)


def build_tasks(config: dict) -> list[dict]:
    manifest_path = ROOT / config["instance_manifest"]
    checksums_path = ROOT / config["instance_checksums"]
    checksums = {}
    for line in checksums_path.read_text(encoding="utf-8").splitlines():
        checksum, relative_path = line.split(maxsplit=1)
        checksums[relative_path] = checksum
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 45 or len({row["instance_id"] for row in rows}) != 45:
        raise RuntimeError("Core manifest must contain 45 unique instances")
    tasks = []
    for row in rows:
        relative_path = row["relative_path"]
        if relative_path not in checksums:
            raise RuntimeError(f"missing frozen instance checksum: {relative_path}")
        instance_path = CORE_ROOT / relative_path
        verify_file(instance_path, checksums[relative_path], row["instance_id"])
        for seed in config["seeds"]:
            tasks.append({
                "instance_id": row["instance_id"],
                "instance_relative_path": relative_path,
                "instance_sha256": checksums[relative_path],
                "scale": row["scale"],
                "CF_level": row["CF_level"],
                "seed": int(seed),
                "number_of_operations": int(row["number_of_operations"]),
            })
    expected_records = 45 * len(config["seeds"])
    if len(tasks) != expected_records or config["expected_records"] != expected_records:
        raise RuntimeError(
            f"provisional Phase 6H Core task count must be {expected_records}"
        )
    return tasks


def result_path(task: dict) -> Path:
    return RUNS / task["instance_id"] / f"seed_{task['seed']}.json"


def live_log_path(task: dict) -> Path:
    return LIVE_LOGS / task["instance_id"] / f"seed_{task['seed']}.parquet"


def partition_tasks(tasks: list[dict], shard_count: int, shard_index: int) -> list[dict]:
    if shard_count < 1:
        raise ValueError("shard-count must be positive")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("shard-index must satisfy 0 <= index < shard-count")
    return [task for index, task in enumerate(tasks) if index % shard_count == shard_index]


def validate_existing_result(
    task: dict,
    config: dict,
    config_hash: str,
    protocol_hash: str,
    implementation_hash: str,
) -> dict | None:
    path = result_path(task)
    if not path.exists():
        return None
    try:
        payload = load_json(path)
        log = live_log_path(task)
        expected_limit = float(config["wall_clock_seconds_per_operation"]) * task["number_of_operations"]
        checks = [
            payload.get("schema") == "initial-manuscript-core-run-v1",
            payload.get("status") == "COMPLETE",
            payload.get("algorithm") == ALGORITHM,
            payload.get("algorithm_version") == "PHASE6H_FROZEN_POLICY",
            payload.get("experiment_status") == "PROVISIONAL",
            payload.get("instance_id") == task["instance_id"],
            payload.get("instance_sha256") == task["instance_sha256"],
            payload.get("seed") == task["seed"],
            payload.get("time_limit_seconds") == expected_limit,
            payload.get("phase6h_policy_sha256") == config["phase6h_policy_sha256"],
            payload.get("checkpoint_sha256") == config["checkpoint_sha256"],
            payload.get("config_sha256") == config_hash,
            payload.get("core_protocol_sha256") == protocol_hash,
            payload.get("implementation_manifest_sha256") == implementation_hash,
            payload.get("feasible") is True,
            payload.get("independent_feasibility_audit", {}).get("feasible") is True,
            log.is_file(),
            payload.get("live_log_sha256") == (digest(log) if log.is_file() else None),
        ]
        if not all(checks):
            raise RuntimeError(f"existing result failed integrity checks: {path}")
        return payload
    except (json.JSONDecodeError, OSError, TypeError, KeyError) as error:
        raise RuntimeError(f"invalid existing result {path}: {error}") from error


def write_progress(
    tasks: list[dict],
    started: float,
    shard_count: int,
    shard_index: int,
) -> None:
    PROGRESS_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with PROGRESS_LOCK.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        completed = {
            (task["instance_id"], task["seed"])
            for task in tasks
            if result_path(task).is_file()
        }
        count = len(completed)
        atomic_json({
            "schema": "initial-manuscript-phase6h-core-progress-v2",
            "status": "COMPLETE" if count == len(tasks) else "RUNNING",
            "algorithm": ALGORITHM,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "completed_runs": count,
            "total_runs": len(tasks),
            "completed_by_scale": {
                scale: sum(
                    (task["instance_id"], task["seed"]) in completed
                    and task["scale"] == scale
                    for task in tasks
                )
                for scale in ("S", "M", "L")
            },
            "configured_solver_concurrency": shard_count,
            "last_updating_shard": shard_index,
            "current_worker_elapsed_seconds": time.perf_counter() - started,
        }, PROGRESS)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit-runs", type=int)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    if args.limit_runs is not None and args.limit_runs < 1:
        raise ValueError("limit-runs must be positive")

    config, protocol, config_hash, implementation_hash = load_and_verify_protocol()
    configured_concurrency = int(config["solver_concurrency"])
    if args.shard_count != configured_concurrency:
        raise RuntimeError(
            f"shard-count {args.shard_count} != frozen concurrency {configured_concurrency}"
        )
    if int(config["gpu_workers"]) != configured_concurrency:
        raise RuntimeError("frozen GPU worker count does not match solver concurrency")
    protocol_hash = digest(PROTOCOL_PATH)
    tasks = build_tasks(config)
    process_started = time.perf_counter()
    completed: set[tuple[str, int]] = set()
    for task in tasks:
        if validate_existing_result(
            task, config, config_hash, protocol_hash, implementation_hash
        ) is not None:
            completed.add((task["instance_id"], task["seed"]))
    assigned = partition_tasks(tasks, args.shard_count, args.shard_index)
    pending = [
        task for task in assigned
        if (task["instance_id"], task["seed"]) not in completed
    ]
    write_progress(tasks, process_started, args.shard_count, args.shard_index)
    print(
        f"PAPER_PHASE6H_CORE_START shard={args.shard_index}/{args.shard_count} "
        f"pending={len(pending)} assigned={len(assigned)} total={len(tasks)} "
        f"device={args.device} algorithm={ALGORITHM}",
        flush=True,
    )
    if args.audit_only:
        print("PAPER_PHASE6H_CORE_AUDIT_RETURNED", flush=True)
        return
    if args.limit_runs is not None:
        pending = pending[:args.limit_runs]

    phase6h_config = load_json(ROOT / config["phase6h_config"])
    alns_raw = load_json(ROOT / config["alns_config"])
    alns_config = ALNSConfig(**{
        key: value for key, value in alns_raw.items()
        if key in ALNSConfig.__dataclass_fields__
    })
    csgni_config = CSGNIConfig(
        intervention_rate=int(config["intervention_rate"]),
        proposal_seed_namespace=int(phase6h_config["rng_namespaces"]["proposal"]),
        ni_repair_seed_namespace=int(phase6h_config["rng_namespaces"]["ni_repair"]),
        acceptance_seed_namespace=int(phase6h_config["rng_namespaces"]["acceptance"]),
        diagnostics_seed_namespace=int(phase6h_config["rng_namespaces"]["diagnostics"]),
    )
    model_load_started = time.perf_counter()
    policy = FrozenLiveInference(
        ROOT / config["phase6f_experiment_freeze"],
        device=args.device,
        proposal_seed_namespace=int(phase6h_config["rng_namespaces"]["proposal"]),
        deployment_artifact=ROOT / config["phase6h_policy"],
    )
    model_load_seconds = time.perf_counter() - model_load_started
    if policy.deployment_artifact_sha256 != config["phase6h_policy_sha256"]:
        raise RuntimeError("loaded Phase 6H policy hash mismatch")

    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": args.device,
        "gpu_model": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else None,
        "cpu_model": cpu_model(),
        "cpu_threads_per_solver": 1,
        "solver_concurrency": configured_concurrency,
        "gpu_workers": int(config["gpu_workers"]),
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
    }
    current_commit = git_commit()
    for index, task in enumerate(pending, 1):
        instance = load_instance(CORE_ROOT / task["instance_relative_path"])
        if instance.num_operations != task["number_of_operations"]:
            raise RuntimeError(f"operation count mismatch: {task['instance_id']}")
        budget = float(config["wall_clock_seconds_per_operation"]) * instance.num_operations
        observer = Phase6HLiveObserver({
            **task,
            "suite": "RCIAS-CB1 Core",
            "algorithm": ALGORITHM,
            "policy_name": policy.policy_name,
        })
        started_at = datetime.now(timezone.utc).isoformat()
        result = solve_csgni(
            instance,
            budget,
            task["seed"],
            policy,
            alns_config=alns_config,
            csgni_config=csgni_config,
            observer=observer,
        )
        trace = validate_incumbent_trace(
            result.convergence_trace, final_best=result.best.makespan
        )
        feasibility = check_schedule(instance, result.best.schedule)
        if not feasibility["feasible"]:
            raise RuntimeError({"task": task, "violations": feasibility["violations"]})
        log = live_log_path(task)
        atomic_parquet(pd.DataFrame(observer.rows), log)
        payload = {
            "schema": "initial-manuscript-core-run-v1",
            "status": "COMPLETE",
            "algorithm": ALGORITHM,
            "algorithm_version": "PHASE6H_FROZEN_POLICY",
            "experiment_status": "PROVISIONAL",
            "suite": "RCIAS-CB1 Core",
            **task,
            "instance_path": str((CORE_ROOT / task["instance_relative_path"]).relative_to(ROOT)),
            "started_at_utc": started_at,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": current_commit,
            "config_path": str(CONFIG_PATH.relative_to(ROOT)),
            "config_sha256": config_hash,
            "core_protocol_sha256": protocol_hash,
            "implementation_manifest_sha256": implementation_hash,
            "phase6h_policy_path": config["phase6h_policy"],
            "phase6h_policy_sha256": config["phase6h_policy_sha256"],
            "checkpoint_sha256": config["checkpoint_sha256"],
            "policy_name": policy.policy_name,
            "time_limit_seconds": budget,
            "runtime": result.runtime,
            "best_makespan": result.best.makespan,
            "best_found_time": result.best_found_time,
            "decoder_evaluations": result.decoder_evaluations,
            "iterations": result.iterations,
            "feasible": True,
            "independent_feasibility_audit": feasibility,
            "model_load_seconds_excluded_from_run_budget": model_load_seconds,
            "live_log_sha256": digest(log),
            "incumbent_trace": trace,
            "normalized_budget_checkpoints": sample_incumbent_trace(
                trace,
                budget=budget,
                fractions=phase6h_config["anytime"]["normalized_budget_fractions"],
            ),
            "diagnostics": result.diagnostics,
            "best_solution": result.best.schedule.to_dict(),
            "best_actions": [asdict(action) for action in result.best.actions],
            "environment": environment,
        }
        atomic_json(payload, result_path(task))
        completed.add((task["instance_id"], task["seed"]))
        write_progress(tasks, process_started, args.shard_count, args.shard_index)
        print(
            f"[{index}/{len(pending)} shard={args.shard_index}/{args.shard_count}] "
            f"{ALGORITHM} {task['instance_id']} "
            f"seed={task['seed']} makespan={result.best.makespan:g} "
            f"evals={result.decoder_evaluations} runtime={result.runtime:.2f}s",
            flush=True,
        )
    print(
        f"PAPER_PHASE6H_CORE_SHARD_RETURNED shard={args.shard_index}/{args.shard_count} "
        f"local_completed={sum(result_path(task).is_file() for task in assigned)}"
        f"/{len(assigned)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
