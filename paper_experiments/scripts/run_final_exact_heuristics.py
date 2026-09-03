#!/usr/bin/env python3
"""Resumable five-method evaluation on the final exact-validation suite."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any


for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rcias_clgri.data.loader import load_instance  # noqa: E402
from rcias_clgri.env.feasibility import check_schedule  # noqa: E402
from rcias_clgri.search.dabc import DABCConfig, solve_dabc  # noqa: E402
from rcias_clgri.search.dcga import DCGAConfig, solve_dcga  # noqa: E402
from rcias_clgri.search.ga import GAConfig, solve_ga  # noqa: E402
from rcias_clgri.search.lghga import LGHGAConfig  # noqa: E402
from rcias_clgri.search.lghga_learning import load_dtr_bundle  # noqa: E402
from rcias_clgri.search.lghga_v2 import solve_lghga_v2  # noqa: E402


PROTOCOL_PATH = ROOT / "paper_experiments/configs/exact_validation/final_protocol.json"
IMPLEMENTATION_PATH = (
    ROOT / "paper_experiments/configs/exact_validation/final_implementation.json"
)
OUTPUT_ROOT = ROOT / "paper_experiments/raw_results/exact_validation_10_final"
PROGRESS_PATH = OUTPUT_ROOT / "progress.json"

CPU_METHODS = (
    "GA",
    "Adapted DCGA",
    "DABC-RIACRSP",
    "LG_HGA-RIACRSP-v2-N4M",
)
CSGNI_METHOD = "CSG-NI Phase6H provisional"
ALL_METHODS = (*CPU_METHODS, CSGNI_METHOD)
METHOD_DIRS = {
    "GA": "GA",
    "Adapted DCGA": "ADAPTED_DCGA",
    "DABC-RIACRSP": "DABC_RIACRSP",
    "LG_HGA-RIACRSP-v2-N4M": "LG_HGA_RIACRSP_V2_N4M",
    CSGNI_METHOD: "CSG_NI_PROVISIONAL_PHASE6H",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def atomic_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        text=True, capture_output=True,
    ).stdout.strip()


def dataclass_config(path: Path, cls: type):
    raw = load_json(path)
    return cls(**{
        key: value for key, value in raw.items()
        if key in cls.__dataclass_fields__
    })


def verify_file(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"missing {label}: {path}")
    observed = digest(path)
    if observed != expected:
        raise RuntimeError(
            f"{label} hash mismatch: expected={expected} observed={observed}"
        )


def verify_protocol() -> tuple[dict[str, Any], str, str]:
    protocol = load_json(PROTOCOL_PATH)
    if protocol.get("status") != "FROZEN_BEFORE_FINAL_EXACT_HEURISTIC_EVALUATION":
        raise RuntimeError("final exact heuristic protocol is not frozen")
    if protocol.get("primary_methods") != list(ALL_METHODS):
        raise RuntimeError("final exact method list changed")
    if protocol.get("seeds") != [530101, 530102, 530103, 530104, 530105]:
        raise RuntimeError("final exact seed list changed")
    for record in protocol["frozen_files"]:
        verify_file(ROOT / record["path"], record["sha256"], record["path"])
    audit = load_json(ROOT / protocol["benchmark_audit"])
    if not audit.get("all_primary_gates_passed"):
        raise RuntimeError("final exact benchmark audit gate did not pass")
    verify_file(
        IMPLEMENTATION_PATH,
        protocol["implementation_manifest_sha256"],
        "final exact implementation manifest",
    )
    implementation = load_json(IMPLEMENTATION_PATH)
    if implementation.get("status") != "FROZEN_BEFORE_FINAL_EXACT_HEURISTIC_EVALUATION":
        raise RuntimeError("final exact implementation is not frozen")
    for record in implementation["files"]:
        verify_file(ROOT / record["path"], record["sha256"], record["path"])
    return protocol, digest(PROTOCOL_PATH), digest(IMPLEMENTATION_PATH)


def build_tasks(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    manifest_path = ROOT / protocol["instance_manifest"]
    benchmark_root = manifest_path.parent.parent
    manifest = load_json(manifest_path)
    references = {}
    with (ROOT / protocol["gurobi_results"]).open(
        newline="", encoding="utf-8"
    ) as handle:
        for row in csv.DictReader(handle):
            references[row["instance_id"]] = float(row["objective_makespan"])
    tasks = []
    for entry in manifest["instances"]:
        instance_path = benchmark_root / entry["relative_path"]
        verify_file(instance_path, entry["sha256"], entry["instance_id"])
        for method in protocol["primary_methods"]:
            for seed in protocol["seeds"]:
                tasks.append({
                    "instance_id": entry["instance_id"],
                    "instance_path": str(instance_path.relative_to(ROOT)),
                    "instance_sha256": entry["sha256"],
                    "number_of_operations": int(entry["operation_count"]),
                    "source_instance_id": entry["source_instance_id"],
                    "source_suite": entry["source_suite"],
                    "proven_optimum": references[entry["instance_id"]],
                    "method": method,
                    "seed": int(seed),
                })
    expected = 10 * 5 * 5
    if len(tasks) != expected:
        raise RuntimeError(f"expected {expected} final exact tasks, got {len(tasks)}")
    return tasks


def result_path(task: dict[str, Any]) -> Path:
    return (
        OUTPUT_ROOT / "runs" / METHOD_DIRS[task["method"]]
        / task["instance_id"] / f"seed_{task['seed']}.json"
    )


def live_log_path(task: dict[str, Any]) -> Path:
    return (
        OUTPUT_ROOT / "live_logs" / task["instance_id"]
        / f"seed_{task['seed']}.parquet"
    )


def validate_existing(
    task: dict[str, Any], protocol_hash: str, implementation_hash: str
) -> bool:
    path = result_path(task)
    if not path.exists():
        return False
    payload = load_json(path)
    required = [
        payload.get("schema") == "initial-manuscript-final-exact-run-v1",
        payload.get("status") == "COMPLETE",
        payload.get("instance_id") == task["instance_id"],
        payload.get("instance_sha256") == task["instance_sha256"],
        payload.get("method") == task["method"],
        payload.get("seed") == task["seed"],
        payload.get("protocol_sha256") == protocol_hash,
        payload.get("implementation_manifest_sha256") == implementation_hash,
        payload.get("feasible") is True,
        payload.get("independent_feasibility_audit", {}).get("feasible") is True,
    ]
    if not all(required):
        raise RuntimeError(f"existing result failed integrity checks: {path}")
    if task["method"] == CSGNI_METHOD:
        log = live_log_path(task)
        if not log.is_file() or payload.get("live_log_sha256") != digest(log):
            raise RuntimeError(f"existing CSG-NI live log failed integrity checks: {log}")
    return True


def base_payload(
    task: dict[str, Any], protocol: dict[str, Any], protocol_hash: str,
    implementation_hash: str, result: Any, audit: dict[str, Any],
) -> dict[str, Any]:
    optimum = float(task["proven_optimum"])
    optimum_reached = result.best.makespan <= optimum + 1e-6
    return {
        "schema": "initial-manuscript-final-exact-run-v1",
        "status": "COMPLETE",
        "suite": "RIACRSP exact validation 10 final",
        **task,
        "started_from_commit": git_commit(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_path": str(PROTOCOL_PATH.relative_to(ROOT)),
        "protocol_sha256": protocol_hash,
        "implementation_manifest_sha256": implementation_hash,
        "time_limit_seconds": (
            float(protocol["wall_clock_seconds_per_operation"])
            * task["number_of_operations"]
        ),
        "runtime_seconds": result.runtime,
        "best_makespan": result.best.makespan,
        "gap_to_proven_optimum_percent": (
            100.0 * (result.best.makespan - optimum) / optimum
        ),
        "optimum_reached": optimum_reached,
        "first_optimum_time_seconds": result.best_found_time if optimum_reached else None,
        "right_censored_without_optimum": not optimum_reached,
        "best_found_time_seconds": result.best_found_time,
        "decoder_evaluations": result.decoder_evaluations,
        "iterations": result.iterations,
        "generations_if_applicable": result.generations_if_applicable,
        "feasible": True,
        "independent_feasibility_audit": audit,
        "best_solution": result.best.schedule.to_dict(),
        "best_actions": [asdict(action) for action in result.best.actions],
        "convergence_trace": [asdict(point) for point in result.convergence_trace],
        "diagnostics": result.diagnostics,
        "compute": {
            "cpu_threads_per_solver": 1,
            "gpu_used": task["method"] == CSGNI_METHOD,
        },
        "environment": {"python": platform.python_version()},
    }


def run_cpu_task(
    packed: tuple[dict[str, Any], dict[str, Any], str, str]
) -> tuple[str, float, float]:
    task, protocol, protocol_hash, implementation_hash = packed
    instance_path = ROOT / task["instance_path"]
    verify_file(instance_path, task["instance_sha256"], task["instance_id"])
    instance = load_instance(instance_path)
    budget = (
        float(protocol["wall_clock_seconds_per_operation"])
        * instance.num_operations
    )
    config_paths = protocol["algorithm_configs"]
    if task["method"] == "GA":
        config = dataclass_config(ROOT / config_paths["GA"], GAConfig)
        result = solve_ga(instance, budget, task["seed"], config)
        method_artifact = {"config": config_paths["GA"], "effective_config": asdict(config)}
    elif task["method"] == "Adapted DCGA":
        config = dataclass_config(ROOT / config_paths["Adapted DCGA"], DCGAConfig)
        result = solve_dcga(instance, budget, task["seed"], config)
        method_artifact = {
            "config": config_paths["Adapted DCGA"], "effective_config": asdict(config)
        }
    elif task["method"] == "DABC-RIACRSP":
        config = dataclass_config(ROOT / config_paths["DABC-RIACRSP"], DABCConfig)
        result = solve_dabc(instance, budget, task["seed"], config)
        method_artifact = {
            "config": config_paths["DABC-RIACRSP"], "effective_config": asdict(config)
        }
    elif task["method"] == "LG_HGA-RIACRSP-v2-N4M":
        config = dataclass_config(
            ROOT / config_paths["LG_HGA-RIACRSP-v2-N4M"], LGHGAConfig
        )
        model_dir = ROOT / protocol["lghga_model_selection"]["model_dir"]
        models = load_dtr_bundle(model_dir)
        result = solve_lghga_v2(instance, budget, task["seed"], models, config)
        method_artifact = {
            "config": config_paths["LG_HGA-RIACRSP-v2-N4M"],
            "effective_config": asdict(config),
            "model_regime": protocol["lghga_model_selection"]["regime"],
            "model_dir": str(model_dir.relative_to(ROOT)),
            "model_manifest_sha256": digest(model_dir / "model_manifest.json"),
            "model_hashes": dict(models.model_hashes),
        }
    else:
        raise ValueError(f"unsupported CPU method: {task['method']}")
    audit = check_schedule(instance, result.best.schedule)
    if not audit["feasible"]:
        raise RuntimeError(f"{task['method']} returned infeasible {task['instance_id']}")
    payload = base_payload(
        task, protocol, protocol_hash, implementation_hash, result, audit
    )
    payload["method_artifact"] = method_artifact
    output = result_path(task)
    atomic_json(payload, output)
    return str(output.relative_to(ROOT)), result.best.makespan, result.runtime


def run_csgni_tasks(
    tasks: list[dict[str, Any]], protocol: dict[str, Any], protocol_hash: str,
    implementation_hash: str, device: str,
) -> None:
    import pandas as pd

    from rcias_clgri.analysis.phase6h import Phase6HLiveObserver
    from rcias_clgri.ni.live_inference import FrozenLiveInference
    from rcias_clgri.search.alns import ALNSConfig
    from rcias_clgri.search.csgni import CSGNIConfig, solve_csgni

    artifacts = protocol["phase6h_csgni"]
    phase_config = load_json(ROOT / artifacts["phase6h_config"])
    alns_config = dataclass_config(ROOT / artifacts["alns_config"], ALNSConfig)
    csgni_config = CSGNIConfig(
        intervention_rate=int(artifacts["intervention_rate"]),
        proposal_seed_namespace=int(phase_config["rng_namespaces"]["proposal"]),
        ni_repair_seed_namespace=int(phase_config["rng_namespaces"]["ni_repair"]),
        acceptance_seed_namespace=int(phase_config["rng_namespaces"]["acceptance"]),
        diagnostics_seed_namespace=int(phase_config["rng_namespaces"]["diagnostics"]),
    )
    load_started = time.perf_counter()
    policy = FrozenLiveInference(
        ROOT / artifacts["phase6f_experiment_freeze"],
        device=device,
        proposal_seed_namespace=int(phase_config["rng_namespaces"]["proposal"]),
        deployment_artifact=ROOT / artifacts["phase6h_policy"],
    )
    model_load_seconds = time.perf_counter() - load_started
    for index, task in enumerate(tasks, 1):
        instance = load_instance(ROOT / task["instance_path"])
        budget = (
            float(protocol["wall_clock_seconds_per_operation"])
            * instance.num_operations
        )
        observer = Phase6HLiveObserver({
            **task,
            "suite": "RIACRSP exact validation 10 final",
            "policy_name": policy.policy_name,
        })
        result = solve_csgni(
            instance, budget, task["seed"], policy,
            alns_config=alns_config, csgni_config=csgni_config, observer=observer,
        )
        audit = check_schedule(instance, result.best.schedule)
        if not audit["feasible"]:
            raise RuntimeError(f"CSG-NI returned infeasible {task['instance_id']}")
        log = live_log_path(task)
        log.parent.mkdir(parents=True, exist_ok=True)
        temporary = log.with_name(f"{log.name}.tmp.{os.getpid()}")
        pd.DataFrame(observer.rows).to_parquet(temporary, index=False, engine="pyarrow")
        temporary.replace(log)
        payload = base_payload(
            task, protocol, protocol_hash, implementation_hash, result, audit
        )
        payload.update({
            "algorithm_version": "PHASE6H_FROZEN_POLICY",
            "experiment_status": "PROVISIONAL",
            "phase6h_policy_sha256": digest(ROOT / artifacts["phase6h_policy"]),
            "checkpoint_sha256": artifacts["checkpoint_sha256"],
            "model_load_seconds_excluded_from_run_budget": model_load_seconds,
            "live_log_sha256": digest(log),
            "device": device,
        })
        atomic_json(payload, result_path(task))
        print(
            f"[{index}/{len(tasks)}] {CSGNI_METHOD} {task['instance_id']} "
            f"seed={task['seed']} makespan={result.best.makespan:g} "
            f"runtime={result.runtime:.2f}s",
            flush=True,
        )
        write_progress(build_tasks(protocol), protocol_hash, implementation_hash)


def write_progress(
    tasks: list[dict[str, Any]], protocol_hash: str, implementation_hash: str
) -> None:
    completed_by_method = {
        method: sum(result_path(task).is_file() for task in tasks if task["method"] == method)
        for method in ALL_METHODS
    }
    total = len(tasks)
    completed = sum(completed_by_method.values())
    atomic_json({
        "schema": "initial-manuscript-final-exact-progress-v1",
        "status": "COMPLETE" if completed == total else "RUNNING",
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "completed_runs": completed,
        "total_runs": total,
        "completed_by_method": completed_by_method,
        "protocol_sha256": protocol_hash,
        "implementation_manifest_sha256": implementation_hash,
    }, PROGRESS_PATH)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", choices=ALL_METHODS, default=list(CPU_METHODS))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if CSGNI_METHOD in args.methods and (len(args.methods) != 1 or args.workers != 1):
        raise ValueError("CSG-NI must run alone with --workers 1")

    protocol, protocol_hash, implementation_hash = verify_protocol()
    if CSGNI_METHOD not in args.methods and args.workers != int(
        protocol["cpu_solver_concurrency"]
    ):
        raise ValueError(
            f"formal CPU concurrency is frozen at {protocol['cpu_solver_concurrency']}"
        )
    tasks = build_tasks(protocol)
    for task in tasks:
        validate_existing(task, protocol_hash, implementation_hash)
    pending = [
        task for task in tasks
        if task["method"] in args.methods
        and not result_path(task).is_file()
    ]
    write_progress(tasks, protocol_hash, implementation_hash)
    print(
        f"FINAL_EXACT_START methods={args.methods} pending={len(pending)} "
        f"workers={args.workers} output={OUTPUT_ROOT.relative_to(ROOT)}",
        flush=True,
    )
    if args.audit_only:
        print("FINAL_EXACT_AUDIT_RETURNED", flush=True)
        return
    if CSGNI_METHOD in args.methods:
        run_csgni_tasks(
            pending, protocol, protocol_hash, implementation_hash, args.device
        )
    else:
        packed = [
            (task, protocol, protocol_hash, implementation_hash) for task in pending
        ]
        if args.workers == 1:
            results = map(run_cpu_task, packed)
            for index, (path, makespan, runtime) in enumerate(results, 1):
                print(
                    f"[{index}/{len(pending)}] {path} makespan={makespan:g} "
                    f"runtime={runtime:.2f}s", flush=True,
                )
                write_progress(tasks, protocol_hash, implementation_hash)
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = [executor.submit(run_cpu_task, item) for item in packed]
                for index, future in enumerate(as_completed(futures), 1):
                    path, makespan, runtime = future.result()
                    print(
                        f"[{index}/{len(pending)}] {path} makespan={makespan:g} "
                        f"runtime={runtime:.2f}s", flush=True,
                    )
                    write_progress(tasks, protocol_hash, implementation_hash)
    print("FINAL_EXACT_RETURNED", flush=True)


if __name__ == "__main__":
    main()
