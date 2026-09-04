#!/usr/bin/env python3
"""Resumable two-shard runner for the P1 random-selection ablation."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import hashlib
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


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from paper_experiments.ablation.csgni_random_policy import (  # noqa: E402
    UniformFullBankAtFrozenGate,
)
from rcias_clgri.analysis.phase6h import (  # noqa: E402
    Phase6HLiveObserver,
    sample_incumbent_trace,
    validate_incumbent_trace,
)
from rcias_clgri.data.loader import load_instance  # noqa: E402
from rcias_clgri.env.feasibility import check_schedule  # noqa: E402
from rcias_clgri.heuristic.dispatching import solve_dispatching  # noqa: E402
from rcias_clgri.ni.live_inference import FrozenLiveInference  # noqa: E402
from rcias_clgri.search.alns import ALNSConfig  # noqa: E402
from rcias_clgri.search.common import candidate_from_actions, decode_candidate  # noqa: E402
from rcias_clgri.search.csgni import CSGNIConfig, solve_csgni  # noqa: E402


ALGORITHM = "CSG_NI_UNIFORM_FULL_BANK_FROZEN_GATE"
PAPER_ROOT = ROOT / "paper_experiments"
ABLATION_ROOT = PAPER_ROOT / "ablation"
CONFIG_PATH = ABLATION_ROOT / "configs/p1_ablation_protocol.json"
MANIFEST_PATH = ABLATION_ROOT / "ablation_instance_manifest.csv"
OUTPUT_ROOT = ABLATION_ROOT / "raw_results/random_full_bank_frozen_gate"
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


def verify_file(path: Path, expected: str, label: str) -> None:
    if not path.is_file() or digest(path) != expected:
        raise RuntimeError(f"{label} is missing or has changed: {path}")


def load_protocol() -> tuple[dict, str, str]:
    config = load_json(CONFIG_PATH)
    if config.get("status") != "FROZEN_BEFORE_P1_EXECUTION":
        raise RuntimeError("P1 config is not frozen before execution")
    if config.get("new_experiment_arm") != ALGORITHM:
        raise RuntimeError("P1 experiment-arm identity changed")
    config_hash = digest(CONFIG_PATH)
    verify_file(MANIFEST_PATH, config["ablation_instance_manifest_sha256"], "P1 instance manifest")
    implementation_path = ROOT / config["implementation_manifest"]
    verify_file(
        implementation_path,
        config["implementation_manifest_sha256"],
        "P1 implementation manifest",
    )
    implementation = load_json(implementation_path)
    for record in implementation["files"]:
        verify_file(ROOT / record["path"], record["sha256"], record["path"])
    for key in (
        "reference_core_config",
        "phase6h_policy",
        "phase6f_experiment_freeze",
        "phase6h_config",
        "alns_config",
    ):
        verify_file(ROOT / config[key], config[f"{key}_sha256"], key)
    return config, config_hash, digest(implementation_path)


def tasks(config: dict) -> list[dict]:
    with MANIFEST_PATH.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 18 or len({row["instance_id"] for row in rows}) != 18:
        raise RuntimeError("P1 requires 18 unique instances")
    result = []
    for row in rows:
        instance_path = ROOT / row["instance_path"]
        verify_file(instance_path, row["instance_sha256"], row["instance_id"])
        for seed in config["seeds"]:
            result.append({
                "instance_id": row["instance_id"],
                "instance_path": row["instance_path"],
                "instance_sha256": row["instance_sha256"],
                "scale": row["scale"],
                "CF_level": row["CF_level"],
                "number_of_operations": int(row["number_of_operations"]),
                "seed": int(seed),
            })
    if len(result) != int(config["new_arm_expected_runs"]):
        raise RuntimeError("P1 expected-run count mismatch")
    return result


def result_path(task: dict) -> Path:
    return RUNS / task["instance_id"] / f"seed_{task['seed']}.json"


def live_log_path(task: dict) -> Path:
    return LIVE_LOGS / task["instance_id"] / f"seed_{task['seed']}.parquet"


def validate_existing(task: dict, config: dict, config_hash: str, implementation_hash: str) -> dict | None:
    path = result_path(task)
    if not path.exists():
        return None
    payload = load_json(path)
    live_log = live_log_path(task)
    expected_limit = float(config["wall_clock_seconds_per_operation"]) * task["number_of_operations"]
    checks = (
        payload.get("schema") == "initial-manuscript-p1-ablation-run-v1",
        payload.get("status") == "COMPLETE",
        payload.get("algorithm") == ALGORITHM,
        payload.get("instance_id") == task["instance_id"],
        payload.get("instance_sha256") == task["instance_sha256"],
        payload.get("seed") == task["seed"],
        payload.get("time_limit_seconds") == expected_limit,
        payload.get("config_sha256") == config_hash,
        payload.get("implementation_manifest_sha256") == implementation_hash,
        payload.get("phase6h_policy_sha256") == config["phase6h_policy_sha256"],
        payload.get("checkpoint_sha256") == config["checkpoint_sha256"],
        payload.get("feasible") is True,
        payload.get("independent_feasibility_audit", {}).get("feasible") is True,
        live_log.is_file(),
        payload.get("live_log_sha256") == (digest(live_log) if live_log.is_file() else None),
    )
    if not all(checks):
        raise RuntimeError(f"existing P1 result failed integrity checks: {path}")
    return payload


def write_progress(all_tasks: list[dict], started: float, shard_count: int, shard_index: int) -> None:
    PROGRESS_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with PROGRESS_LOCK.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        completed = {
            (task["instance_id"], task["seed"])
            for task in all_tasks
            if result_path(task).is_file()
        }
        atomic_json({
            "schema": "initial-manuscript-p1-progress-v1",
            "status": "COMPLETE" if len(completed) == len(all_tasks) else "RUNNING",
            "algorithm": ALGORITHM,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "completed_runs": len(completed),
            "total_runs": len(all_tasks),
            "completed_by_scale": {
                scale: sum(
                    task["scale"] == scale and (task["instance_id"], task["seed"]) in completed
                    for task in all_tasks
                )
                for scale in ("S", "M", "L")
            },
            "configured_solver_concurrency": shard_count,
            "last_updating_shard": shard_index,
            "current_worker_elapsed_seconds": time.perf_counter() - started,
        }, PROGRESS)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


class P1Observer(Phase6HLiveObserver):
    def __call__(self, event: dict[str, object]) -> None:
        super().__call__(event)
        timing = dict(event.get("ni_timing_ms") or {})
        self.rows[-1]["uniform_selection_bank_rebuild_ms"] = float(
            timing.get("uniform_selection_bank_rebuild", 0.0)
        )


def build_policy(config: dict, device: str, *, force_intervention: bool = False) -> UniformFullBankAtFrozenGate:
    phase6h_config = load_json(ROOT / config["phase6h_config"])
    reference = FrozenLiveInference(
        ROOT / config["phase6f_experiment_freeze"],
        device=device,
        proposal_seed_namespace=int(phase6h_config["rng_namespaces"]["proposal"]),
        deployment_artifact=ROOT / config["phase6h_policy"],
        force_intervention=force_intervention,
    )
    if reference.deployment_artifact_sha256 != config["phase6h_policy_sha256"]:
        raise RuntimeError("loaded Phase6H policy hash mismatch")
    return UniformFullBankAtFrozenGate(
        reference,
        selection_seed_namespace=int(config["uniform_selection_seed_namespace"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--smoke-policy", action="store_true")
    parser.add_argument("--limit-runs", type=int)
    parser.add_argument("--shard-count", type=int, default=2)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    if args.shard_index < 0 or args.shard_index >= args.shard_count:
        raise ValueError("shard-index must satisfy 0 <= index < shard-count")
    if args.limit_runs is not None and args.limit_runs < 1:
        raise ValueError("limit-runs must be positive")

    config, config_hash, implementation_hash = load_protocol()
    if args.shard_count != int(config["solver_concurrency"]):
        raise RuntimeError("runtime shard count differs from frozen P1 config")
    all_tasks = tasks(config)
    process_started = time.perf_counter()
    completed = set()
    for task in all_tasks:
        if validate_existing(task, config, config_hash, implementation_hash) is not None:
            completed.add((task["instance_id"], task["seed"]))
    write_progress(all_tasks, process_started, args.shard_count, args.shard_index)
    assigned = [task for index, task in enumerate(all_tasks) if index % args.shard_count == args.shard_index]
    pending = [task for task in assigned if (task["instance_id"], task["seed"]) not in completed]
    print(
        f"P1_ABLATION_START shard={args.shard_index}/{args.shard_count} "
        f"pending={len(pending)} assigned={len(assigned)} total={len(all_tasks)} device={args.device}",
        flush=True,
    )
    if args.audit_only:
        print("P1_ABLATION_AUDIT_RETURNED", flush=True)
        return 0

    model_load_started = time.perf_counter()
    policy = build_policy(config, args.device, force_intervention=args.smoke_policy)
    model_load_seconds = time.perf_counter() - model_load_started
    if args.smoke_policy:
        task = all_tasks[0]
        instance = load_instance(ROOT / task["instance_path"])
        h1 = solve_dispatching(instance, "H1")
        current = decode_candidate(instance, candidate_from_actions(instance, h1.actions))
        decision = policy.decide(
            instance,
            current,
            state_id=f"P1_SMOKE__{task['instance_id']}",
            destroy_count=max(2, round(instance.num_operations * 0.15)),
            search_progress=0.0,
            search_stage="0-20%",
        )
        if not decision.intervene or decision.requested_proposal_count != 24:
            raise RuntimeError("forced P1 policy smoke did not execute the full production bank")
        smoke = {
            "schema": "initial-manuscript-p1-policy-smoke-v1",
            "status": "PASS",
            "device": args.device,
            "instance_id": task["instance_id"],
            "requested_rule_count": decision.requested_proposal_count,
            "unique_target_count": decision.proposal_count,
            "duplicate_target_count": decision.duplicate_proposal_count,
            "selected_target_set_id": decision.selected_target_set_id,
            "selected_target_size": len(decision.destroyed_operations),
            "policy_name": decision.policy_name,
            "timings_ms": dict(decision.timings_ms or {}),
            "checkpoint_sha256": policy.checkpoint_sha256,
            "phase6h_policy_sha256": policy.deployment_artifact_sha256,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        atomic_json(smoke, ABLATION_ROOT / "audit/policy_smoke.json")
        print(json.dumps(smoke, indent=2, sort_keys=True), flush=True)
        return 0

    if args.limit_runs is not None:
        pending = pending[: args.limit_runs]
    phase6h_config = load_json(ROOT / config["phase6h_config"])
    alns_raw = load_json(ROOT / config["alns_config"])
    alns_config = ALNSConfig(**{
        key: value for key, value in alns_raw.items() if key in ALNSConfig.__dataclass_fields__
    })
    csgni_config = CSGNIConfig(
        intervention_rate=int(config["intervention_rate"]),
        proposal_seed_namespace=int(phase6h_config["rng_namespaces"]["proposal"]),
        ni_repair_seed_namespace=int(phase6h_config["rng_namespaces"]["ni_repair"]),
        acceptance_seed_namespace=int(phase6h_config["rng_namespaces"]["acceptance"]),
        diagnostics_seed_namespace=int(phase6h_config["rng_namespaces"]["diagnostics"]),
    )
    current_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()
    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": args.device,
        "gpu_model": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else None,
        "cpu_threads_per_solver": 1,
        "solver_concurrency": args.shard_count,
        "shard_index": args.shard_index,
    }
    for index, task in enumerate(pending, 1):
        instance = load_instance(ROOT / task["instance_path"])
        if instance.num_operations != task["number_of_operations"]:
            raise RuntimeError(f"operation count mismatch: {task['instance_id']}")
        budget = float(config["wall_clock_seconds_per_operation"]) * instance.num_operations
        observer = P1Observer({
            **task,
            "suite": "RCIAS-CB1 Core P1 subset",
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
        trace = validate_incumbent_trace(result.convergence_trace, final_best=result.best.makespan)
        feasibility = check_schedule(instance, result.best.schedule)
        if not feasibility["feasible"]:
            raise RuntimeError({"task": task, "violations": feasibility["violations"]})
        live_log = live_log_path(task)
        atomic_parquet(pd.DataFrame(observer.rows), live_log)
        payload = {
            "schema": "initial-manuscript-p1-ablation-run-v1",
            "status": "COMPLETE",
            "algorithm": ALGORITHM,
            "algorithm_version": "P1_RANDOM_SELECTION_FROZEN_PHASE6H_GATE",
            "suite": "RCIAS-CB1 Core P1 subset",
            **task,
            "started_at_utc": started_at,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": current_commit,
            "config_path": str(CONFIG_PATH.relative_to(ROOT)),
            "config_sha256": config_hash,
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
            "live_log_sha256": digest(live_log),
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
        write_progress(all_tasks, process_started, args.shard_count, args.shard_index)
        print(
            f"[{index}/{len(pending)} shard={args.shard_index}/{args.shard_count}] "
            f"{task['instance_id']} seed={task['seed']} makespan={result.best.makespan:g} "
            f"evals={result.decoder_evaluations} runtime={result.runtime:.2f}s",
            flush=True,
        )
    print(f"P1_ABLATION_SHARD_RETURNED shard={args.shard_index}/{args.shard_count}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
