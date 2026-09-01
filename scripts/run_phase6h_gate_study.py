#!/usr/bin/env python3
"""Run the preregistered CAL-FIT-only Phase 6H intervention-gate study."""

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
    validate_incumbent_trace,
)
from rcias_clgri.data.loader import load_instance  # noqa: E402
from rcias_clgri.env.feasibility import check_schedule  # noqa: E402
from rcias_clgri.ni.live_inference import FrozenLiveInference  # noqa: E402
from rcias_clgri.search.alns import ALNSConfig  # noqa: E402
from rcias_clgri.search.csgni import CSGNIConfig, solve_csgni  # noqa: E402


CONFIG_PATH = ROOT / "configs/phase6h_live_calibration.json"
CALIBRATION = ROOT / "outputs/phase6h_calibration/calibration"
OUT = ROOT / "outputs/phase6h_calibration/gate_study"
RUNS = OUT / "runs"
LIVE_LOGS = OUT / "live_logs"


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


def read_alns_config() -> ALNSConfig:
    raw = load_json(ROOT / "configs/phase5c_alns.json")
    return ALNSConfig(**{
        key: value for key, value in raw.items()
        if key in ALNSConfig.__dataclass_fields__
    })


def load_candidate_manifest(config: dict) -> pd.DataFrame:
    integrity = load_json(CALIBRATION / "fit_integrity.json")
    if integrity.get("status") != "PASS" or integrity.get("cal_holdout_opened") is not False:
        raise RuntimeError("CAL-FIT calibration artifact must pass before the gate study")
    manifest_path = CALIBRATION / "candidate_policy_manifest.csv"
    if digest(manifest_path) != integrity["candidate_manifest_sha256"]:
        raise RuntimeError("candidate policy manifest hash mismatch")
    manifest = pd.read_csv(manifest_path)
    expected = config["intervention_gate_study"]["candidates"]
    if set(manifest.policy_name) != set(expected):
        raise RuntimeError("candidate policy set differs from the preregistration")
    for row in manifest.itertuples(index=False):
        if digest(ROOT / row.relative_path) != row.sha256:
            raise RuntimeError(f"candidate policy hash mismatch: {row.policy_name}")
    return manifest.set_index("policy_name").loc[expected].reset_index()


def build_tasks(config: dict, instance_manifest: pd.DataFrame, policies: pd.DataFrame) -> list[dict]:
    fit = instance_manifest[instance_manifest.calibration_split == "CAL_FIT"]
    if len(fit) != 9 or set(fit.replicate) != {"R07"}:
        raise RuntimeError("gate study may access only the nine CAL-FIT instances")
    return [
        {
            "policy_name": policy.policy_name,
            "policy_relative_path": policy.relative_path,
            "policy_sha256": policy.sha256,
            "instance_id": row.instance_id,
            "instance_relative_path": row.relative_path,
            "scale": row.scale,
            "CF_level": row.CF_level,
            "seed": int(seed),
        }
        for policy in policies.itertuples(index=False)
        for row in fit.itertuples(index=False)
        for seed in config["seeds"]["CAL_FIT_GATE_STUDY"]
    ]


def result_path(task: dict) -> Path:
    return (
        RUNS / task["policy_name"] / task["instance_id"]
        / f"seed_{task['seed']}.json"
    )


def log_path(task: dict) -> Path:
    return (
        LIVE_LOGS / task["policy_name"] / task["instance_id"]
        / f"seed_{task['seed']}.parquet"
    )


def valid_result(task: dict) -> dict | None:
    path = result_path(task)
    if not path.exists():
        return None
    try:
        payload = load_json(path)
        live_log = log_path(task)
        if (
            payload.get("status") != "COMPLETE"
            or payload.get("policy_sha256") != task["policy_sha256"]
            or not live_log.exists()
            or payload.get("live_log_sha256") != digest(live_log)
        ):
            return None
        return payload
    except (json.JSONDecodeError, KeyError, OSError):
        return None


def write_progress(tasks: list[dict], process_started: float) -> None:
    completed = sum(valid_result(task) is not None for task in tasks)
    elapsed = time.perf_counter() - process_started
    rate = completed / elapsed if elapsed else 0.0
    atomic_json({
        "schema": "phase6h-gate-study-progress-v1",
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "completed_runs": completed,
        "total_runs": len(tasks),
        "current_process_naive_remaining_seconds": (
            (len(tasks) - completed) / rate if rate else None
        ),
        "status": "COMPLETE" if completed == len(tasks) else "RUNNING",
    }, OUT / "progress.json")


def summarize(tasks: list[dict]) -> None:
    rows = []
    for task in tasks:
        payload = valid_result(task)
        if payload is None:
            continue
        live = pd.read_parquet(log_path(task))
        interventions = live[live.ni_intervention]
        rows.append({
            **{key: task[key] for key in (
                "policy_name", "policy_sha256", "instance_id", "scale",
                "CF_level", "seed",
            )},
            "best_makespan": payload["best_makespan"],
            "time_to_best": payload["time_to_best"],
            "evals_to_best": payload["evals_to_best"],
            "runtime": payload["runtime"],
            "decoder_evaluations": payload["decoder_evaluations"],
            "iterations": payload["iterations"],
            "feasible": payload["feasible"],
            "intervention_coverage": float(live.ni_intervention.mean()),
            "fallback_coverage": float(live.fallback.mean()),
            "intervention_positive_fraction": (
                float(interventions.realized_positive.mean())
                if len(interventions) else None
            ),
            "intervention_mean_utility": (
                float(interventions.realized_immediate_utility.mean())
                if len(interventions) else None
            ),
            "acceptance_rate": float(live.accepted.mean()),
            "global_best_hit_rate": float(live.new_global_best.mean()),
            "mean_decision_overhead_ms": float(live.ni_overhead_ms.mean()),
        })
    runs = pd.DataFrame(rows)
    atomic_csv(runs, OUT / "gate_study_run_summary.csv")
    if len(runs) != len(tasks):
        return
    instance_rows = []
    for (policy, instance_id), part in runs.groupby(["policy_name", "instance_id"]):
        instance_rows.append({
            "policy_name": policy,
            "instance_id": instance_id,
            "scale": part.scale.iloc[0],
            "CF_level": part.CF_level.iloc[0],
            "run_count": len(part),
            "mean_final_makespan": float(part.best_makespan.mean()),
            "best_final_makespan": float(part.best_makespan.min()),
            "std_final_makespan": float(part.best_makespan.std(ddof=0)),
            "feasibility_rate": float(part.feasible.mean()),
            "mean_decoder_evaluations": float(part.decoder_evaluations.mean()),
            "mean_decision_overhead_ms": float(part.mean_decision_overhead_ms.mean()),
            "mean_intervention_coverage": float(part.intervention_coverage.mean()),
        })
    instances = pd.DataFrame(instance_rows)
    atomic_csv(instances, OUT / "gate_study_instance_summary.csv")
    policy_rows = []
    for policy, part in instances.groupby("policy_name"):
        policy_rows.append({
            "policy_name": policy,
            "instance_count": len(part),
            "mean_of_instance_mean_final_makespan": float(
                part.mean_final_makespan.mean()
            ),
            "mean_of_instance_best_final_makespan": float(
                part.best_final_makespan.mean()
            ),
            "feasibility_rate": float(part.feasibility_rate.mean()),
            "mean_decoder_evaluations": float(part.mean_decoder_evaluations.mean()),
            "mean_decision_overhead_ms": float(part.mean_decision_overhead_ms.mean()),
            "mean_intervention_coverage": float(part.mean_intervention_coverage.mean()),
        })
    policies = pd.DataFrame(policy_rows)
    eligible = policies[policies.feasibility_rate == 1.0]
    if eligible.empty:
        raise RuntimeError("no gate candidate has 100% final-schedule feasibility")
    selected_name = str(eligible.sort_values([
        "mean_of_instance_mean_final_makespan",
        "mean_decoder_evaluations",
        "mean_decision_overhead_ms",
        "policy_name",
    ]).iloc[0].policy_name)
    policies["selected"] = policies.policy_name == selected_name
    atomic_csv(policies, OUT / "gate_study_policy_summary.csv")
    atomic_json({
        "schema": "phase6h-gate-study-integrity-v1",
        "status": "PASS",
        "expected_runs": len(tasks),
        "complete_runs": len(runs),
        "all_final_schedules_feasible": bool(runs.feasible.all()),
        "selected_policy": selected_name,
        "selection_data": "CAL_FIT_ONLY",
        "cal_holdout_opened": False,
    }, OUT / "gate_study_integrity.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit-runs", type=int)
    parser.add_argument("--policies", nargs="+")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()

    config = load_json(CONFIG_PATH)
    policies = load_candidate_manifest(config)
    if args.policies:
        unknown = set(args.policies) - set(policies.policy_name)
        if unknown:
            raise ValueError(f"unknown policy candidates: {sorted(unknown)}")
    instance_manifest = pd.read_csv(ROOT / config["calibration_instances"]["manifest"])
    all_tasks = build_tasks(config, instance_manifest, policies)
    selected_tasks = (
        all_tasks if not args.policies
        else [task for task in all_tasks if task["policy_name"] in args.policies]
    )
    process_started = time.perf_counter()
    if args.summarize_only:
        summarize(all_tasks)
        write_progress(all_tasks, process_started)
        print("PHASE6H_GATE_STUDY_SUMMARY_RETURNED", flush=True)
        return
    pending = [task for task in selected_tasks if valid_result(task) is None]
    if args.limit_runs is not None:
        pending = pending[:args.limit_runs]
    print(
        f"PHASE6H_GATE_STUDY_START pending={len(pending)} total={len(all_tasks)}",
        flush=True,
    )
    alns_config = read_alns_config()
    csgni_config = CSGNIConfig(
        intervention_rate=int(config["search"]["intervention_rate"]),
        proposal_seed_namespace=int(config["rng_namespaces"]["proposal"]),
        ni_repair_seed_namespace=int(config["rng_namespaces"]["ni_repair"]),
        acceptance_seed_namespace=int(config["rng_namespaces"]["acceptance"]),
        diagnostics_seed_namespace=int(config["rng_namespaces"]["diagnostics"]),
    )
    instance_root = ROOT / config["calibration_instances"]["root"]
    policy = None
    active_policy = None
    model_load_seconds = None
    for index, task in enumerate(pending, 1):
        if active_policy != task["policy_name"]:
            load_started = time.perf_counter()
            policy = FrozenLiveInference(
                ROOT / config["frozen_phase6f"]["experiment_freeze"],
                device=args.device,
                proposal_seed_namespace=config["rng_namespaces"]["proposal"],
                deployment_artifact=ROOT / task["policy_relative_path"],
            )
            model_load_seconds = time.perf_counter() - load_started
            active_policy = task["policy_name"]
        instance = load_instance(instance_root / task["instance_relative_path"])
        budget = float(config["search"]["wall_clock_seconds_per_operation"]) * instance.num_operations
        observer = Phase6HLiveObserver({
            **task,
            "calibration_split": "CAL_FIT",
        })
        result = solve_csgni(
            instance,
            budget,
            task["seed"],
            policy,
            alns_config=alns_config,
            csgni_config=csgni_config,
            observer=observer,
        )
        feasibility = check_schedule(instance, result.best.schedule)
        if not feasibility["feasible"]:
            raise RuntimeError({"task": task, "violations": feasibility["violations"]})
        trace = validate_incumbent_trace(
            result.convergence_trace, final_best=result.best.makespan
        )
        live_log = log_path(task)
        atomic_parquet(pd.DataFrame(observer.rows), live_log)
        payload = {
            "schema": "phase6h-gate-study-run-v1",
            "status": "COMPLETE",
            **task,
            "calibration_split": "CAL_FIT",
            "time_limit_seconds": budget,
            "best_makespan": result.best.makespan,
            "time_to_best": result.best_found_time,
            "evals_to_best": trace[-1]["decoder_evaluations"],
            "runtime": result.runtime,
            "decoder_evaluations": result.decoder_evaluations,
            "iterations": result.iterations,
            "initialization_seconds": result.diagnostics["initialization_seconds"],
            "feasible": True,
            "model_load_seconds": model_load_seconds,
            "live_log_sha256": digest(live_log),
            "convergence_trace": trace,
            "diagnostics": result.diagnostics,
            "best_solution": result.best.schedule.to_dict(),
            "best_actions": [asdict(action) for action in result.best.actions],
        }
        atomic_json(payload, result_path(task))
        write_progress(all_tasks, process_started)
        print(
            f"[{index}/{len(pending)}] {task['policy_name']} {task['instance_id']} "
            f"seed={task['seed']} makespan={result.best.makespan:g} "
            f"runtime={result.runtime:.2f}s",
            flush=True,
        )
    summarize(all_tasks)
    write_progress(all_tasks, process_started)
    print("PHASE6H_GATE_STUDY_BATCH_RETURNED", flush=True)


if __name__ == "__main__":
    main()
