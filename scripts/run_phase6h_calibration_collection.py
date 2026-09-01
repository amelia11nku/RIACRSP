#!/usr/bin/env python3
"""Resumable CAL-FIT live-state/outcome collection for Phase 6H."""

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
OUT = ROOT / "outputs/phase6h_calibration/collection"
RUNS = OUT / "runs"
LIVE_LOGS = OUT / "live_logs"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


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


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_alns_config() -> ALNSConfig:
    raw = load_json(ROOT / "configs/phase5c_alns.json")
    return ALNSConfig(**{
        key: value for key, value in raw.items()
        if key in ALNSConfig.__dataclass_fields__
    })


def build_tasks(config: dict, manifest: pd.DataFrame) -> list[dict]:
    fit = manifest[manifest.calibration_split == "CAL_FIT"]
    if len(fit) != 9 or set(fit.replicate) != {"R07"}:
        raise RuntimeError("CAL-FIT must be exactly the nine frozen R07 instances")
    return [
        {
            "instance_id": row.instance_id,
            "instance_relative_path": row.relative_path,
            "scale": row.scale,
            "CF_level": row.CF_level,
            "RI_level": row.RI_level,
            "TI_level": row.TI_level,
            "seed": int(seed),
        }
        for row in fit.itertuples(index=False)
        for seed in config["seeds"]["CAL_FIT_COLLECTION"]
    ]


def result_path(task: dict) -> Path:
    return RUNS / task["instance_id"] / f"seed_{task['seed']}.json"


def log_path(task: dict) -> Path:
    return LIVE_LOGS / task["instance_id"] / f"seed_{task['seed']}.parquet"


def valid_result(task: dict) -> dict | None:
    path = result_path(task)
    if not path.exists():
        return None
    try:
        payload = load_json(path)
        live_log = log_path(task)
        if (
            payload.get("status") != "COMPLETE"
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
    rate = completed / elapsed if elapsed > 0 else 0.0
    remaining_seconds = (len(tasks) - completed) / rate if rate > 0 else None
    atomic_json({
        "schema": "phase6h-cal-fit-collection-progress-v1",
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "completed_runs": completed,
        "total_runs": len(tasks),
        "current_process_elapsed_seconds": elapsed,
        "current_process_naive_remaining_seconds": remaining_seconds,
        "status": "COMPLETE" if completed == len(tasks) else "RUNNING",
    }, OUT / "progress.json")


def summarize(tasks: list[dict], config: dict) -> None:
    rows = []
    failures = []
    for task in tasks:
        payload = valid_result(task)
        if payload is None:
            continue
        live_log = pd.read_parquet(log_path(task))
        checks = {
            "all_states_eligible": bool(live_log.ni_eligible.all()),
            "all_selected_actions_evaluated": bool(live_log.ni_intervention.all()),
            "all_candidates_feasible": bool(live_log.candidate_feasible.all()),
            "all_raw_scores_present": bool(live_log.raw_score.notna().all()),
            "all_raw_utilities_present": bool(live_log.raw_utility.notna().all()),
            "outcomes_post_decoder_present": bool(
                live_log.realized_immediate_utility.notna().all()
            ),
            "forced_collection_policy_only": set(live_log.policy_name) == {
                "FORCED_FROZEN_TOP1_COLLECTION"
            },
        }
        if not all(checks.values()):
            failures.append({"task": task, "checks": checks})
        rows.append({
            **task,
            "status": payload["status"],
            "time_limit_seconds": payload["time_limit_seconds"],
            "best_makespan": payload["best_makespan"],
            "time_to_best": payload["time_to_best"],
            "evals_to_best": payload["evals_to_best"],
            "runtime": payload["runtime"],
            "decoder_evaluations": payload["decoder_evaluations"],
            "iterations": payload["iterations"],
            "state_count": len(live_log),
            "realized_positive_fraction": float(live_log.realized_positive.mean()),
            "mean_realized_immediate_utility": float(
                live_log.realized_immediate_utility.mean()
            ),
            "mean_raw_probability": float(live_log.raw_probability.mean()),
            "mean_phase6g_probability": float(
                live_log.calibrated_probability.mean()
            ),
            "feasible": payload["feasible"],
            "live_log_sha256": payload["live_log_sha256"],
        })
    frame = pd.DataFrame(rows)
    atomic_csv(frame, OUT / "collection_run_summary.csv")
    complete = len(frame) == len(tasks) and not failures
    atomic_json({
        "schema": "phase6h-cal-fit-collection-integrity-v1",
        "phase6h_config_sha256": digest(CONFIG_PATH),
        "expected_runs": len(tasks),
        "complete_runs": len(frame),
        "total_live_states": int(frame.state_count.sum()) if len(frame) else 0,
        "all_final_schedules_feasible": bool(frame.feasible.all()) if len(frame) else False,
        "all_labels_post_decoder": not failures,
        "failures": failures,
        "cal_holdout_opened": False,
        "status": "PASS" if complete else "INCOMPLETE",
    }, OUT / "collection_integrity.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit-runs", type=int)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()

    config = load_json(CONFIG_PATH)
    if config.get("status") != "PREREGISTERED_BEFORE_CAL_FIT_COLLECTION":
        raise RuntimeError("Phase 6H protocol is not in the preregistered state")
    instance_root = ROOT / config["calibration_instances"]["root"]
    manifest = pd.read_csv(ROOT / config["calibration_instances"]["manifest"])
    audit = load_json(instance_root / "manifests/calibration_instance_audit.json")
    if audit.get("status") != "PASS":
        raise RuntimeError("Phase 6H calibration instance audit did not pass")
    tasks = build_tasks(config, manifest)
    process_started = time.perf_counter()
    if args.summarize_only:
        summarize(tasks, config)
        write_progress(tasks, process_started)
        print("PHASE6H_CAL_FIT_COLLECTION_SUMMARY_RETURNED", flush=True)
        return
    pending = [task for task in tasks if valid_result(task) is None]
    if args.limit_runs is not None:
        pending = pending[:args.limit_runs]
    print(
        f"PHASE6H_CAL_FIT_COLLECTION_START pending={len(pending)} total={len(tasks)}",
        flush=True,
    )

    phase6g = load_json(ROOT / config["frozen_phase6g_config"])
    load_started = time.perf_counter()
    policy = FrozenLiveInference(
        ROOT / config["frozen_phase6f"]["experiment_freeze"],
        device=args.device,
        proposal_seed_namespace=config["rng_namespaces"]["proposal"],
        force_intervention=True,
    )
    model_load_seconds = time.perf_counter() - load_started
    if policy.checkpoint_sha256 != config["frozen_phase6f"]["checkpoint_sha256"]:
        raise RuntimeError("Phase 6F checkpoint differs from the Phase 6H freeze")
    if phase6g["search"]["candidate_trials"] != config["search"]["candidate_trials"]:
        raise RuntimeError("Phase 6H candidate trials differ from Phase 6G")
    alns_config = read_alns_config()
    csgni_config = CSGNIConfig(
        intervention_rate=int(config["search"]["intervention_rate"]),
        proposal_seed_namespace=int(config["rng_namespaces"]["proposal"]),
        ni_repair_seed_namespace=int(config["rng_namespaces"]["ni_repair"]),
        acceptance_seed_namespace=int(config["rng_namespaces"]["acceptance"]),
        diagnostics_seed_namespace=int(config["rng_namespaces"]["diagnostics"]),
    )

    for index, task in enumerate(pending, 1):
        instance = load_instance(instance_root / task["instance_relative_path"])
        budget = float(config["search"]["wall_clock_seconds_per_operation"]) * instance.num_operations
        observer = Phase6HLiveObserver({
            **task,
            "calibration_split": "CAL_FIT",
            "collection_policy": "FORCED_FROZEN_TOP1",
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
            "schema": "phase6h-cal-fit-collection-run-v1",
            "status": "COMPLETE",
            **task,
            "calibration_split": "CAL_FIT",
            "policy_name": "FORCED_FROZEN_TOP1_COLLECTION",
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
            "checkpoint_sha256": policy.checkpoint_sha256,
            "live_log_relative_path": str(live_log.relative_to(ROOT)),
            "live_log_sha256": digest(live_log),
            "convergence_trace": trace,
            "diagnostics": result.diagnostics,
            "best_solution": result.best.schedule.to_dict(),
            "best_actions": [asdict(action) for action in result.best.actions],
        }
        atomic_json(payload, result_path(task))
        write_progress(tasks, process_started)
        print(
            f"[{index}/{len(pending)}] {task['instance_id']} seed={task['seed']} "
            f"states={len(observer.rows)} makespan={result.best.makespan:g} "
            f"runtime={result.runtime:.2f}s",
            flush=True,
        )

    summarize(tasks, config)
    write_progress(tasks, process_started)
    print("PHASE6H_CAL_FIT_COLLECTION_BATCH_RETURNED", flush=True)


if __name__ == "__main__":
    main()
