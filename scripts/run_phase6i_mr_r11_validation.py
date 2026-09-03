#!/usr/bin/env python3
"""Resumable one-time Phase 6I-MR R11 holdout validation."""

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

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
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
from rcias_clgri.analysis.phase6i_mr import (  # noqa: E402
    decode_forced_candidate,
    score_frozen_candidate_bank,
    select_forced_candidate_roles,
)
from rcias_clgri.data.phase6i_access import load_phase6i_instance  # noqa: E402
from rcias_clgri.env.feasibility import check_schedule  # noqa: E402
from rcias_clgri.heuristic.dispatching import solve_dispatching  # noqa: E402
from rcias_clgri.ni.live_inference import FrozenLiveInference  # noqa: E402
from rcias_clgri.ni.phase6i_live_inference import Phase6IMRLiveInference  # noqa: E402
from rcias_clgri.search.alns import ALNSConfig, solve_alns  # noqa: E402
from rcias_clgri.search.csgni import CSGNIConfig, solve_csgni  # noqa: E402


CONFIG_PATH = ROOT / "configs/phase6i_mr_live_utility_revision.json"
MANIFEST_PATH = (
    ROOT / "instances/controlled/RCIAS-CB1-TRAIN-LIVE-R09R11/manifests"
    / "phase6i_instance_manifest.csv"
)
ARTIFACT_PATH = ROOT / "outputs/phase6i_mr/frozen/selected_artifact.json"
FREEZE_RECORD = ROOT / "outputs/phase6i_mr/frozen/artifact_freeze.json"
PHASE6H_POLICY = ROOT / "outputs/phase6h_calibration/frozen/phase6h_policy.json"
OUT = ROOT / "outputs/phase6i_mr/r11_validation"
RUNS = OUT / "runs"
LIVE_LOGS = OUT / "live_logs"
FORCED = OUT / "forced_diagnostics"
METHODS = ("H1", "ALNS", "PHASE6H_CSGNI", "PHASE6I_MR_CSGNI")
TARGET_FRACTIONS = tuple((index + 0.5) / 10.0 for index in range(10))
CHECKPOINT_FRACTIONS = (0.05, 0.10, 0.25, 0.50, 0.75, 1.0)


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


def cpu_model() -> str:
    for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
        if line.lower().startswith("model name"):
            return line.split(":", 1)[1].strip()
    return "UNKNOWN"


def read_alns_config() -> ALNSConfig:
    raw = load_json(ROOT / "configs/phase5c_alns.json")
    return ALNSConfig(**{
        key: value for key, value in raw.items()
        if key in ALNSConfig.__dataclass_fields__
    })


def verify_unlock() -> tuple[dict, dict, str, str]:
    config = load_json(CONFIG_PATH)
    artifact = load_json(ARTIFACT_PATH)
    record = load_json(FREEZE_RECORD)
    artifact_hash = digest(ARTIFACT_PATH)
    freeze_hash = digest(FREEZE_RECORD)
    checks = [
        record.get("schema") == "phase6i-mr-artifact-freeze-v1",
        record.get("status") == "FROZEN_BEFORE_R11",
        record.get("r11_content_accessed") is False,
        record.get("artifact_sha256") == artifact_hash,
        artifact.get("status") == "FROZEN_BEFORE_R11",
        artifact.get("r11_accessed") is False,
        artifact.get("r11_protocol", {}).get("seeds") == [
            681401, 681402, 681403, 681404, 681405
        ],
        artifact.get("code_hashes", {}).get(
            "scripts/run_phase6i_mr_r11_validation.py"
        ) == digest(Path(__file__)),
        artifact.get("source_hashes", {}).get("config") == digest(CONFIG_PATH),
    ]
    if not all(checks):
        raise RuntimeError("R11 selected-artifact freeze is invalid")
    return config, artifact, artifact_hash, freeze_hash


def build_tasks(config: dict) -> list[dict]:
    manifest = pd.read_csv(MANIFEST_PATH)
    selected = manifest[manifest.replicate.eq("R11")].sort_values(
        ["scale", "CF_level", "cell_replicate", "instance_id"], kind="stable"
    )
    if len(selected) != 18 or selected.instance_id.nunique() != 18:
        raise RuntimeError("R11 must contain exactly 18 unique instances")
    seeds = tuple(int(seed) for seed in config["r11_protocol"]["seeds"])
    if seeds != (681401, 681402, 681403, 681404, 681405):
        raise RuntimeError("R11 seed protocol changed")
    tasks = []
    for row in selected.itertuples(index=False):
        common = {
            "instance_id": row.instance_id,
            "instance_relative_path": row.relative_path,
            "instance_sha256": row.sha256,
            "scale": row.scale,
            "CF_level": row.CF_level,
            "cell_replicate": row.cell_replicate,
        }
        tasks.append({**common, "method": "H1", "seed": None})
        for seed in seeds:
            for method in METHODS[1:]:
                tasks.append({**common, "method": method, "seed": seed})
    if len(tasks) != 288:
        raise RuntimeError("R11 task count must be 288")
    return tasks


def result_path(task: dict) -> Path:
    filename = "result.json" if task["seed"] is None else f"seed_{task['seed']}.json"
    return RUNS / task["method"] / task["instance_id"] / filename


def live_log_path(task: dict) -> Path:
    return LIVE_LOGS / task["method"] / task["instance_id"] / f"seed_{task['seed']}.parquet"


def forced_path(task: dict) -> Path:
    return FORCED / task["instance_id"] / f"seed_{task['seed']}.parquet"


def valid_result(task: dict, artifact_hash: str, freeze_hash: str) -> dict | None:
    path = result_path(task)
    if not path.is_file():
        return None
    try:
        payload = load_json(path)
        if not all([
            payload.get("status") == "COMPLETE",
            payload.get("method") == task["method"],
            payload.get("instance_id") == task["instance_id"],
            payload.get("seed") == task["seed"],
            payload.get("instance_sha256") == task["instance_sha256"],
            payload.get("selected_artifact_sha256") == artifact_hash,
            payload.get("artifact_freeze_sha256") == freeze_hash,
            payload.get("r11_accessed") is True,
        ]):
            return None
        if task["method"] in {"PHASE6H_CSGNI", "PHASE6I_MR_CSGNI"}:
            log = live_log_path(task)
            if not log.is_file() or payload.get("live_log_sha256") != digest(log):
                return None
        if task["method"] == "PHASE6I_MR_CSGNI":
            forced = forced_path(task)
            if not all([
                forced.is_file(),
                payload.get("forced_diagnostics_sha256") == digest(forced),
                payload.get("forced_diagnostic_states") == 10,
                payload.get("forced_diagnostic_actions") == 40,
            ]):
                return None
        return payload
    except (json.JSONDecodeError, OSError, TypeError, KeyError):
        return None


class R11Observer:
    def __init__(self, metadata: dict) -> None:
        self.live = Phase6HLiveObserver(metadata)
        self.events: list[dict] = []

    def __call__(self, event: dict) -> None:
        self.live(event)
        if not event.get("ni_eligible") or event.get("ni_state_id") is None:
            return
        self.events.append({
            "state_id": str(event["ni_state_id"]),
            "iteration": int(event["iteration"]),
            "search_progress": float(
                (event.get("ni_state_feature_summary") or {}).get(
                    "search_progress", 0.0
                )
            ),
            "elapsed_wall_time": float(event["elapsed_time"]),
            "decoder_evaluations": int(event["decoder_evaluations"]),
            "current": event["current_before"],
        })

    def snapshots(self) -> list[dict]:
        if len(self.events) < len(TARGET_FRACTIONS):
            raise RuntimeError("R11 run lacks ten forced-diagnostic states")
        selected = []
        used: set[int] = set()
        for target in TARGET_FRACTIONS:
            index = min(
                (i for i in range(len(self.events)) if i not in used),
                key=lambda i: (
                    round(abs(self.events[i]["search_progress"] - target), 12),
                    self.events[i]["iteration"],
                ),
            )
            used.add(index)
            selected.append({**self.events[index], "target_progress": target})
        return selected


def forced_diagnostics(
    task: dict,
    instance,
    observer: R11Observer,
    reference: FrozenLiveInference,
    revised: Phase6IMRLiveInference,
    config: dict,
    alns_config: ALNSConfig,
) -> pd.DataFrame:
    rows = []
    count = min(
        max(2, round(instance.num_operations * alns_config.destroy_fraction)),
        instance.num_operations,
    )
    for snapshot in observer.snapshots():
        progress = float(snapshot["search_progress"])
        stage_bins = ("0-20%", "20-40%", "40-60%", "60-80%", "80-100%")
        stage = stage_bins[min(4, int(max(0.0, min(progress, .999999)) * 5))]
        current = snapshot["current"]
        bank = score_frozen_candidate_bank(
            reference,
            instance,
            current,
            state_id=snapshot["state_id"],
            destroy_count=count,
            search_progress=progress,
            search_stage=stage,
        )
        roles = select_forced_candidate_roles(bank.arms)
        revised_evaluation = revised.evaluate(
            instance,
            current,
            state_id=snapshot["state_id"],
            destroy_count=count,
            search_progress=progress,
            search_stage=stage,
        )
        predictions = {
            row["target_set_id"]: row
            for row in revised_evaluation.candidate_diagnostics
        }
        state_rows = []
        for role in roles:
            decoded = decode_forced_candidate(
                instance,
                current,
                role.arm,
                state_id=snapshot["state_id"],
                repair_seed_namespace=int(config["rng_namespaces"]["forced_repair"]),
                candidate_trials=int(config["search"]["repair_decoder_trials_per_target"]),
            )
            utility = (current.makespan - decoded.candidate.makespan) / current.makespan
            prediction = predictions[role.arm.target_set_id]
            state_rows.append({
                **{key: task[key] for key in (
                    "instance_id", "scale", "CF_level", "cell_replicate", "seed"
                )},
                "state_id": snapshot["state_id"],
                "iteration": snapshot["iteration"],
                "target_progress": snapshot["target_progress"],
                "search_progress": progress,
                "search_stage": stage,
                "candidate_role": role.role,
                "target_set_id": role.arm.target_set_id,
                "ensemble_raw_score": prediction["raw_score"],
                "ensemble_raw_value": prediction["ensemble_raw_utility"],
                "calibrated_probability": prediction["calibrated_probability"],
                "calibrated_utility": prediction["calibrated_utility"],
                "supported": revised_evaluation.decision.support_in_range,
                "decoded_immediate_utility": utility,
                "positive_label": utility > 0,
                "candidate_feasible": decoded.candidate.feasible,
                "forced_decoder_evaluations": decoded.decoder_evaluations,
                "forced_decode_ms": decoded.runtime_ms,
                "labels_post_trajectory": True,
                "r11_accessed": True,
            })
        fallback = next(
            row for row in state_rows
            if row["candidate_role"] == "ALNS_RELATED_FALLBACK"
        )
        best = max(row["decoded_immediate_utility"] for row in state_rows)
        for row in state_rows:
            row["fallback_target_set_id"] = fallback["target_set_id"]
            row["fallback_decoded_utility"] = fallback["decoded_immediate_utility"]
            row["regret_to_best"] = best - row["decoded_immediate_utility"]
        rows.extend(state_rows)
    frame = pd.DataFrame(rows)
    if not all([
        frame.state_id.nunique() == 10,
        len(frame) == 40,
        frame.groupby("state_id").size().eq(4).all(),
        frame.candidate_feasible.astype(bool).all(),
        frame.labels_post_trajectory.astype(bool).all(),
    ]):
        raise RuntimeError("R11 forced-diagnostic integrity failed")
    return frame


def write_progress(tasks: list[dict], artifact_hash: str, freeze_hash: str) -> None:
    complete = [
        task for task in tasks
        if valid_result(task, artifact_hash, freeze_hash) is not None
    ]
    atomic_json({
        "schema": "phase6i-mr-r11-validation-progress-v1",
        "status": "COMPLETE" if len(complete) == len(tasks) else "RUNNING",
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "completed_runs": len(complete),
        "total_runs": len(tasks),
        "completed_by_method": {
            method: sum(task["method"] == method for task in complete)
            for method in METHODS
        },
        "r11_accessed": len(complete) > 0,
    }, OUT / "progress.json")


def summarize(tasks: list[dict], artifact_hash: str, freeze_hash: str) -> None:
    records = [valid_result(task, artifact_hash, freeze_hash) for task in tasks]
    rows = [{
        "method": record["method"],
        "instance_id": record["instance_id"],
        "scale": record["scale"],
        "CF_level": record["CF_level"],
        "cell_replicate": record["cell_replicate"],
        "seed": record["seed"],
        "final_best": record["final_best"],
        "total_runtime": record["total_runtime"],
        "total_decoder_evals": record["total_decoder_evals"],
        "time_to_best": record["time_to_best"],
        "evals_to_best": record["evals_to_best"],
        "feasible": record["feasible"],
        "model_load_seconds": record.get("model_load_seconds_excluded_from_budget"),
        "candidate_proposal_seconds": record["runtime_components"]["candidate_proposal_seconds"],
        "csg_seconds": record["runtime_components"]["csg_seconds"],
        "neural_seconds": record["runtime_components"]["neural_seconds"],
        "calibration_gate_seconds": record["runtime_components"]["calibration_gate_seconds"],
        "repair_seconds": record["runtime_components"]["repair_seconds"],
        "interventions": record["diagnostics"].get("ni_interventions", 0),
        "fallbacks": record["diagnostics"].get("ni_fallbacks", 0),
    } for record in records if record is not None]
    atomic_csv(pd.DataFrame(rows), OUT / "validation_run_summary.csv")
    forced_frames = [
        pd.read_parquet(forced_path(task))
        for task in tasks if task["method"] == "PHASE6I_MR_CSGNI"
    ]
    atomic_parquet(pd.concat(forced_frames, ignore_index=True), OUT / "forced_diagnostics.parquet")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config, artifact, artifact_hash, freeze_hash = verify_unlock()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("R11 frozen protocol requires an available CUDA device")
    tasks = build_tasks(config)
    pending = [
        task for task in tasks
        if valid_result(task, artifact_hash, freeze_hash) is None
    ]
    write_progress(tasks, artifact_hash, freeze_hash)
    access_ledger = OUT / "r11_access_ledger.json"
    if not access_ledger.exists():
        atomic_json({
            "schema": "phase6i-mr-r11-access-ledger-v1",
            "status": "UNLOCKED_FOR_SINGLE_RESUMABLE_PASS",
            "first_access_authorized_at_utc": datetime.now(timezone.utc).isoformat(),
            "selected_artifact_sha256": artifact_hash,
            "artifact_freeze_sha256": freeze_hash,
            "allowed_split": "R11_LIVE_REV_HOLDOUT",
            "allowed_runner_sha256": digest(Path(__file__)),
            "retuning_forbidden": True,
        }, access_ledger)
    atomic_json({
        "schema": "phase6i-mr-r11-runtime-environment-v1",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "cpu_threads_per_solver": 1,
        "cpu_model": cpu_model(),
        "worker_count": 1,
        "solver_concurrency": 1,
        "device": args.device,
        "gpu_model": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "batch_size": 1,
        "warmup_policy": "NO_SEPARATE_WARMUP_RUN",
        "checkpoint_load_handling": "LOAD_ONCE_REPORTED_SEPARATELY_EXCLUDED_FROM_RUN_BUDGET",
    }, OUT / "runtime_environment.json")
    print(f"PHASE6I_MR_R11_START pending={len(pending)} total={len(tasks)}", flush=True)
    alns_config = read_alns_config()
    csgni_config = CSGNIConfig(
        intervention_rate=int(config["search"]["intervention_rate"]),
        proposal_seed_namespace=int(config["rng_namespaces"]["frozen_live_proposal"]),
        ni_repair_seed_namespace=int(config["rng_namespaces"]["frozen_live_repair"]),
        acceptance_seed_namespace=int(config["rng_namespaces"]["frozen_live_acceptance"]),
        diagnostics_seed_namespace=int(config["rng_namespaces"]["frozen_live_diagnostics"]),
    )
    load_started = time.perf_counter()
    phase6h = FrozenLiveInference(
        ROOT / config["locked_inputs"]["phase6f_experiment_freeze"],
        device=args.device,
        proposal_seed_namespace=config["rng_namespaces"]["frozen_live_proposal"],
        deployment_artifact=PHASE6H_POLICY,
    )
    phase6h_load = time.perf_counter() - load_started
    load_started = time.perf_counter()
    revised = Phase6IMRLiveInference(
        ROOT, ARTIFACT_PATH, device=args.device, required_status="FROZEN_BEFORE_R11"
    )
    revised_load = time.perf_counter() - load_started
    instance_root = ROOT / config["instance_suite"]["root"]
    budget_factor = 2.0
    for index, task in enumerate(pending, 1):
        instance_path = instance_root / task["instance_relative_path"]
        if digest(instance_path) != task["instance_sha256"]:
            raise RuntimeError(f"R11 instance hash mismatch: {task['instance_id']}")
        instance = load_phase6i_instance(
            instance_path,
            freeze_record_path=FREEZE_RECORD,
            artifact_path=ARTIFACT_PATH,
        )
        method = task["method"]
        observer = None
        alns_timing_rows = []
        if method == "H1":
            result = solve_dispatching(instance, "H1")
            final_best = float(result.objective.makespan)
            trace = [{"elapsed_time": float(result.runtime_seconds), "decoder_evaluations": 1, "current_best_makespan": final_best}]
            total_runtime = float(result.runtime_seconds)
            total_evals = 1
            iterations = 0
            initialization_seconds = total_runtime
            best_schedule = result.schedule
            best_actions = result.actions
            diagnostics = {}
            budget = None
        else:
            budget = budget_factor * instance.num_operations
            if method == "ALNS":
                result = solve_alns(
                    instance, budget, task["seed"], alns_config,
                    observer=lambda event: alns_timing_rows.append({
                        "repair_excluding_decoder_ms": float(
                            event.get(
                                "repair_excluding_decoder_runtime",
                                event["repair_runtime"],
                            )
                        ) * 1000.0,
                        "decoder_time_ms": float(
                            event.get("decoder_runtime", 0.0)
                        ) * 1000.0,
                    }),
                )
            elif method == "PHASE6H_CSGNI":
                observer = Phase6HLiveObserver({**task, "policy_name": phase6h.policy_name})
                result = solve_csgni(instance, budget, task["seed"], phase6h, alns_config=alns_config, csgni_config=csgni_config, observer=observer)
            else:
                observer = R11Observer({**task, "policy_name": revised.policy_name})
                result = solve_csgni(instance, budget, task["seed"], revised, alns_config=alns_config, csgni_config=csgni_config, observer=observer)
            final_best = float(result.best.makespan)
            trace = validate_incumbent_trace(result.convergence_trace, final_best=final_best)
            total_runtime = float(result.runtime)
            total_evals = int(result.decoder_evaluations)
            iterations = int(result.iterations)
            initialization_seconds = float(result.diagnostics.get("initialization_seconds", trace[0]["elapsed_time"]))
            best_schedule = result.best.schedule
            best_actions = result.best.actions
            diagnostics = result.diagnostics
        feasibility = check_schedule(instance, best_schedule)
        if not feasibility["feasible"]:
            raise RuntimeError({"task": task, "violations": feasibility["violations"]})
        live_frame = None
        if observer is not None:
            live_frame = pd.DataFrame(observer.live.rows if isinstance(observer, R11Observer) else observer.rows)
            log_path = live_log_path(task)
            atomic_parquet(live_frame, log_path)
            live_hash = digest(log_path)
        else:
            live_hash = None
        forced_hash = None
        forced_states = forced_actions = 0
        if isinstance(observer, R11Observer):
            forced = forced_diagnostics(task, instance, observer, phase6h, revised, config, alns_config)
            path = forced_path(task)
            atomic_parquet(forced, path)
            forced_hash = digest(path)
            forced_states = forced.state_id.nunique()
            forced_actions = len(forced)
        timings = (
            live_frame if live_frame is not None else pd.DataFrame(alns_timing_rows)
        )
        component = {
            "candidate_proposal_seconds": float(timings.get("target_bank_ms", pd.Series(dtype=float)).sum() / 1000.0),
            "csg_seconds": float(timings.get("csg_build_ms", pd.Series(dtype=float)).sum() / 1000.0),
            "neural_seconds": float((timings.get("model_inference_ms", pd.Series(dtype=float)).sum() + timings.get("action_scoring_ms", pd.Series(dtype=float)).sum()) / 1000.0),
            "calibration_gate_seconds": float(timings.get("calibration_gate_ms", pd.Series(dtype=float)).sum() / 1000.0),
            "repair_seconds": float(timings.get("repair_excluding_decoder_ms", pd.Series(dtype=float)).sum() / 1000.0),
            "decoder_seconds": float(timings.get("decoder_time_ms", pd.Series(dtype=float)).sum() / 1000.0),
        }
        payload = {
            "schema": "phase6i-mr-r11-validation-run-v1",
            "status": "COMPLETE",
            **task,
            "time_limit_seconds": budget,
            "final_best": final_best,
            "time_to_best": float(trace[-1]["elapsed_time"]),
            "evals_to_best": int(trace[-1]["decoder_evaluations"]),
            "total_runtime": total_runtime,
            "total_decoder_evals": total_evals,
            "initialization_seconds": initialization_seconds,
            "iterations": iterations,
            "feasible": True,
            "model_load_seconds_excluded_from_budget": ({"PHASE6H_CSGNI": phase6h_load, "PHASE6I_MR_CSGNI": revised_load}.get(method)),
            "selected_artifact_sha256": artifact_hash,
            "artifact_freeze_sha256": freeze_hash,
            "live_log_sha256": live_hash,
            "forced_diagnostics_sha256": forced_hash,
            "forced_diagnostic_states": forced_states,
            "forced_diagnostic_actions": forced_actions,
            "incumbent_trace": trace,
            "normalized_budget_checkpoints": [] if budget is None else sample_incumbent_trace(trace, budget=budget, fractions=CHECKPOINT_FRACTIONS),
            "runtime_components": component,
            "diagnostics": diagnostics,
            "best_solution": best_schedule.to_dict(),
            "best_actions": [asdict(action) for action in best_actions],
            "r11_accessed": True,
        }
        atomic_json(payload, result_path(task))
        write_progress(tasks, artifact_hash, freeze_hash)
        print(f"[{index}/{len(pending)}] {method} {task['instance_id']} seed={task['seed']} makespan={final_best:g} evals={total_evals} runtime={total_runtime:.2f}s", flush=True)
    summarize(tasks, artifact_hash, freeze_hash)
    write_progress(tasks, artifact_hash, freeze_hash)
    print("PHASE6I_MR_R11_BATCH_RETURNED", flush=True)


if __name__ == "__main__":
    main()
