#!/usr/bin/env python3
"""Run the resumable R09 rapid-failure-analysis pilot for Phase 6I-MR."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_clgri.analysis.phase6a import schedule_features  # noqa: E402
from rcias_clgri.analysis.phase6h import validate_incumbent_trace  # noqa: E402
from rcias_clgri.analysis.phase6i_mr import (  # noqa: E402
    decode_forced_candidate,
    score_frozen_candidate_bank,
    select_forced_candidate_roles,
    select_top_eight_audit_candidates,
)
from rcias_clgri.data.phase6i_access import load_phase6i_instance  # noqa: E402
from rcias_clgri.env.feasibility import check_schedule  # noqa: E402
from rcias_clgri.heuristic.dispatching import solve_dispatching  # noqa: E402
from rcias_clgri.ni.live_inference import FrozenLiveInference  # noqa: E402
from rcias_clgri.search.alns import ALNSConfig  # noqa: E402
from rcias_clgri.search.common import Candidate, candidate_from_actions, decode_candidate  # noqa: E402
from rcias_clgri.search.csgni import CSGNIConfig, solve_csgni  # noqa: E402


CONFIG_PATH = ROOT / "configs/phase6i_mr_live_utility_revision.json"
OUT = ROOT / "outputs/phase6i_mr/pilot_v12"
RUNS = OUT / "runs"
FORCED = OUT / "forced_actions"
FULL_BANK = OUT / "full_bank_audit"
REPLAYS = OUT / "state_replays"


def configure_output_root(path: Path) -> None:
    global OUT, RUNS, FORCED, FULL_BANK, REPLAYS
    OUT = path
    RUNS = OUT / "runs"
    FORCED = OUT / "forced_actions"
    FULL_BANK = OUT / "full_bank_audit"
    REPLAYS = OUT / "state_replays"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


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


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_alns_config() -> ALNSConfig:
    raw = load_json(ROOT / "configs/phase5c_alns.json")
    return ALNSConfig(**{
        key: value
        for key, value in raw.items()
        if key in ALNSConfig.__dataclass_fields__
    })


def _candidate_dict(candidate: Candidate) -> dict[str, list[str]]:
    return {
        "operation_order": list(candidate.operation_order),
        "island_assignment": list(candidate.island_assignment),
        "w_assignment": list(candidate.w_assignment),
        "f_assignment": list(candidate.f_assignment),
    }


def _candidate_from_dict(payload: dict[str, list[str]]) -> Candidate:
    return Candidate(
        tuple(payload["operation_order"]),
        tuple(payload["island_assignment"]),
        tuple(payload["w_assignment"]),
        tuple(payload["f_assignment"]),
    )


def _search_stage(progress: float) -> str:
    bounds = ("0-20%", "20-40%", "40-60%", "60-80%", "80-100%")
    return bounds[min(4, int(max(0.0, min(progress, 0.999999)) * 5))]


def _diagnostic_stage(target_progress: float) -> str:
    if target_progress <= 0.25:
        return "EARLY"
    if target_progress <= 0.60:
        return "MIDDLE"
    return "LATE"


class SnapshotObserver:
    """Retain outcome-blind source states; forced decoding occurs after solve returns."""

    def __init__(self, budget: float, target_fractions: list[float]) -> None:
        self.budget = budget
        self.target_fractions = target_fractions
        self.events: list[dict[str, object]] = []

    def __call__(self, event: dict[str, object]) -> None:
        if not event.get("ni_eligible") or event.get("ni_state_id") is None:
            return
        features = dict(event.get("ni_state_feature_summary") or {})
        current = event["current_before"]
        progress = float(features.get("search_progress", 0.0))
        self.events.append({
            "state_id": str(event["ni_state_id"]),
            "iteration": int(event["iteration"]),
            "search_progress": progress,
            "elapsed_wall_time": float(event["elapsed_time"]),
            "current_makespan": float(current.makespan),
            "current_candidate": _candidate_dict(current.candidate),
        })

    def selected(self) -> list[dict[str, object]]:
        if len(self.events) < len(self.target_fractions):
            raise RuntimeError(
                f"pilot trajectory produced {len(self.events)} states for "
                f"{len(self.target_fractions)} required snapshots"
            )
        selected = []
        used: set[int] = set()
        for target in self.target_fractions:
            index = min(
                (index for index in range(len(self.events)) if index not in used),
                key=lambda index: (
                    round(
                        abs(float(self.events[index]["search_progress"]) - target),
                        12,
                    ),
                    int(self.events[index]["iteration"]),
                ),
            )
            used.add(index)
            selected.append({
                **self.events[index],
                "target_progress": float(target),
                "diagnostic_stage": _diagnostic_stage(float(target)),
            })
        return selected


def _dag_depth(instance) -> int:
    depths: dict[str, int] = {}
    pending = set(instance.operations)
    while pending:
        ready = sorted(
            operation
            for operation in pending
            if instance.predecessors[operation].issubset(depths)
        )
        if not ready:
            raise RuntimeError("precedence graph is cyclic")
        for operation in ready:
            depths[operation] = 1 + max(
                (depths[parent] for parent in instance.predecessors[operation]),
                default=0,
            )
            pending.remove(operation)
    return max(depths.values(), default=0)


def _dag_width(instance) -> int:
    completed: set[str] = set()
    maximum = 0
    while len(completed) < instance.num_operations:
        ready = sorted(
            operation
            for operation in instance.operations
            if operation not in completed
            and instance.predecessors[operation].issubset(completed)
        )
        if not ready:
            raise RuntimeError("precedence graph is cyclic")
        maximum = max(maximum, len(ready))
        completed.add(ready[0])
    return maximum


def _critical_path_proxy(instance, schedule) -> float:
    features = schedule_features(instance, schedule)
    return float(sum(
        schedule.operation_schedules[operation].processing_time
        for operation in instance.operations
        if features[operation]["is_on_processing_critical_path"]
    ))


def _state_context(instance, current, h1_schedule, bank) -> dict[str, object]:
    schedule = current.schedule
    makespan = max(current.makespan, 1.0)
    operation_records = schedule.operation_schedules.values()
    island_counts = np.asarray(
        list(Counter(record.island_id for record in operation_records).values()),
        dtype=float,
    )
    w_delay = float(sum(
        max(0.0, record.w_ready_time - record.product_ready_time)
        for record in schedule.operation_schedules.values()
    ))
    f_delay = float(sum(
        max(0.0, record.f_ready_time - record.product_ready_time)
        for record in schedule.operation_schedules.values()
    ))
    reconfiguration = float(sum(
        record.reconfiguration_end - record.reconfiguration_start
        for record in schedule.operation_schedules.values()
    ))
    h1_makespan = max(
        record.completion_time for record in h1_schedule.operation_schedules.values()
    )
    current_critical = _critical_path_proxy(instance, schedule)
    h1_critical = max(_critical_path_proxy(instance, h1_schedule), 1.0)
    node_counts = {key: len(value) for key, value in bank.graph.nodes.items()}
    edge_counts = {key: len(value) for key, value in bank.graph.edges.items()}
    total_nodes = sum(node_counts.values())
    total_edges = sum(edge_counts.values())
    initial_workload = float(sum(
        min(
            instance.processing_time[(operation, island)]
            for island in instance.operation_data[operation].eligible_islands
        )
        for operation in instance.operations
    ))
    context = {
        "node_counts_by_type": json.dumps(node_counts, sort_keys=True),
        "edge_counts_by_type": json.dumps(edge_counts, sort_keys=True),
        "graph_node_count": total_nodes,
        "graph_edge_count": total_edges,
        "operation_count": instance.num_operations,
        "dag_depth_proxy": _dag_depth(instance),
        "dag_width_proxy": _dag_width(instance),
        "eligibility_density": float(np.mean([
            len(instance.operation_data[operation].eligible_islands)
            / max(1, len(instance.islands))
            for operation in instance.operations
        ])),
        "resource_load_cv": float(
            np.std(island_counts) / max(np.mean(island_counts), 1e-12)
        ),
        "current_makespan": float(current.makespan),
        "h1_makespan": float(h1_makespan),
        "current_makespan_over_h1_makespan": float(current.makespan / h1_makespan),
        "current_critical_path_proxy": current_critical,
        "h1_critical_path_proxy": h1_critical,
        "critical_path_over_h1_critical_path": current_critical / h1_critical,
        "remaining_operations": 0,
        "remaining_workload": 0.0,
        "initial_processing_workload": initial_workload,
        "remaining_workload_ratio": 0.0,
        "w_delay_total": w_delay,
        "f_delay_total": f_delay,
        "reconfiguration_total": reconfiguration,
        "w_delay_over_current_makespan": w_delay / makespan,
        "f_delay_over_current_makespan": f_delay / makespan,
        "reconfiguration_over_current_makespan": reconfiguration / makespan,
    }
    context.update(bank.state_feature_summary)
    return context


def _rank_and_label(rows: list[dict[str, object]]) -> None:
    by_state: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_state.setdefault(str(row["state_id"]), []).append(row)
    for state_rows in by_state.values():
        truth = sorted(
            state_rows,
            key=lambda row: (
                -float(row["decoded_immediate_utility"]),
                str(row["target_set_id"]),
            ),
        )
        predicted = sorted(
            state_rows,
            key=lambda row: (-float(row["raw_score"]), str(row["target_set_id"])),
        )
        true_rank = {id(row): rank for rank, row in enumerate(truth, 1)}
        predicted_rank = {id(row): rank for rank, row in enumerate(predicted, 1)}
        best_utility = float(truth[0]["decoded_immediate_utility"])
        fallback = next(
            row for row in state_rows if row["candidate_role"] == "ALNS_RELATED_FALLBACK"
        )
        top1 = next(
            row for row in state_rows if row["candidate_role"] == "FROZEN_NEURAL_TOP1"
        )
        inversion = float(top1["decoded_immediate_utility"]) < best_utility - 1e-12
        for row in state_rows:
            decoded = float(row["decoded_immediate_utility"])
            predicted_utility = float(row["calibrated_utility"])
            row["within_state_true_rank"] = true_rank[id(row)]
            row["within_state_predicted_rank"] = predicted_rank[id(row)]
            row["regret_to_best"] = best_utility - decoded
            row["fallback_target_set_id"] = fallback["target_set_id"]
            row["fallback_decoded_utility"] = fallback["decoded_immediate_utility"]
            row["within_state_inversion"] = inversion
            row["sign_error"] = (predicted_utility > 0) != (decoded > 0)


def build_tasks(config: dict, manifest: pd.DataFrame) -> list[dict]:
    fit = manifest[manifest.live_revision_split == "LIVE_REV_FIT"]
    fit = fit[fit.cell_replicate == config["pilot"]["cell_replicate"]]
    if len(fit) != 9 or set(fit.replicate) != {"R09"}:
        raise RuntimeError("formal pilot must be the nine C02 R09 instances")
    seed = int(config["seeds"]["R09_PILOT_TRAJECTORY"][0])
    return [
        {
            "instance_id": row.instance_id,
            "instance_relative_path": row.relative_path,
            "scale": row.scale,
            "CF_level": row.CF_level,
            "RI_level": row.RI_level,
            "TI_level": row.TI_level,
            "seed": seed,
        }
        for row in fit.sort_values("instance_id").itertuples(index=False)
    ]


def result_path(task: dict) -> Path:
    return RUNS / f"{task['instance_id']}.json"


def forced_path(task: dict) -> Path:
    return FORCED / f"{task['instance_id']}.parquet"


def full_bank_path(task: dict) -> Path:
    return FULL_BANK / f"{task['instance_id']}.parquet"


def valid_result(task: dict) -> dict | None:
    path = result_path(task)
    table = forced_path(task)
    audit_table = full_bank_path(task)
    if not path.exists() or not table.exists() or not audit_table.exists():
        return None
    try:
        result = load_json(path)
    except (OSError, json.JSONDecodeError):
        return None
    if (
        result.get("status") != "COMPLETE"
        or result.get("forced_actions_sha256") != digest(table)
        or int(result.get("sampled_states", -1)) != 6
        or int(result.get("forced_actions", -1)) != 24
        or result.get("full_bank_audit_sha256") != digest(audit_table)
        or int(result.get("full_bank_audit_actions", -1)) < 8
    ):
        return None
    return result


def write_progress(tasks: list[dict], started: float) -> None:
    complete = sum(valid_result(task) is not None for task in tasks)
    elapsed = time.perf_counter() - started
    rate = complete / elapsed if elapsed > 0 else 0.0
    remaining = (len(tasks) - complete) / rate if rate > 0 else None
    atomic_json({
        "schema": "phase6i-mr-r09-pilot-progress-v1",
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "completed_runs": complete,
        "total_runs": len(tasks),
        "current_process_elapsed_seconds": elapsed,
        "current_process_naive_remaining_seconds": remaining,
        "status": "COMPLETE" if complete == len(tasks) else "RUNNING",
    }, OUT / "progress.json")


def summarize(tasks: list[dict]) -> None:
    run_rows = []
    action_frames = []
    audit_frames = []
    failures = []
    for task in tasks:
        result = valid_result(task)
        if result is None:
            continue
        frame = pd.read_parquet(forced_path(task))
        audit_frame = pd.read_parquet(full_bank_path(task))
        checks = {
            "six_unique_states": frame.state_id.nunique() == 6,
            "four_actions_per_state": frame.groupby("state_id").size().eq(4).all(),
            "four_roles_per_state": frame.groupby("state_id").candidate_role.nunique().eq(4).all(),
            "unique_targets_per_state": frame.groupby("state_id").target_set_id.nunique().eq(4).all(),
            "all_labels_post_trajectory": bool(frame.labels_post_trajectory.all()),
            "all_feasible": bool(frame.candidate_feasible.all()),
            "one_full_bank_state": audit_frame.state_id.nunique() == 1,
            "top_eight_exact": int(audit_frame.in_top_eight.sum()) == 8,
            "broad_four_retained": int(audit_frame.in_broad_four.sum()) == 4,
            "full_bank_unique": (
                audit_frame.target_set_id.nunique() == len(audit_frame)
                and len(audit_frame) >= 8
            ),
            "full_bank_all_feasible": bool(audit_frame.candidate_feasible.all()),
        }
        if not all(checks.values()):
            failures.append({"task": task, "checks": checks})
        run_rows.append({
            **task,
            "runtime": result["runtime"],
            "time_limit_seconds": result["time_limit_seconds"],
            "best_makespan": result["best_makespan"],
            "iterations": result["iterations"],
            "decoder_evaluations": result["decoder_evaluations"],
            "sampled_states": len(frame.state_id.unique()),
            "forced_actions": len(frame),
            "forced_decoder_evaluations": int(frame.forced_decoder_evaluations.sum()),
            "mean_forced_decode_ms": float(frame.forced_decode_ms.mean()),
            "feasible": result["feasible"],
        })
        action_frames.append(frame)
        audit_frames.append(audit_frame)
    runs = pd.DataFrame(run_rows)
    actions = pd.concat(action_frames, ignore_index=True) if action_frames else pd.DataFrame()
    audit_actions = (
        pd.concat(audit_frames, ignore_index=True) if audit_frames else pd.DataFrame()
    )
    atomic_csv(runs, OUT / "pilot_run_summary.csv")
    if len(actions):
        atomic_parquet(actions, OUT / "forced_action_failure_table.parquet")
        atomic_csv(actions, OUT / "forced_action_failure_table.csv")
    if len(audit_actions):
        atomic_parquet(audit_actions, OUT / "full_bank_audit_table.parquet")
        atomic_csv(audit_actions, OUT / "full_bank_audit_table.csv")
    complete = len(runs) == len(tasks) and not failures
    atomic_json({
        "schema": "phase6i-mr-r09-pilot-integrity-v1",
        "phase6i_config_sha256": digest(CONFIG_PATH),
        "expected_runs": len(tasks),
        "complete_runs": len(runs),
        "expected_states": 54,
        "complete_states": int(actions.state_id.nunique()) if len(actions) else 0,
        "expected_forced_actions": 216,
        "complete_forced_actions": len(actions),
        "expected_full_bank_states": 9,
        "complete_full_bank_states": (
            int(audit_actions.state_id.nunique()) if len(audit_actions) else 0
        ),
        "complete_full_bank_actions": len(audit_actions),
        "all_eight_and_true_full_bank_logic_verified": not failures,
        "all_source_trajectories_completed_before_forced_labels": not failures,
        "r10_accessed": False,
        "r11_accessed": False,
        "failures": failures,
        "status": "PASS" if complete else "INCOMPLETE",
    }, OUT / "pilot_integrity.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--protocol-revision", choices=["1.2"], default="1.2")
    parser.add_argument("--limit-runs", type=int)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()

    config = load_json(CONFIG_PATH)
    if config.get("status") != (
        "REVISION_1_2_PREREGISTERED_BEFORE_FORMAL_R09_COLLECTION_OR_R10_R11_CONTENT_ACCESS"
    ):
        raise RuntimeError("Phase 6I-MR Revision 1.2 config is not preregistered")
    configure_output_root(ROOT / config["pilot"]["output_root"])
    instance_root = ROOT / config["instance_suite"]["root"]
    manifest = pd.read_csv(ROOT / config["instance_suite"]["manifest"])
    audit = load_json(instance_root / "manifests/phase6i_integrity_audit.json")
    if audit.get("status") != "PASS":
        raise RuntimeError("Phase 6I-MR instance integrity audit did not pass")
    tasks = build_tasks(config, manifest)
    process_started = time.perf_counter()
    if args.summarize_only:
        summarize(tasks)
        write_progress(tasks, process_started)
        print("PHASE6I_MR_R09_PILOT_SUMMARY_RETURNED", flush=True)
        return

    pending = [task for task in tasks if valid_result(task) is None]
    if args.limit_runs is not None:
        pending = pending[: args.limit_runs]
    print(
        f"PHASE6I_MR_R09_PILOT_START pending={len(pending)} total={len(tasks)}",
        flush=True,
    )
    load_started = time.perf_counter()
    policy = FrozenLiveInference(
        ROOT / config["locked_inputs"]["phase6f_experiment_freeze"],
        device=args.device,
        proposal_seed_namespace=config["rng_namespaces"]["frozen_live_proposal"],
        deployment_artifact=ROOT / config["locked_inputs"]["phase6h_policy"],
    )
    model_load_seconds = time.perf_counter() - load_started
    if policy.checkpoint_sha256 != config["locked_inputs"]["phase6f_checkpoint_sha256"]:
        raise RuntimeError("Phase 6F checkpoint differs from preregistration")
    if policy.deployment_artifact_sha256 != config["locked_inputs"]["phase6h_policy_sha256"]:
        raise RuntimeError("Phase 6H policy differs from preregistration")

    alns_config = read_alns_config()
    if alns_config.candidate_trials != config["search"]["repair_decoder_trials_per_target"]:
        raise RuntimeError("repair trial count differs from frozen search")
    csgni_config = CSGNIConfig(
        intervention_rate=int(config["search"]["intervention_rate"]),
        proposal_seed_namespace=int(config["rng_namespaces"]["frozen_live_proposal"]),
        ni_repair_seed_namespace=int(config["rng_namespaces"]["frozen_live_repair"]),
        acceptance_seed_namespace=int(config["rng_namespaces"]["frozen_live_acceptance"]),
        diagnostics_seed_namespace=int(config["rng_namespaces"]["frozen_live_diagnostics"]),
    )
    fractions = [float(value) for value in config["pilot"]["sample_progress_fractions"]]

    for index, task in enumerate(pending, 1):
        instance = load_phase6i_instance(
            instance_root / task["instance_relative_path"]
        )
        budget = float(config["search"]["pilot_wall_clock_seconds_per_operation"]) * instance.num_operations
        observer = SnapshotObserver(budget, fractions)
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
        h1 = solve_dispatching(instance, "H1")
        forced_rows: list[dict[str, object]] = []
        full_bank_rows: list[dict[str, object]] = []
        replay_paths: list[str] = []
        for snapshot in observer.selected():
            current = decode_candidate(
                instance, _candidate_from_dict(snapshot["current_candidate"])
            )
            if not math.isclose(
                current.makespan,
                float(snapshot["current_makespan"]),
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise RuntimeError("snapshot replay makespan mismatch")
            progress = float(snapshot["search_progress"])
            destroy_count = min(
                max(2, round(instance.num_operations * alns_config.destroy_fraction)),
                instance.num_operations,
            )
            bank = score_frozen_candidate_bank(
                policy,
                instance,
                current,
                state_id=str(snapshot["state_id"]),
                destroy_count=destroy_count,
                search_progress=progress,
                search_stage=_search_stage(progress),
            )
            selected = select_forced_candidate_roles(bank.arms)
            if len(selected) != 4:
                raise RuntimeError(
                    f"{snapshot['state_id']} has only {len(selected)} unique forced roles"
                )
            context = _state_context(instance, current, h1.schedule, bank)
            state_rows = []
            for role in selected:
                decoded = decode_forced_candidate(
                    instance,
                    current,
                    role.arm,
                    state_id=str(snapshot["state_id"]),
                    repair_seed_namespace=int(config["rng_namespaces"]["forced_repair"]),
                    candidate_trials=int(config["search"]["repair_decoder_trials_per_target"]),
                )
                utility = (current.makespan - decoded.candidate.makespan) / current.makespan
                state_rows.append({
                    **task,
                    "split": "R09_PILOT",
                    "state_id": snapshot["state_id"],
                    "iteration": snapshot["iteration"],
                    "target_progress": snapshot["target_progress"],
                    "search_progress": progress,
                    "search_stage": snapshot["diagnostic_stage"],
                    "source_elapsed_wall_time": snapshot["elapsed_wall_time"],
                    "candidate_role": role.role,
                    "duplicate_replacement": role.replacement,
                    "target_set_id": role.arm.target_set_id,
                    "target_operation_ids": json.dumps(role.arm.destroyed_operations),
                    "origin_family": role.arm.arm_family,
                    "origin_destroy_operator": role.arm.origin_destroy_operator,
                    "origin_rules": json.dumps(role.arm.origin_rules),
                    "raw_score": role.arm.raw_score,
                    "raw_probability": role.arm.raw_probability,
                    "raw_utility": role.arm.raw_utility,
                    "calibrated_probability": role.arm.calibrated_probability,
                    "calibrated_utility": role.arm.calibrated_utility,
                    "decoded_candidate_makespan": decoded.candidate.makespan,
                    "decoded_immediate_utility": utility,
                    "positive_label": utility > 0,
                    "candidate_feasible": decoded.candidate.feasible,
                    "repair_seed": decoded.repair_seed,
                    "repair_trial_makespans": json.dumps(decoded.trial_makespans),
                    "forced_decoder_evaluations": decoded.decoder_evaluations,
                    "forced_decode_ms": decoded.runtime_ms,
                    "csg_build_ms": bank.timings_ms["csg_build"],
                    "proposal_bank_ms": bank.timings_ms["proposal_bank"],
                    "tensorization_ms": bank.timings_ms["tensorization_and_transfer"],
                    "model_inference_and_scoring_ms": bank.timings_ms[
                        "model_inference_and_action_scoring"
                    ],
                    "calibration_ms": bank.timings_ms["calibration"],
                    "bank_total_ms": bank.timings_ms["total"],
                    "requested_bank_size": bank.requested_arm_count,
                    "unique_bank_size": bank.unique_arm_count,
                    "duplicate_bank_size": bank.duplicate_arm_count,
                    "labels_post_trajectory": True,
                    **context,
                })
            _rank_and_label(state_rows)
            forced_rows.extend(state_rows)
            audit_progress = {"CF1": 0.1, "CF2": 0.45, "CF3": 0.9}[
                str(task["CF_level"])
            ]
            if math.isclose(
                float(snapshot["target_progress"]), audit_progress, abs_tol=1e-12
            ):
                top_eight = select_top_eight_audit_candidates(bank.arms, selected)
                top_eight_ids = {arm.target_set_id for arm in top_eight}
                broad_roles = {
                    choice.arm.target_set_id: choice.role for choice in selected
                }
                audit_state_rows = []
                for audit_order, arm in enumerate(
                    sorted(
                        bank.arms,
                        key=lambda item: (-item.raw_score, item.target_set_id),
                    ),
                    1,
                ):
                    decoded = decode_forced_candidate(
                        instance,
                        current,
                        arm,
                        state_id=str(snapshot["state_id"]),
                        repair_seed_namespace=int(
                            config["rng_namespaces"]["forced_repair"]
                        ),
                        candidate_trials=int(
                            config["search"]["repair_decoder_trials_per_target"]
                        ),
                    )
                    utility = (
                        current.makespan - decoded.candidate.makespan
                    ) / current.makespan
                    audit_state_rows.append({
                        **task,
                        "split": "R09_PILOT_V12_FULL_BANK_AUDIT",
                        "state_id": snapshot["state_id"],
                        "search_progress": progress,
                        "search_stage": snapshot["diagnostic_stage"],
                        "target_set_id": arm.target_set_id,
                        "target_operation_ids": json.dumps(arm.destroyed_operations),
                        "origin_family": arm.arm_family,
                        "origin_destroy_operator": arm.origin_destroy_operator,
                        "origin_rules": json.dumps(arm.origin_rules),
                        "raw_score": arm.raw_score,
                        "raw_probability": arm.raw_probability,
                        "raw_utility": arm.raw_utility,
                        "calibrated_probability": arm.calibrated_probability,
                        "calibrated_utility": arm.calibrated_utility,
                        "audit_score_order": audit_order,
                        "in_broad_four": arm.target_set_id in broad_roles,
                        "broad_role": broad_roles.get(arm.target_set_id),
                        "in_top_eight": arm.target_set_id in top_eight_ids,
                        "decoded_candidate_makespan": decoded.candidate.makespan,
                        "decoded_immediate_utility": utility,
                        "positive_label": utility > 0,
                        "candidate_feasible": decoded.candidate.feasible,
                        "repair_seed": decoded.repair_seed,
                        "forced_decoder_evaluations": decoded.decoder_evaluations,
                        "forced_decode_ms": decoded.runtime_ms,
                        "labels_post_trajectory": True,
                    })
                ordered_truth = sorted(
                    audit_state_rows,
                    key=lambda row: (
                        -float(row["decoded_immediate_utility"]),
                        str(row["target_set_id"]),
                    ),
                )
                best_utility = float(
                    ordered_truth[0]["decoded_immediate_utility"]
                )
                true_rank = {
                    str(row["target_set_id"]): rank
                    for rank, row in enumerate(ordered_truth, 1)
                }
                for row in audit_state_rows:
                    row["full_bank_true_rank"] = true_rank[
                        str(row["target_set_id"])
                    ]
                    row["regret_to_full_bank_best"] = (
                        best_utility - float(row["decoded_immediate_utility"])
                    )
                full_bank_rows.extend(audit_state_rows)
            replay = {
                "schema": "phase6i-mr-r09-pilot-state-replay-v1",
                **{key: snapshot[key] for key in (
                    "state_id",
                    "iteration",
                    "target_progress",
                    "search_progress",
                    "current_makespan",
                    "current_candidate",
                )},
                "instance_id": task["instance_id"],
                "instance_relative_path": task["instance_relative_path"],
                "trajectory_seed": task["seed"],
                "forced_selection_inputs": [
                    {
                        "target_set_id": arm.target_set_id,
                        "destroyed_operations": list(arm.destroyed_operations),
                        "origin_rules": list(arm.origin_rules),
                        "raw_score": arm.raw_score,
                        "raw_probability": arm.raw_probability,
                        "raw_utility": arm.raw_utility,
                        "calibrated_probability": arm.calibrated_probability,
                        "calibrated_utility": arm.calibrated_utility,
                    }
                    for arm in bank.arms
                ],
                "forced_results": state_rows,
            }
            replay_path = REPLAYS / task["instance_id"] / f"{snapshot['state_id']}.json"
            atomic_json(replay, replay_path)
            replay_paths.append(str(replay_path.relative_to(ROOT)))
        table = forced_path(task)
        atomic_parquet(pd.DataFrame(forced_rows), table)
        audit_table = full_bank_path(task)
        atomic_parquet(pd.DataFrame(full_bank_rows), audit_table)
        payload = {
            "schema": "phase6i-mr-r09-pilot-run-v1",
            "status": "COMPLETE",
            **task,
            "time_limit_seconds": budget,
            "best_makespan": result.best.makespan,
            "time_to_best": result.best_found_time,
            "runtime": result.runtime,
            "decoder_evaluations": result.decoder_evaluations,
            "iterations": result.iterations,
            "feasible": True,
            "model_load_seconds": model_load_seconds,
            "checkpoint_sha256": policy.checkpoint_sha256,
            "phase6h_policy_sha256": policy.deployment_artifact_sha256,
            "source_trajectory_completed_before_forced_labels": True,
            "sampled_states": len({row["state_id"] for row in forced_rows}),
            "forced_actions": len(forced_rows),
            "forced_actions_relative_path": str(table.relative_to(ROOT)),
            "forced_actions_sha256": digest(table),
            "full_bank_audit_actions": len(full_bank_rows),
            "full_bank_audit_relative_path": str(audit_table.relative_to(ROOT)),
            "full_bank_audit_sha256": digest(audit_table),
            "state_replays": replay_paths,
            "convergence_trace": trace,
            "diagnostics": result.diagnostics,
            "best_solution": result.best.schedule.to_dict(),
            "best_actions": [asdict(action) for action in result.best.actions],
        }
        atomic_json(payload, result_path(task))
        write_progress(tasks, process_started)
        print(
            f"[{index}/{len(pending)}] {task['instance_id']} "
            f"states=6 actions=24 iterations={result.iterations} "
            f"runtime={result.runtime:.2f}s",
            flush=True,
        )

    summarize(tasks)
    write_progress(tasks, process_started)
    print("PHASE6I_MR_R09_PILOT_BATCH_RETURNED", flush=True)


if __name__ == "__main__":
    main()
