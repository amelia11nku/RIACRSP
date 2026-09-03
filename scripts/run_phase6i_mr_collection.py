#!/usr/bin/env python3
"""Collect resumable Phase 6I-MR live-state labels on R09 or authorized R10."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import time

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

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
from rcias_clgri.search.csgni import CSGNIConfig, solve_csgni  # noqa: E402
from rcias_clgri.search.common import decode_candidate  # noqa: E402
from scripts.run_phase6i_mr_pilot import (  # noqa: E402
    _candidate_dict,
    _candidate_from_dict,
    _rank_and_label,
    _search_stage,
    _state_context,
    atomic_csv,
    atomic_json,
    atomic_parquet,
    digest,
    load_json,
    read_alns_config,
)


CONFIG_PATH = ROOT / "configs/phase6i_mr_live_utility_revision.json"
FREEZE_PATH = ROOT / "outputs/phase6i_mr/frozen/collection_protocol.json"
OUT = ROOT / "outputs/phase6i_mr/collection/r09"
RUNS = OUT / "runs"
FORCED = OUT / "forced_actions"
TOP_EIGHT = OUT / "top_eight_audit"
FULL_BANK = OUT / "full_bank_audit"
REPLAYS = OUT / "state_replays"


def configure_output_root(path: Path) -> None:
    global OUT, RUNS, FORCED, TOP_EIGHT, FULL_BANK, REPLAYS
    OUT = path
    RUNS = OUT / "runs"
    FORCED = OUT / "forced_actions"
    TOP_EIGHT = OUT / "top_eight_audit"
    FULL_BANK = OUT / "full_bank_audit"
    REPLAYS = OUT / "state_replays"


def recorded_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def diagnostic_stage(target: float) -> str:
    if target < 1.0 / 3.0:
        return "EARLY"
    if target < 2.0 / 3.0:
        return "MIDDLE"
    return "LATE"


class CollectionObserver:
    """Retain source states and aggregate source-policy component timings."""

    def __init__(self, target_fractions: list[float]) -> None:
        self.target_fractions = target_fractions
        self.events: list[dict[str, object]] = []
        self.component_ms: Counter[str] = Counter()

    def __call__(self, event: dict[str, object]) -> None:
        timing = dict(event.get("ni_timing_ms") or {})
        for key, value in timing.items():
            self.component_ms[str(key)] += float(value)
        self.component_ms["search_repair_decoder"] += (
            float(event.get("repair_runtime", 0.0)) * 1000.0
        )
        if not event.get("ni_eligible") or event.get("ni_state_id") is None:
            return
        features = dict(event.get("ni_state_feature_summary") or {})
        current = event["current_before"]
        self.events.append({
            "state_id": str(event["ni_state_id"]),
            "iteration": int(event["iteration"]),
            "search_progress": float(features.get("search_progress", 0.0)),
            "elapsed_wall_time": float(event["elapsed_time"]),
            "decoder_evaluations": int(event["decoder_evaluations"]),
            "current_makespan": float(current.makespan),
            "current_candidate": _candidate_dict(current.candidate),
        })

    def selected(self) -> list[dict[str, object]]:
        if len(self.events) < len(self.target_fractions):
            raise RuntimeError(
                f"source trajectory produced {len(self.events)} eligible states; "
                f"need {len(self.target_fractions)}"
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
                "search_stage": diagnostic_stage(float(target)),
            })
        return selected


def verify_freeze(config: dict, split: str) -> dict:
    freeze = load_json(FREEZE_PATH)
    if not all([
        freeze.get("schema") == "phase6i-mr-collection-protocol-freeze-v1.2",
        freeze.get("status") == "FROZEN_BEFORE_FORMAL_R09_COLLECTION",
        freeze.get("input_hashes", {}).get("phase6i_config_sha256")
        == digest(CONFIG_PATH),
        freeze.get("input_hashes", {}).get("instance_manifest_sha256")
        == digest(ROOT / config["instance_suite"]["manifest"]),
        freeze.get("diagnostic_branch", {}).get("branch")
        == "TARGET_MISMATCH",
        freeze.get("diagnostic_branch", {}).get("u2h_activated") is True,
        freeze.get("access_control", {}).get("r11_accessed") is False,
    ]):
        raise RuntimeError("collection protocol freeze is invalid")
    if split == "R09":
        if freeze["access_control"].get("r09_collection_authorized") is not True:
            raise RuntimeError("R09 collection is not authorized")
    else:
        authorization_path = (
            ROOT / "outputs/phase6i_mr/frozen/r10_collection_authorization.json"
        )
        if not authorization_path.is_file():
            raise RuntimeError(
                "R10 remains locked until all training-side artifacts and rules are frozen"
            )
        authorization = load_json(authorization_path)
        if not all([
            authorization.get("status") == "FROZEN_BEFORE_ONE_TIME_R10_ACCESS",
            authorization.get("r10_accessed") is False,
            authorization.get("r11_accessed") is False,
            authorization.get("collection_protocol_sha256") == digest(FREEZE_PATH),
        ]):
            raise RuntimeError("R10 collection authorization is invalid")
    return freeze


def build_tasks(config: dict, manifest: pd.DataFrame, split: str) -> list[dict]:
    split_label = {"R09": "LIVE_REV_FIT", "R10": "LIVE_REV_SELECT"}[split]
    seeds = [int(value) for value in config["seeds"][f"{split}_FULL_TRAJECTORY"]]
    selected = manifest[manifest.live_revision_split == split_label].copy()
    if len(selected) != 18 or set(selected.replicate) != {split}:
        raise RuntimeError(f"{split} collection requires exactly 18 instances")
    audit_order = selected.sort_values(
        ["scale", "CF_level", "cell_replicate", "instance_id"]
    )
    audit_targets = {
        row.instance_id: (0.15, 0.50, 0.85)[index % 3]
        for index, row in enumerate(audit_order.itertuples(index=False))
    }
    scale_order = {"S": 0, "M": 1, "L": 2}
    selected["_scale_order"] = selected.scale.map(scale_order)
    tasks = []
    audit_seed = min(seeds)
    for row in selected.sort_values(
        ["_scale_order", "CF_level", "cell_replicate", "instance_id"]
    ).itertuples(index=False):
        for seed in seeds:
            tasks.append({
                "split": split,
                "instance_id": row.instance_id,
                "instance_relative_path": row.relative_path,
                "instance_sha256": row.sha256,
                "scale": row.scale,
                "CF_level": row.CF_level,
                "RI_level": row.RI_level,
                "TI_level": row.TI_level,
                "cell_replicate": row.cell_replicate,
                "seed": seed,
                "top_eight_audit_target": (
                    float(audit_targets[row.instance_id])
                    if seed == audit_seed else None
                ),
                "full_bank_audit_target": (
                    float({"CF1": 0.15, "CF2": 0.50, "CF3": 0.85}[row.CF_level])
                    if seed == audit_seed and row.cell_replicate == "C02"
                    else None
                ),
            })
    if len(tasks) != 54:
        raise RuntimeError(f"{split} collection requires exactly 54 trajectories")
    return tasks


def stem(task: dict) -> str:
    return f"{task['instance_id']}__seed{task['seed']}"


def result_path(task: dict) -> Path:
    return RUNS / f"{stem(task)}.json"


def forced_path(task: dict) -> Path:
    return FORCED / f"{stem(task)}.parquet"


def top_eight_path(task: dict) -> Path:
    return TOP_EIGHT / f"{stem(task)}.parquet"


def full_bank_path(task: dict) -> Path:
    return FULL_BANK / f"{stem(task)}.parquet"


def valid_result(task: dict, *, config_hash: str, freeze_hash: str) -> dict | None:
    paths = (
        result_path(task),
        forced_path(task),
        top_eight_path(task),
        full_bank_path(task),
    )
    if not all(path.is_file() for path in paths):
        return None
    try:
        result = load_json(result_path(task))
    except (OSError, json.JSONDecodeError):
        return None
    expected_top_eight = 8 if task["top_eight_audit_target"] is not None else 0
    expected_full_minimum = 8 if task["full_bank_audit_target"] is not None else 0
    checks = [
        result.get("status") == "COMPLETE",
        result.get("split") == task["split"],
        result.get("instance_id") == task["instance_id"],
        result.get("seed") == task["seed"],
        result.get("config_sha256") == config_hash,
        result.get("collection_protocol_sha256") == freeze_hash,
        result.get("sampled_states") == 30,
        result.get("forced_actions") == 120,
        result.get("top_eight_audit_actions") == expected_top_eight,
        result.get("full_bank_audit_actions", 0) >= expected_full_minimum,
        result.get("forced_actions_sha256") == digest(forced_path(task)),
        result.get("top_eight_audit_sha256") == digest(top_eight_path(task)),
        result.get("full_bank_audit_sha256") == digest(full_bank_path(task)),
        result.get("r11_accessed") is False,
    ]
    if task["full_bank_audit_target"] is None:
        checks.append(result.get("full_bank_audit_actions") == 0)
    if task["split"] == "R09":
        checks.append(result.get("r10_accessed") is False)
    return result if all(checks) else None


def nearest_snapshot(
    snapshots: list[dict[str, object]], target: float | None
) -> dict[str, object] | None:
    if target is None:
        return None
    return min(
        snapshots,
        key=lambda row: (
            round(abs(float(row["target_progress"]) - target), 12),
            float(row["target_progress"]),
            str(row["state_id"]),
        ),
    )


def audit_row(task: dict, snapshot: dict, arm, decoded) -> dict[str, object]:
    utility = (
        float(snapshot["current_makespan"]) - decoded.candidate.makespan
    ) / float(snapshot["current_makespan"])
    return {
        **{key: task[key] for key in (
            "split", "instance_id", "scale", "CF_level", "RI_level",
            "TI_level", "cell_replicate", "seed",
        )},
        "state_id": snapshot["state_id"],
        "target_progress": snapshot["target_progress"],
        "search_progress": snapshot["search_progress"],
        "search_stage": snapshot["search_stage"],
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
        "decoded_candidate_makespan": decoded.candidate.makespan,
        "decoded_immediate_utility": utility,
        "positive_label": utility > 0,
        "candidate_feasible": decoded.candidate.feasible,
        "repair_seed": decoded.repair_seed,
        "forced_decoder_evaluations": decoded.decoder_evaluations,
        "forced_decode_ms": decoded.runtime_ms,
        "labels_post_trajectory": True,
    }


def rank_audit_rows(rows: list[dict[str, object]], prefix: str) -> None:
    truth = sorted(
        rows,
        key=lambda row: (
            -float(row["decoded_immediate_utility"]),
            str(row["target_set_id"]),
        ),
    )
    best = float(truth[0]["decoded_immediate_utility"])
    ranks = {str(row["target_set_id"]): rank for rank, row in enumerate(truth, 1)}
    for row in rows:
        row[f"{prefix}_true_rank"] = ranks[str(row["target_set_id"])]
        row[f"regret_to_{prefix}_best"] = (
            best - float(row["decoded_immediate_utility"])
        )


def write_progress(
    tasks: list[dict],
    *,
    started: float,
    config_hash: str,
    freeze_hash: str,
) -> None:
    complete = sum(
        valid_result(task, config_hash=config_hash, freeze_hash=freeze_hash)
        is not None
        for task in tasks
    )
    elapsed = time.perf_counter() - started
    rate = complete / elapsed if elapsed > 0 else 0.0
    remaining = (len(tasks) - complete) / rate if rate > 0 else None
    atomic_json({
        "schema": "phase6i-mr-collection-progress-v1.2",
        "split": tasks[0]["split"],
        "status": "COMPLETE" if complete == len(tasks) else "RUNNING",
        "completed_trajectories": complete,
        "total_trajectories": len(tasks),
        "current_process_elapsed_seconds": elapsed,
        "current_process_naive_remaining_seconds": remaining,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "r10_accessed": tasks[0]["split"] == "R10" and complete > 0,
        "r11_accessed": False,
    }, OUT / "progress.json")


def summarize(
    tasks: list[dict],
    *,
    config_hash: str,
    freeze_hash: str,
) -> None:
    run_rows = []
    forced_frames = []
    top_frames = []
    full_frames = []
    for task in tasks:
        result = valid_result(task, config_hash=config_hash, freeze_hash=freeze_hash)
        if result is None:
            continue
        run_rows.append({
            **{key: task[key] for key in (
                "split", "instance_id", "scale", "CF_level", "cell_replicate", "seed"
            )},
            "runtime": result["runtime"],
            "time_limit_seconds": result["time_limit_seconds"],
            "best_makespan": result["best_makespan"],
            "time_to_best": result["time_to_best"],
            "decoder_evaluations": result["decoder_evaluations"],
            "iterations": result["iterations"],
            "source_eligible_states": result["source_eligible_states"],
            "forced_decode_seconds": result["forced_decode_seconds"],
            "feasible": result["feasible"],
        })
        forced_frames.append(pd.read_parquet(forced_path(task)))
        top = pd.read_parquet(top_eight_path(task))
        full = pd.read_parquet(full_bank_path(task))
        if len(top):
            top_frames.append(top)
        if len(full):
            full_frames.append(full)
    runs = pd.DataFrame(run_rows)
    forced = pd.concat(forced_frames, ignore_index=True) if forced_frames else pd.DataFrame()
    top_eight = pd.concat(top_frames, ignore_index=True) if top_frames else pd.DataFrame()
    full_bank = pd.concat(full_frames, ignore_index=True) if full_frames else pd.DataFrame()
    atomic_csv(runs, OUT / "collection_run_summary.csv")
    if len(forced):
        atomic_parquet(forced, OUT / "forced_action_labels.parquet")
        atomic_csv(forced, OUT / "forced_action_labels.csv")
    if len(top_eight):
        atomic_parquet(top_eight, OUT / "top_eight_audit_table.parquet")
        atomic_csv(top_eight, OUT / "top_eight_audit_table.csv")
    if len(full_bank):
        atomic_parquet(full_bank, OUT / "full_bank_audit_table.parquet")
        atomic_csv(full_bank, OUT / "full_bank_audit_table.csv")

    complete = len(runs) == 54
    checks = {
        "exactly_54_trajectories": len(runs) == 54,
        "exactly_18_instances": runs.instance_id.nunique() == 18 if len(runs) else False,
        "exactly_1620_states": forced.state_id.nunique() == 1620 if len(forced) else False,
        "exactly_6480_broad_actions": len(forced) == 6480,
        "four_roles_per_state": bool(
            forced.groupby("state_id").candidate_role.nunique().eq(4).all()
        ) if len(forced) else False,
        "unique_broad_targets_per_state": bool(
            forced.groupby("state_id").target_set_id.nunique().eq(4).all()
        ) if len(forced) else False,
        "exactly_18_top_eight_states": (
            top_eight.state_id.nunique() == 18 if len(top_eight) else False
        ),
        "exactly_eight_actions_per_top_eight_state": bool(
            top_eight.groupby("state_id").size().eq(8).all()
        ) if len(top_eight) else False,
        "exactly_9_full_bank_states": (
            full_bank.state_id.nunique() == 9 if len(full_bank) else False
        ),
        "all_feasible": bool(
            forced.candidate_feasible.all()
            and top_eight.candidate_feasible.all()
            and full_bank.candidate_feasible.all()
            and runs.feasible.all()
        ) if len(forced) and len(top_eight) and len(full_bank) else False,
        "all_labels_post_trajectory": bool(
            forced.labels_post_trajectory.all()
            and top_eight.labels_post_trajectory.all()
            and full_bank.labels_post_trajectory.all()
        ) if len(forced) and len(top_eight) and len(full_bank) else False,
    }
    split = tasks[0]["split"]
    integrity = {
        "schema": "phase6i-mr-collection-integrity-v1.2",
        "split": split,
        "status": "PASS" if complete and all(checks.values()) else "INCOMPLETE",
        "config_sha256": config_hash,
        "collection_protocol_sha256": freeze_hash,
        "checks": checks,
        "completed_trajectories": len(runs),
        "completed_states": int(forced.state_id.nunique()) if len(forced) else 0,
        "completed_broad_actions": len(forced),
        "completed_top_eight_actions": len(top_eight),
        "completed_full_bank_actions": len(full_bank),
        "r10_accessed": split == "R10" and len(runs) > 0,
        "r11_accessed": False,
    }
    atomic_json(integrity, OUT / "collection_integrity.json")
    atomic_json({
        "schema": "phase6i-mr-collection-access-ledger-v1.2",
        "split": split,
        "collection_protocol_sha256": freeze_hash,
        "completed_trajectories": len(runs),
        "r09_accessed": split == "R09" and len(runs) > 0,
        "r10_accessed": split == "R10" and len(runs) > 0,
        "r11_accessed": False,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }, OUT / "access_ledger.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["R09", "R10"], required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit-runs", type=int)
    parser.add_argument("--only-instance")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()

    config = load_json(CONFIG_PATH)
    freeze = verify_freeze(config, args.split)
    if args.output_root is not None:
        configure_output_root(args.output_root.resolve())
    else:
        configure_output_root(
            ROOT / f"outputs/phase6i_mr/collection/{args.split.lower()}"
        )
    if args.smoke and args.output_root is None:
        raise RuntimeError("smoke mode requires a non-formal --output-root")
    if not args.smoke and args.output_root is not None:
        raise RuntimeError("non-smoke collection must use the frozen formal output root")

    manifest = pd.read_csv(ROOT / config["instance_suite"]["manifest"])
    tasks = build_tasks(config, manifest, args.split)
    if args.only_instance is not None:
        tasks = [task for task in tasks if task["instance_id"] == args.only_instance]
        if not tasks:
            raise RuntimeError(f"unknown split instance: {args.only_instance}")
    config_hash = digest(CONFIG_PATH)
    freeze_hash = digest(FREEZE_PATH)
    process_started = time.perf_counter()
    if args.summarize_only:
        summarize(tasks, config_hash=config_hash, freeze_hash=freeze_hash)
        write_progress(
            tasks,
            started=process_started,
            config_hash=config_hash,
            freeze_hash=freeze_hash,
        )
        return
    pending = [
        task for task in tasks
        if valid_result(task, config_hash=config_hash, freeze_hash=freeze_hash) is None
    ]
    if args.limit_runs is not None:
        pending = pending[:args.limit_runs]
    print(
        f"PHASE6I_MR_{args.split}_COLLECTION_START "
        f"pending={len(pending)} total={len(tasks)} smoke={args.smoke}",
        flush=True,
    )
    write_progress(
        tasks,
        started=process_started,
        config_hash=config_hash,
        freeze_hash=freeze_hash,
    )
    policy_load_started = time.perf_counter()
    policy = FrozenLiveInference(
        ROOT / config["locked_inputs"]["phase6f_experiment_freeze"],
        device=args.device,
        proposal_seed_namespace=config["rng_namespaces"]["frozen_live_proposal"],
        deployment_artifact=ROOT / config["locked_inputs"]["phase6h_policy"],
    )
    model_load_seconds = time.perf_counter() - policy_load_started
    if policy.checkpoint_sha256 != config["locked_inputs"]["phase6f_checkpoint_sha256"]:
        raise RuntimeError("Phase 6F checkpoint differs from collection freeze")
    if policy.deployment_artifact_sha256 != config["locked_inputs"]["phase6h_policy_sha256"]:
        raise RuntimeError("Phase 6H policy differs from collection freeze")
    alns_config = read_alns_config()
    if alns_config.candidate_trials != freeze["candidate_contract"][
        "repair_decoder_trials_per_target"
    ]:
        raise RuntimeError("frozen ALNS repair trial count changed")
    csgni_config = CSGNIConfig(
        intervention_rate=int(config["search"]["intervention_rate"]),
        proposal_seed_namespace=int(config["rng_namespaces"]["frozen_live_proposal"]),
        ni_repair_seed_namespace=int(config["rng_namespaces"]["frozen_live_repair"]),
        acceptance_seed_namespace=int(config["rng_namespaces"]["frozen_live_acceptance"]),
        diagnostics_seed_namespace=int(config["rng_namespaces"]["frozen_live_diagnostics"]),
    )
    fractions = [float(value) for value in freeze["collection"][
        "normalized_progress_centers"
    ]]
    instance_root = ROOT / config["instance_suite"]["root"]
    budget_factor = (
        0.05 if args.smoke
        else float(freeze["collection"]["wall_clock_seconds_per_operation"])
    )

    for task_index, task in enumerate(pending, 1):
        instance_path = instance_root / task["instance_relative_path"]
        if digest(instance_path) != task["instance_sha256"]:
            raise RuntimeError(f"instance hash mismatch: {task['instance_id']}")
        instance = load_phase6i_instance(instance_path)
        budget = budget_factor * instance.num_operations
        observer = CollectionObserver(fractions)
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
            raise RuntimeError({
                "task": task,
                "violations": feasibility["violations"],
            })
        trace = validate_incumbent_trace(
            result.convergence_trace,
            final_best=result.best.makespan,
        )
        snapshots = observer.selected()
        top_snapshot = nearest_snapshot(
            snapshots, task["top_eight_audit_target"]
        )
        full_snapshot = nearest_snapshot(
            snapshots, task["full_bank_audit_target"]
        )
        top_state_id = str(top_snapshot["state_id"]) if top_snapshot else None
        full_state_id = str(full_snapshot["state_id"]) if full_snapshot else None
        h1 = solve_dispatching(instance, "H1")
        forced_rows: list[dict[str, object]] = []
        top_rows: list[dict[str, object]] = []
        full_rows: list[dict[str, object]] = []
        replay_paths: list[str] = []
        forced_decode_started = time.perf_counter()
        for snapshot in snapshots:
            current = decode_candidate(
                instance, _candidate_from_dict(snapshot["current_candidate"])
            )
            if not math.isclose(
                current.makespan,
                float(snapshot["current_makespan"]),
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise RuntimeError("collection snapshot replay makespan mismatch")
            destroy_count = min(
                max(2, round(instance.num_operations * alns_config.destroy_fraction)),
                instance.num_operations,
            )
            progress = float(snapshot["search_progress"])
            bank = score_frozen_candidate_bank(
                policy,
                instance,
                current,
                state_id=str(snapshot["state_id"]),
                destroy_count=destroy_count,
                search_progress=progress,
                search_stage=_search_stage(progress),
            )
            selected_roles = select_forced_candidate_roles(bank.arms)
            if len(selected_roles) != 4:
                raise RuntimeError("fewer than four unique broad roles")
            context = _state_context(instance, current, h1.schedule, bank)
            decoded_cache = {}
            state_rows = []
            for role in selected_roles:
                decoded = decode_forced_candidate(
                    instance,
                    current,
                    role.arm,
                    state_id=str(snapshot["state_id"]),
                    repair_seed_namespace=int(config["rng_namespaces"]["forced_repair"]),
                    candidate_trials=int(config["search"]["repair_decoder_trials_per_target"]),
                )
                decoded_cache[role.arm.target_set_id] = decoded
                utility = (current.makespan - decoded.candidate.makespan) / current.makespan
                state_rows.append({
                    **{key: task[key] for key in (
                        "split", "instance_id", "instance_relative_path", "scale",
                        "CF_level", "RI_level", "TI_level", "cell_replicate", "seed",
                    )},
                    "state_id": snapshot["state_id"],
                    "iteration": snapshot["iteration"],
                    "target_progress": snapshot["target_progress"],
                    "search_progress": progress,
                    "search_stage": snapshot["search_stage"],
                    "source_elapsed_wall_time": snapshot["elapsed_wall_time"],
                    "source_decoder_evaluations": snapshot["decoder_evaluations"],
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
            broad_roles = {
                choice.arm.target_set_id: choice.role for choice in selected_roles
            }

            def decode_arm(arm):
                if arm.target_set_id not in decoded_cache:
                    decoded_cache[arm.target_set_id] = decode_forced_candidate(
                        instance,
                        current,
                        arm,
                        state_id=str(snapshot["state_id"]),
                        repair_seed_namespace=int(config["rng_namespaces"]["forced_repair"]),
                        candidate_trials=int(config["search"]["repair_decoder_trials_per_target"]),
                    )
                return decoded_cache[arm.target_set_id]

            top_eight = select_top_eight_audit_candidates(bank.arms, selected_roles)
            top_eight_ids = {arm.target_set_id for arm in top_eight}
            if str(snapshot["state_id"]) == top_state_id:
                selected_top_rows = []
                for order, arm in enumerate(
                    sorted(top_eight, key=lambda item: (-item.raw_score, item.target_set_id)),
                    1,
                ):
                    row = audit_row(task, snapshot, arm, decode_arm(arm))
                    row.update({
                        "audit_score_order": order,
                        "in_broad_four": arm.target_set_id in broad_roles,
                        "broad_role": broad_roles.get(arm.target_set_id),
                        "audit_target_progress": task["top_eight_audit_target"],
                    })
                    selected_top_rows.append(row)
                rank_audit_rows(selected_top_rows, "top_eight")
                top_rows.extend(selected_top_rows)
            if str(snapshot["state_id"]) == full_state_id:
                selected_full_rows = []
                for order, arm in enumerate(
                    sorted(bank.arms, key=lambda item: (-item.raw_score, item.target_set_id)),
                    1,
                ):
                    row = audit_row(task, snapshot, arm, decode_arm(arm))
                    row.update({
                        "audit_score_order": order,
                        "in_broad_four": arm.target_set_id in broad_roles,
                        "broad_role": broad_roles.get(arm.target_set_id),
                        "in_top_eight": arm.target_set_id in top_eight_ids,
                        "audit_target_progress": task["full_bank_audit_target"],
                    })
                    selected_full_rows.append(row)
                rank_audit_rows(selected_full_rows, "full_bank")
                full_rows.extend(selected_full_rows)

            replay = {
                "schema": "phase6i-mr-collection-state-replay-v1.2",
                **{key: snapshot[key] for key in (
                    "state_id", "iteration", "target_progress", "search_progress",
                    "search_stage", "current_makespan", "current_candidate",
                )},
                "split": task["split"],
                "instance_id": task["instance_id"],
                "instance_relative_path": task["instance_relative_path"],
                "trajectory_seed": task["seed"],
                "top_eight_audit_state": str(snapshot["state_id"]) == top_state_id,
                "full_bank_audit_state": str(snapshot["state_id"]) == full_state_id,
                "forced_selection_inputs": [
                    {
                        "target_set_id": arm.target_set_id,
                        "arm_family": arm.arm_family,
                        "origin_destroy_operator": arm.origin_destroy_operator,
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
            replay_path = REPLAYS / stem(task) / f"{snapshot['state_id']}.json"
            atomic_json(replay, replay_path)
            replay_paths.append(recorded_path(replay_path))

        forced_table = forced_path(task)
        top_table = top_eight_path(task)
        full_table = full_bank_path(task)
        atomic_parquet(pd.DataFrame(forced_rows), forced_table)
        atomic_parquet(pd.DataFrame(top_rows), top_table)
        atomic_parquet(pd.DataFrame(full_rows), full_table)
        payload = {
            "schema": "phase6i-mr-collection-run-v1.2",
            "status": "COMPLETE",
            **task,
            "config_sha256": config_hash,
            "collection_protocol_sha256": freeze_hash,
            "smoke": args.smoke,
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
            "source_eligible_states": len(observer.events),
            "source_component_timing_ms": dict(observer.component_ms),
            "sampled_states": len({row["state_id"] for row in forced_rows}),
            "forced_actions": len(forced_rows),
            "forced_decode_seconds": time.perf_counter() - forced_decode_started,
            "forced_actions_relative_path": recorded_path(forced_table),
            "forced_actions_sha256": digest(forced_table),
            "top_eight_audit_actions": len(top_rows),
            "top_eight_audit_relative_path": recorded_path(top_table),
            "top_eight_audit_sha256": digest(top_table),
            "full_bank_audit_actions": len(full_rows),
            "full_bank_audit_relative_path": recorded_path(full_table),
            "full_bank_audit_sha256": digest(full_table),
            "state_replays": replay_paths,
            "convergence_trace": trace,
            "diagnostics": result.diagnostics,
            "best_solution": result.best.schedule.to_dict(),
            "best_actions": [asdict(action) for action in result.best.actions],
            "r10_accessed": task["split"] == "R10",
            "r11_accessed": False,
        }
        atomic_json(payload, result_path(task))
        write_progress(
            tasks,
            started=process_started,
            config_hash=config_hash,
            freeze_hash=freeze_hash,
        )
        print(
            f"PHASE6I_MR_{args.split}_COLLECTION_RUN "
            f"{task_index}/{len(pending)} {stem(task)} states=30 "
            f"actions=120 runtime={result.runtime:.2f}s",
            flush=True,
        )

    summarize(tasks, config_hash=config_hash, freeze_hash=freeze_hash)
    write_progress(
        tasks,
        started=process_started,
        config_hash=config_hash,
        freeze_hash=freeze_hash,
    )
    print(f"PHASE6I_MR_{args.split}_COLLECTION_BATCH_RETURNED", flush=True)


if __name__ == "__main__":
    main()
