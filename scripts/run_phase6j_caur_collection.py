#!/usr/bin/env python3
"""Collect the resumable full-bank R12 CAUR training labels."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
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
from rcias_clgri.analysis.phase6j_caur import (  # noqa: E402
    FULL_BANK_SCOPE,
    build_candidate_source_features,
    continue_frozen_alns_at_horizons,
    critical_and_bottleneck_operations,
    fallback_relative_advantage,
    validate_full_bank_feature_rows,
    validate_grouped_label_records,
)
from rcias_clgri.data.phase6j_access import (  # noqa: E402
    load_phase6j_instance,
    verify_r12_collection_authorization,
)
from rcias_clgri.env.feasibility import check_schedule  # noqa: E402
from rcias_clgri.search.common import decode_candidate  # noqa: E402
from rcias_clgri.search.csgni import CSGNIConfig, solve_csgni  # noqa: E402
from rcias_clgri.search.phase6c import generate_revised_target_arms  # noqa: E402
from scripts.run_phase6j_caur_pilot import (  # noqa: E402
    SnapshotObserver,
    atomic_csv,
    atomic_json,
    atomic_parquet,
    candidate_from_dict,
    digest,
    git_head,
    load_json,
    load_policy,
    read_alns_config,
    search_stage,
    validate_preregistration,
)


CONFIG_PATH = ROOT / "configs/phase6j_caur.json"
FREEZE_PATH = ROOT / "outputs/phase6j_caur/frozen/r12_horizon_freeze.json"
OUT = ROOT / "outputs/phase6j_caur/r12_collection"
SOURCE_RUNS = OUT / "source_runs"
RAW_LABELS = OUT / "raw_seed_labels"
GROUPED_LABELS = OUT / "grouped_labels"
STATE_REPLAYS = OUT / "state_replays"
STATE_STATUS = OUT / "state_status"


def build_collection_tasks(config: dict) -> list[dict]:
    manifest = pd.read_csv(ROOT / config["instance_suite"]["manifest"])
    fit = manifest[manifest.caur_split.eq("CAUR_FIT")].copy()
    if (
        len(fit) != int(config["r12_collection"]["instances"])
        or set(fit.replicate) != {"R12"}
        or set(fit.cell_replicate) != {"C01", "C02"}
    ):
        raise RuntimeError("R12 collection requires all 18 frozen fit instances")
    fit["scale_order"] = fit.scale.map({"S": 0, "M": 1, "L": 2})
    seeds = [int(value) for value in config["rng"]["r12_trajectory_seeds"]]
    tasks = []
    for row in fit.sort_values(
        ["scale_order", "CF_level", "cell_replicate"]
    ).itertuples(index=False):
        for seed in seeds:
            tasks.append({
                "instance_id": row.instance_id,
                "instance_relative_path": row.relative_path,
                "instance_sha256": row.sha256,
                "scale": row.scale,
                "CF_level": row.CF_level,
                "RI_level": row.RI_level,
                "TI_level": row.TI_level,
                "cell_replicate": row.cell_replicate,
                "trajectory_seed": seed,
            })
    expected = (
        int(config["r12_collection"]["instances"])
        * int(config["r12_collection"]["source_trajectories_per_instance"])
    )
    if len(tasks) != expected:
        raise RuntimeError(f"R12 collection requires exactly {expected} source trajectories")
    return tasks


def source_path(task: dict) -> Path:
    return SOURCE_RUNS / f"{task['instance_id']}__seed{task['trajectory_seed']}.json"


def valid_source(task: dict, authorization_sha256: str) -> dict | None:
    path = source_path(task)
    if not path.is_file():
        return None
    try:
        record = load_json(path)
    except (OSError, json.JSONDecodeError):
        return None
    if (
        record.get("schema") != "phase6j-caur-r12-collection-source-run-v1"
        or record.get("status") != "COMPLETE"
        or record.get("instance_id") != task["instance_id"]
        or record.get("trajectory_seed") != task["trajectory_seed"]
        or record.get("instance_sha256") != task["instance_sha256"]
        or record.get("r12_horizon_freeze_sha256") != authorization_sha256
        or record.get("source_completed_before_labels") is not True
        or record.get("feasible") is not True
        or len(record.get("snapshots", [])) != 8
    ):
        return None
    return record


def run_source_trajectory(
    task: dict,
    config: dict,
    policy,
    alns_config,
    authorization_sha256: str,
) -> dict:
    instance_path = ROOT / config["instance_suite"]["root"] / task["instance_relative_path"]
    instance = load_phase6j_instance(instance_path)
    if digest(instance_path) != task["instance_sha256"]:
        raise RuntimeError("R12 collection instance hash mismatch")
    if config["r12_pilot"]["source_budget_seconds_per_instance"] != "0.25N":
        raise RuntimeError("R12 source budget contract changed")
    budget = 0.25 * instance.num_operations
    anchors = [float(value) for value in config["r12_collection"]["progress_anchors"]]
    observer = SnapshotObserver(anchors)
    csgni = CSGNIConfig(
        intervention_rate=100,
        proposal_seed_namespace=int(config["rng"]["proposal_namespace"]),
        ni_repair_seed_namespace=int(config["rng"]["repair_namespace"]),
        acceptance_seed_namespace=int(config["rng"]["source_acceptance_namespace"]),
        diagnostics_seed_namespace=int(config["rng"]["source_diagnostics_namespace"]),
    )
    result = solve_csgni(
        instance,
        budget,
        int(task["trajectory_seed"]),
        policy,
        alns_config=alns_config,
        csgni_config=csgni,
        observer=observer,
    )
    feasibility = check_schedule(instance, result.best.schedule)
    if not feasibility["feasible"]:
        raise RuntimeError({
            "instance_id": task["instance_id"],
            "violations": feasibility["violations"],
        })
    record = {
        "schema": "phase6j-caur-r12-collection-source-run-v1",
        "status": "COMPLETE",
        "implementation_commit": git_head(),
        "r12_horizon_freeze_sha256": authorization_sha256,
        **task,
        "budget_seconds": budget,
        "runtime_seconds": result.runtime,
        "iterations": result.iterations,
        "decoder_evaluations": result.decoder_evaluations,
        "best_makespan": result.best.makespan,
        "best_found_seconds": result.best_found_time,
        "feasible": True,
        "phase6f_checkpoint_sha256": policy.checkpoint_sha256,
        "phase6h_policy_sha256": policy.deployment_artifact_sha256,
        "source_completed_before_labels": True,
        "snapshots": observer.selected(),
        "convergence_trace": validate_incumbent_trace(
            result.convergence_trace, final_best=result.best.makespan
        ),
        "diagnostics": result.diagnostics,
    }
    atomic_json(record, source_path(task))
    return record


def state_paths(state_id: str) -> tuple[Path, Path, Path, Path]:
    return (
        RAW_LABELS / f"{state_id}.parquet",
        GROUPED_LABELS / f"{state_id}.parquet",
        STATE_REPLAYS / f"{state_id}.json",
        STATE_STATUS / f"{state_id}.json",
    )


def valid_state_status(path: Path, authorization_sha256: str) -> dict | None:
    if not path.is_file():
        return None
    try:
        record = load_json(path)
        raw, grouped, replay, _ = state_paths(str(record["state_id"]))
    except (OSError, KeyError, json.JSONDecodeError):
        return None
    if (
        record.get("schema") != "phase6j-caur-r12-collection-state-status-v1"
        or record.get("status") != "COMPLETE"
        or record.get("r12_horizon_freeze_sha256") != authorization_sha256
        or not all(item.is_file() for item in (raw, grouped, replay))
        or record.get("raw_labels_sha256") != digest(raw)
        or record.get("grouped_labels_sha256") != digest(grouped)
        or record.get("replay_sha256") != digest(replay)
    ):
        return None
    return record


def collect_state(
    task: dict,
    snapshot: dict,
    config: dict,
    policy,
    alns_config,
    authorization: dict,
    authorization_sha256: str,
) -> dict:
    state_id = str(snapshot["state_id"])
    raw_path, grouped_path, replay_path, status_path = state_paths(state_id)
    existing = valid_state_status(status_path, authorization_sha256)
    if existing is not None:
        return existing
    started = time.perf_counter()
    instance_path = ROOT / config["instance_suite"]["root"] / task["instance_relative_path"]
    instance = load_phase6j_instance(instance_path)
    current = decode_candidate(instance, candidate_from_dict(snapshot["current_candidate"]))
    if not math.isclose(current.makespan, float(snapshot["current_makespan"]), abs_tol=1e-9):
        raise RuntimeError(f"snapshot replay mismatch: {state_id}")
    progress = float(snapshot["search_progress"])
    destroy_count = min(
        max(2, round(instance.num_operations * alns_config.destroy_fraction)),
        instance.num_operations,
    )
    bank = score_frozen_candidate_bank(
        policy,
        instance,
        current,
        state_id=state_id,
        destroy_count=destroy_count,
        search_progress=progress,
        search_stage=search_stage(progress),
    )
    generated = generate_revised_target_arms(
        instance,
        current,
        state_id,
        destroy_count,
        int(config["rng"]["proposal_namespace"]),
    )
    if [arm.target_set_id for arm in bank.arms] != [arm.target_set_id for arm in generated.arms]:
        raise RuntimeError("scored and regenerated full banks differ")
    roles = select_forced_candidate_roles(bank.arms)
    fallback_selections = [row for row in roles if row.role == "ALNS_RELATED_FALLBACK"]
    if len(fallback_selections) != 1:
        raise RuntimeError("one frozen ALNS-related fallback is required")
    fallback_id = fallback_selections[0].arm.target_set_id
    top8 = {arm.target_set_id for arm in select_top_eight_audit_candidates(bank.arms, roles)}
    critical, bottleneck, bottleneck_proxy = critical_and_bottleneck_operations(instance, current)
    features = build_candidate_source_features(
        generated,
        state_id=state_id,
        operation_count=instance.num_operations,
        fallback_target_set_id=fallback_id,
        frozen_scores={arm.target_set_id: arm.raw_score for arm in bank.arms},
        critical_operations=critical,
        bottleneck_operations=bottleneck,
    )
    validate_full_bank_feature_rows(generated, features)
    feature_by_id = {str(row["target_set_id"]): row for row in features}
    horizon = int(authorization["selected_horizon"])
    continuation_seeds = [int(value) for value in config["rng"]["continuation_crn_seeds"]]
    decoded_by_id = {}
    continuations = {}
    forced_runtime_seconds = 0.0
    for arm in bank.arms:
        decoded = decode_forced_candidate(
            instance,
            current,
            arm,
            state_id=state_id,
            repair_seed_namespace=int(config["rng"]["repair_namespace"]),
            candidate_trials=int(config["candidate_bank"]["candidate_trials_per_target"]),
        )
        if (
            not decoded.candidate.feasible
            or not check_schedule(instance, decoded.candidate.schedule)["feasible"]
        ):
            raise RuntimeError(f"infeasible forced candidate: {state_id}/{arm.target_set_id}")
        decoded_by_id[arm.target_set_id] = decoded
        forced_runtime_seconds += decoded.runtime_ms / 1000.0
        continuations[arm.target_set_id] = {
            seed: continue_frozen_alns_at_horizons(
                instance,
                decoded.candidate,
                state_id=state_id,
                continuation_seed=seed,
                seed_namespace=int(config["rng"]["continuation_namespace"]),
                horizons=[horizon],
                config=alns_config,
            )[horizon]
            for seed in continuation_seeds
        }

    fallback_continuations = continuations[fallback_id]
    raw_rows: list[dict] = []
    grouped_rows: list[dict] = []
    arm_by_id = {arm.target_set_id: arm for arm in bank.arms}
    generated_by_id = {arm.target_set_id: arm for arm in generated.arms}
    for target_id, arm_continuations in continuations.items():
        arm = arm_by_id[target_id]
        decoded = decoded_by_id[target_id]
        immediate = (current.makespan - decoded.candidate.makespan) / current.makespan
        base = {
            **task,
            **feature_by_id[target_id],
            "split": "R12_CAUR_FIT",
            "target_progress": snapshot["target_progress"],
            "search_progress": progress,
            "search_stage": search_stage(progress),
            "source_iteration": snapshot["iteration"],
            "source_elapsed_wall_time": snapshot["source_elapsed_wall_time"],
            "fallback_target_set_id": fallback_id,
            "target_operation_ids": json.dumps(arm.destroyed_operations),
            "origin_rules": json.dumps(arm.origin_rules),
            "origin_families": json.dumps(generated_by_id[target_id].origin_families),
            "frozen_raw_score": arm.raw_score,
            "frozen_calibrated_probability": arm.calibrated_probability,
            "frozen_immediate_utility_prediction": arm.calibrated_utility,
            "candidate_decoded_makespan": decoded.candidate.makespan,
            "immediate_utility": immediate,
            "candidate_feasible": True,
            "repair_seed": decoded.repair_seed,
            "repair_trial_makespans": json.dumps(decoded.trial_makespans),
            "repair_decoder_evaluations": decoded.decoder_evaluations,
            "forced_decode_seconds": decoded.runtime_ms / 1000.0,
            "requested_bank_count": generated.requested_arm_count,
            "full_bank_unique_count": generated.unique_arm_count,
            "duplicate_bank_count": generated.duplicate_arm_count,
            "in_reduced_top8_audit": target_id in top8,
            "inclusion_probability": 1.0,
            "bottleneck_proxy": bottleneck_proxy,
            "critical_operation_ids": json.dumps(critical),
            "bottleneck_operation_ids": json.dumps(bottleneck),
            "labels_post_source_trajectory": True,
            "r12_horizon_freeze_sha256": authorization_sha256,
        }
        advantages = []
        candidate_results = []
        fallback_results = []
        for seed in continuation_seeds:
            candidate_prefix = arm_continuations[seed]
            fallback_prefix = fallback_continuations[seed]
            candidate_result = candidate_prefix.result
            fallback_result = fallback_prefix.result
            if candidate_result.derived_seed != fallback_result.derived_seed:
                raise RuntimeError("candidate/fallback CRN identity mismatch")
            advantage = fallback_relative_advantage(
                fallback_result.best_makespan,
                candidate_result.best_makespan,
                current.makespan,
            )
            advantages.append(advantage)
            candidate_results.append(candidate_result)
            fallback_results.append(fallback_result)
            raw_rows.append({
                **base,
                "continuation_seed": seed,
                "horizon": horizon,
                "candidate_continuation_best_makespan": candidate_result.best_makespan,
                "fallback_continuation_best_makespan": fallback_result.best_makespan,
                "continuation_advantage": advantage,
                "paired_derived_seed": candidate_result.derived_seed,
                "candidate_continuation_decoder_evaluations": candidate_result.decoder_evaluations,
                "candidate_continuation_seconds": candidate_result.runtime_ms / 1000.0,
                "candidate_continuation_decoder_seconds": candidate_prefix.decoder_seconds,
                "candidate_continuation_neighbor_seconds": candidate_prefix.neighbor_seconds,
                "fallback_continuation_decoder_evaluations": fallback_result.decoder_evaluations,
                "fallback_continuation_seconds": fallback_result.runtime_ms / 1000.0,
                "fallback_continuation_decoder_seconds": fallback_prefix.decoder_seconds,
                "fallback_continuation_neighbor_seconds": fallback_prefix.neighbor_seconds,
                "decoder_seconds": candidate_prefix.decoder_seconds,
            })
        grouped_rows.append({
            **base,
            "horizon": horizon,
            "continuation_seed_count": len(continuation_seeds),
            "continuation_advantage_mean": float(np.mean(advantages)),
            "continuation_advantage_std": float(np.std(advantages)),
            "beats_fallback": float(np.mean(np.asarray(advantages) > 0)),
            "continuation_best_makespan": float(np.mean([
                row.best_makespan for row in candidate_results
            ])),
            "fallback_continuation_best_makespan": float(np.mean([
                row.best_makespan for row in fallback_results
            ])),
            "decoder_seconds": float(sum(
                arm_continuations[seed].decoder_seconds for seed in continuation_seeds
            )),
        })

    validate_grouped_label_records(grouped_rows)
    raw_frame = pd.DataFrame(raw_rows).sort_values(
        ["target_set_id", "continuation_seed"]
    ).reset_index(drop=True)
    grouped_frame = pd.DataFrame(grouped_rows).sort_values("target_set_id").reset_index(drop=True)
    atomic_parquet(raw_frame, raw_path)
    atomic_parquet(grouped_frame, grouped_path)
    replay = {
        "schema": "phase6j-caur-r12-collection-state-replay-v1",
        "implementation_commit": git_head(),
        "r12_horizon_freeze_sha256": authorization_sha256,
        "state_id": state_id,
        "instance_id": task["instance_id"],
        "instance_relative_path": task["instance_relative_path"],
        "instance_sha256": task["instance_sha256"],
        "trajectory_seed": task["trajectory_seed"],
        "snapshot": snapshot,
        "current_replay_makespan": current.makespan,
        "requested_bank_count": generated.requested_arm_count,
        "unique_bank_count": generated.unique_arm_count,
        "duplicate_bank_count": generated.duplicate_arm_count,
        "fallback_target_set_id": fallback_id,
        "full_bank_target_ids": [arm.target_set_id for arm in bank.arms],
        "full_bank_target_operations": {
            arm.target_set_id: list(arm.destroyed_operations) for arm in bank.arms
        },
        "raw_labels_relative_path": str(raw_path.relative_to(ROOT)),
        "grouped_labels_relative_path": str(grouped_path.relative_to(ROOT)),
        "source_completed_before_labels": True,
    }
    atomic_json(replay, replay_path)
    status = {
        "schema": "phase6j-caur-r12-collection-state-status-v1",
        "status": "COMPLETE",
        "implementation_commit": git_head(),
        "r12_horizon_freeze_sha256": authorization_sha256,
        "state_id": state_id,
        "instance_id": task["instance_id"],
        "unique_candidates": generated.unique_arm_count,
        "raw_rows": len(raw_frame),
        "grouped_rows": len(grouped_frame),
        "horizon": horizon,
        "continuation_seeds": continuation_seeds,
        "repair_decoder_evaluations": int(
            grouped_frame.repair_decoder_evaluations.sum()
        ),
        "continuation_decoder_evaluations": int(
            raw_frame.candidate_continuation_decoder_evaluations.sum()
        ),
        "decoder_seconds": float(
            raw_frame.candidate_continuation_decoder_seconds.sum()
        ),
        "forced_decode_seconds": forced_runtime_seconds,
        "elapsed_seconds": time.perf_counter() - started,
        "raw_labels_sha256": digest(raw_path),
        "grouped_labels_sha256": digest(grouped_path),
        "replay_sha256": digest(replay_path),
    }
    atomic_json(status, status_path)
    return status


def all_known_states(
    tasks: list[dict], authorization_sha256: str
) -> list[tuple[dict, dict]]:
    known = []
    for task in tasks:
        source = valid_source(task, authorization_sha256)
        if source is not None:
            known.extend((task, snapshot) for snapshot in source["snapshots"])
    return known


def write_progress(
    tasks: list[dict], authorization_sha256: str, process_started: float
) -> dict:
    known = all_known_states(tasks, authorization_sha256)
    statuses = [
        valid_state_status(state_paths(str(snapshot["state_id"]))[3], authorization_sha256)
        for _, snapshot in known
    ]
    completed = [row for row in statuses if row is not None]
    expected_states = 288
    elapsed = time.perf_counter() - process_started
    measured = sum(float(row["elapsed_seconds"]) for row in completed)
    seconds_per_state = measured / len(completed) if completed else None
    source_complete = sum(
        valid_source(task, authorization_sha256) is not None for task in tasks
    )
    payload = {
        "schema": "phase6j-caur-r12-collection-progress-v1",
        "status": "COMPLETE" if len(completed) == expected_states else "RUNNING",
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "r12_horizon_freeze_sha256": authorization_sha256,
        "source_runs_complete": source_complete,
        "source_runs_expected": len(tasks),
        "states_complete": len(completed),
        "states_expected": expected_states,
        "unique_candidates_complete": sum(int(row["unique_candidates"]) for row in completed),
        "process_elapsed_seconds": elapsed,
        "measured_seconds_per_state": seconds_per_state,
        "measured_remaining_state_seconds": (
            seconds_per_state * (expected_states - len(completed))
            if seconds_per_state is not None else None
        ),
        "decoder_seconds_complete": sum(float(row["decoder_seconds"]) for row in completed),
    }
    atomic_json(payload, OUT / "progress.json")
    return payload


def _collection_files() -> list[Path]:
    files = []
    for directory in (SOURCE_RUNS, RAW_LABELS, GROUPED_LABELS, STATE_REPLAYS, STATE_STATUS):
        files.extend(sorted(path for path in directory.glob("*") if path.is_file()))
    return files


def summarize(tasks: list[dict], authorization: dict, authorization_sha256: str) -> None:
    known = all_known_states(tasks, authorization_sha256)
    statuses = [
        valid_state_status(state_paths(str(snapshot["state_id"]))[3], authorization_sha256)
        for _, snapshot in known
    ]
    if len(known) != 288 or len(statuses) != 288 or not all(statuses):
        return
    raw = pd.concat([
        pd.read_parquet(state_paths(str(snapshot["state_id"]))[0])
        for _, snapshot in known
    ], ignore_index=True)
    grouped = pd.concat([
        pd.read_parquet(state_paths(str(snapshot["state_id"]))[1])
        for _, snapshot in known
    ], ignore_index=True)
    atomic_parquet(raw, OUT / "r12_seed_labels.parquet")
    atomic_parquet(grouped, OUT / "r12_grouped_labels.parquet")
    status_rows = [row for row in statuses if row is not None]
    checks = {
        "exactly_36_source_runs": len(tasks) == 36
        and sum(valid_source(task, authorization_sha256) is not None for task in tasks) == 36,
        "exactly_288_states": grouped.state_id.nunique() == 288,
        "unique_state_ids": len(known) == len({snapshot["state_id"] for _, snapshot in known}),
        "true_full_bank_scope_only": set(grouped.label_scope) == {FULL_BANK_SCOPE},
        "all_requested_banks_are_24_rules": set(grouped.requested_bank_count.astype(int)) == {24},
        "all_unique_candidates_present": bool(
            grouped.groupby("state_id").size().eq(
                grouped.groupby("state_id").full_bank_unique_count.first()
            ).all()
        ),
        "unit_inclusion_probability": set(grouped.inclusion_probability.astype(float)) == {1.0},
        "two_crn_seeds": set(raw.continuation_seed) == set(
            load_json(CONFIG_PATH)["rng"]["continuation_crn_seeds"]
        ),
        "frozen_horizon_only": set(raw.horizon.astype(int))
        == {int(authorization["selected_horizon"])}
        and set(grouped.horizon.astype(int)) == {int(authorization["selected_horizon"])},
        "all_feasible": bool(grouped.candidate_feasible.all()),
        "all_labels_post_source": bool(grouped.labels_post_source_trajectory.all()),
        "decoder_seconds_present": bool(raw.candidate_continuation_decoder_seconds.gt(0).all()),
        "authorization_hash_uniform": set(grouped.r12_horizon_freeze_sha256)
        == {authorization_sha256},
        "r13_r14_locked": not (
            ROOT / "outputs/phase6j_caur/r13_selection/access_ledger.json"
        ).exists() and not (
            ROOT / "outputs/phase6j_caur/r14_holdout/access_ledger.json"
        ).exists(),
    }
    manifest_rows = [{
        "relative_path": str(path.relative_to(ROOT)),
        "size_bytes": path.stat().st_size,
        "sha256": digest(path),
    } for path in _collection_files()]
    manifest_path = OUT / "collection_shard_manifest.csv"
    atomic_csv(pd.DataFrame(manifest_rows), manifest_path)
    integrity = {
        "schema": "phase6j-caur-r12-collection-integrity-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": {key: bool(value) for key, value in checks.items()},
        "r12_horizon_freeze_sha256": authorization_sha256,
        "selected_horizon": int(authorization["selected_horizon"]),
        "source_runs": 36,
        "states": int(grouped.state_id.nunique()),
        "unique_candidate_state_pairs": int(
            grouped[["state_id", "target_set_id"]].drop_duplicates().shape[0]
        ),
        "raw_seed_rows": len(raw),
        "grouped_rows": len(grouped),
        "repair_decoder_evaluations": sum(
            int(row["repair_decoder_evaluations"]) for row in status_rows
        ),
        "continuation_decoder_evaluations": sum(
            int(row["continuation_decoder_evaluations"]) for row in status_rows
        ),
        "decoder_seconds": sum(float(row["decoder_seconds"]) for row in status_rows),
        "state_elapsed_seconds": sum(float(row["elapsed_seconds"]) for row in status_rows),
        "shard_files": len(manifest_rows),
        "shard_manifest_sha256": digest(manifest_path),
        "r12_seed_labels_sha256": digest(OUT / "r12_seed_labels.parquet"),
        "r12_grouped_labels_sha256": digest(OUT / "r12_grouped_labels.parquet"),
        "r13_accessed": False,
        "r14_accessed": False,
    }
    atomic_json(integrity, OUT / "collection_integrity.json")
    report = f"""# Phase 6J R12 Full-Bank Collection

- Status: `{integrity['status']}`
- Source trajectories: {integrity['source_runs']}/36
- State lists: {integrity['states']}/288
- Unique state-candidate pairs: {integrity['unique_candidate_state_pairs']}
- Raw paired-seed rows: {integrity['raw_seed_rows']}
- Frozen continuation horizon: H={integrity['selected_horizon']}
- Repair decoder evaluations: {integrity['repair_decoder_evaluations']}
- Continuation decoder evaluations: {integrity['continuation_decoder_evaluations']}
- Measured continuation decoder seconds: {integrity['decoder_seconds']:.3f}
- Measured state collection seconds: {integrity['state_elapsed_seconds']:.3f}

Every sampled state uses the true full deduplicated candidate bank. R13 and
R14 remained locked throughout collection.
"""
    (OUT / "phase6j_caur_r12_collection_report.md").write_text(report, encoding="utf-8")
    if integrity["status"] != "PASS":
        raise RuntimeError(f"R12 collection integrity failed: {integrity}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--max-new-states", type=int)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    config = load_json(CONFIG_PATH)
    validate_preregistration(config)
    authorization = verify_r12_collection_authorization(
        FREEZE_PATH,
        project_root=ROOT,
        config_path=CONFIG_PATH,
        collection_script_path=Path(__file__),
    )
    authorization_sha256 = digest(FREEZE_PATH)
    tasks = build_collection_tasks(config)
    started = time.perf_counter()
    if args.summarize_only:
        summarize(tasks, authorization, authorization_sha256)
        write_progress(tasks, authorization_sha256, started)
        print("PHASE6J_CAUR_R12_COLLECTION_SUMMARY_RETURNED", flush=True)
        return

    alns_config = read_alns_config(config)
    if alns_config.candidate_trials != int(config["candidate_bank"]["candidate_trials_per_target"]):
        raise RuntimeError("candidate_trials no longer means eight trials per target")
    policy = load_policy(config, args.device)
    print(
        "PHASE6J_CAUR_R12_COLLECTION_START "
        f"device={args.device} horizon={authorization['selected_horizon']} expected_states=288",
        flush=True,
    )
    new_states = 0
    stop = False
    for task in tasks:
        source = valid_source(task, authorization_sha256) or run_source_trajectory(
            task, config, policy, alns_config, authorization_sha256
        )
        for snapshot in source["snapshots"]:
            status_path = state_paths(str(snapshot["state_id"]))[3]
            if valid_state_status(status_path, authorization_sha256) is not None:
                continue
            if args.max_new_states is not None and new_states >= args.max_new_states:
                stop = True
                break
            status = collect_state(
                task,
                snapshot,
                config,
                policy,
                alns_config,
                authorization,
                authorization_sha256,
            )
            new_states += 1
            progress = write_progress(tasks, authorization_sha256, started)
            print(
                f"PHASE6J_CAUR_R12_COLLECTION_STATE {progress['states_complete']}/288 "
                f"state_id={status['state_id']} candidates={status['unique_candidates']} "
                f"elapsed={status['elapsed_seconds']:.2f}s",
                flush=True,
            )
        if stop:
            break
    summarize(tasks, authorization, authorization_sha256)
    progress = write_progress(tasks, authorization_sha256, started)
    print(
        f"PHASE6J_CAUR_R12_COLLECTION_RETURNED status={progress['status']} "
        f"states={progress['states_complete']}/288",
        flush=True,
    )


if __name__ == "__main__":
    main()
