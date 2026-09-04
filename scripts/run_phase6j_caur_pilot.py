#!/usr/bin/env python3
"""Collect the resumable R12 full-bank continuation-horizon pilot."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
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
    select_shortest_adequate_horizon,
    validate_full_bank_feature_rows,
    validate_grouped_label_records,
)
from rcias_clgri.data.phase6j_access import load_phase6j_instance  # noqa: E402
from rcias_clgri.env.feasibility import check_schedule  # noqa: E402
from rcias_clgri.ni.live_inference import FrozenLiveInference  # noqa: E402
from rcias_clgri.search.alns import ALNSConfig  # noqa: E402
from rcias_clgri.search.common import Candidate, decode_candidate  # noqa: E402
from rcias_clgri.search.csgni import CSGNIConfig, solve_csgni  # noqa: E402
from rcias_clgri.search.phase6c import generate_revised_target_arms  # noqa: E402


CONFIG_PATH = ROOT / "configs/phase6j_caur.json"
COMMANDS_PATH = ROOT / "configs/phase6j_caur_command_manifest.json"
OUT = ROOT / "outputs/phase6j_caur/r12_pilot"
SOURCE_RUNS = OUT / "source_runs"
RAW_LABELS = OUT / "raw_seed_horizon_labels"
GROUPED_LABELS = OUT / "grouped_horizon_labels"
STATE_REPLAYS = OUT / "state_replays"
STATE_STATUS = OUT / "state_status"


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


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def candidate_to_dict(candidate: Candidate) -> dict[str, list[str]]:
    return {
        "operation_order": list(candidate.operation_order),
        "island_assignment": list(candidate.island_assignment),
        "w_assignment": list(candidate.w_assignment),
        "f_assignment": list(candidate.f_assignment),
    }


def candidate_from_dict(payload: dict[str, list[str]]) -> Candidate:
    return Candidate(
        tuple(payload["operation_order"]),
        tuple(payload["island_assignment"]),
        tuple(payload["w_assignment"]),
        tuple(payload["f_assignment"]),
    )


def search_stage(progress: float) -> str:
    bounds = ("0-20%", "20-40%", "40-60%", "60-80%", "80-100%")
    return bounds[min(4, int(max(0.0, min(progress, 0.999999)) * 5))]


class SnapshotObserver:
    """Capture source states; all forced outcomes are computed after solve returns."""

    def __init__(self, targets: list[float]) -> None:
        self.targets = targets
        self.events: list[dict[str, object]] = []

    def __call__(self, event: dict[str, object]) -> None:
        if not event.get("ni_eligible") or event.get("ni_state_id") is None:
            return
        current = event["current_before"]
        features = dict(event.get("ni_state_feature_summary") or {})
        self.events.append({
            "state_id": str(event["ni_state_id"]),
            "iteration": int(event["iteration"]),
            "search_progress": float(features.get("search_progress", 0.0)),
            "source_elapsed_wall_time": float(event["elapsed_time"]),
            "source_decoder_evaluations": int(event["decoder_evaluations"]),
            "current_makespan": float(current.makespan),
            "current_candidate": candidate_to_dict(current.candidate),
        })

    def selected(self) -> list[dict[str, object]]:
        if len(self.events) < len(self.targets):
            raise RuntimeError(
                f"source trajectory produced {len(self.events)} eligible states; "
                f"{len(self.targets)} are required"
            )
        selected: list[dict[str, object]] = []
        used: set[int] = set()
        for target in self.targets:
            index = min(
                (position for position in range(len(self.events)) if position not in used),
                key=lambda position: (
                    round(abs(float(self.events[position]["search_progress"]) - target), 12),
                    int(self.events[position]["iteration"]),
                ),
            )
            used.add(index)
            selected.append({**self.events[index], "target_progress": target})
        return selected


def read_alns_config(config: dict) -> ALNSConfig:
    path = ROOT / config["locked_inputs"]["frozen_alns_config"]
    if digest(path) != config["locked_inputs"]["frozen_alns_config_sha256"]:
        raise RuntimeError("frozen ALNS config hash mismatch")
    raw = load_json(path)
    return ALNSConfig(**{
        key: value for key, value in raw.items() if key in ALNSConfig.__dataclass_fields__
    })


def validate_preregistration(config: dict) -> None:
    commands = load_json(COMMANDS_PATH)
    manifest = ROOT / config["instance_suite"]["manifest"]
    audit = load_json(manifest.parent / "phase6j_integrity_audit.json")
    checks = (
        config["status"] == "PREREGISTERED_BEFORE_R12_PILOT_OR_R13_R14_CONTENT_ACCESS",
        commands["config_sha256"] == digest(CONFIG_PATH),
        config["instance_suite"]["manifest_sha256"] == digest(manifest),
        audit["status"] == "PASS" and all(audit["checks"].values()),
        not (ROOT / "outputs/phase6j_caur/r13_selection/access_ledger.json").exists(),
        not (ROOT / "outputs/phase6j_caur/r14_holdout/access_ledger.json").exists(),
    )
    if not all(checks):
        raise RuntimeError("Phase 6J preregistration or access-lock boundary failed")


def build_tasks(config: dict) -> list[dict]:
    manifest = pd.read_csv(ROOT / config["instance_suite"]["manifest"])
    fit = manifest[
        manifest.caur_split.eq("CAUR_FIT") & manifest.cell_replicate.eq("C02")
    ].copy()
    if len(fit) != 9 or set(fit.replicate) != {"R12"}:
        raise RuntimeError("R12 pilot must contain nine C02 structural cells")
    fit["scale_order"] = fit.scale.map({"S": 0, "M": 1, "L": 2})
    seed = int(config["rng"]["pilot_trajectory_seeds"][0])
    return [{
        "instance_id": row.instance_id,
        "instance_relative_path": row.relative_path,
        "instance_sha256": row.sha256,
        "scale": row.scale,
        "CF_level": row.CF_level,
        "RI_level": row.RI_level,
        "TI_level": row.TI_level,
        "trajectory_seed": seed,
    } for row in fit.sort_values(["scale_order", "CF_level"]).itertuples(index=False)]


def source_path(task: dict) -> Path:
    return SOURCE_RUNS / f"{task['instance_id']}.json"


def valid_source(task: dict) -> dict | None:
    path = source_path(task)
    if not path.is_file():
        return None
    try:
        record = load_json(path)
    except (OSError, json.JSONDecodeError):
        return None
    if (
        record.get("status") != "COMPLETE"
        or record.get("instance_sha256") != task["instance_sha256"]
        or len(record.get("snapshots", [])) != 3
        or len({row["state_id"] for row in record["snapshots"]}) != 3
    ):
        return None
    return record


def run_source_trajectory(
    task: dict,
    config: dict,
    policy: FrozenLiveInference,
    alns_config: ALNSConfig,
) -> dict:
    instance_path = ROOT / config["instance_suite"]["root"] / task["instance_relative_path"]
    instance = load_phase6j_instance(instance_path)
    if digest(instance_path) != task["instance_sha256"]:
        raise RuntimeError("R12 instance hash mismatch")
    budget = 0.25 * instance.num_operations
    observer = SnapshotObserver([float(value) for value in config["r12_pilot"]["progress_anchors"]])
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
        raise RuntimeError({"instance_id": task["instance_id"], "violations": feasibility["violations"]})
    record = {
        "schema": "phase6j-caur-r12-pilot-source-run-v1",
        "status": "COMPLETE",
        "implementation_commit": git_head(),
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


def valid_state_status(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        record = load_json(path)
        raw, grouped, replay, _ = state_paths(str(record["state_id"]))
    except (OSError, KeyError, json.JSONDecodeError):
        return None
    if (
        record.get("status") != "COMPLETE"
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
    policy: FrozenLiveInference,
    alns_config: ALNSConfig,
) -> dict:
    state_id = str(snapshot["state_id"])
    raw_path, grouped_path, replay_path, status_path = state_paths(state_id)
    existing = valid_state_status(status_path)
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
    top8 = {
        arm.target_set_id for arm in select_top_eight_audit_candidates(bank.arms, roles)
    }
    critical, bottleneck, bottleneck_proxy = critical_and_bottleneck_operations(
        instance, current
    )
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
    horizons = [int(value) for value in config["r12_pilot"]["horizons"]]
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
        if not decoded.candidate.feasible or not check_schedule(instance, decoded.candidate.schedule)["feasible"]:
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
                horizons=horizons,
                config=alns_config,
            )
            for seed in continuation_seeds
        }

    fallback_continuations = continuations[fallback_id]
    raw_rows: list[dict] = []
    grouped_rows: list[dict] = []
    arm_by_id = {arm.target_set_id: arm for arm in bank.arms}
    for target_id, arm_continuations in continuations.items():
        arm = arm_by_id[target_id]
        decoded = decoded_by_id[target_id]
        immediate = (current.makespan - decoded.candidate.makespan) / current.makespan
        base = {
            **task,
            **feature_by_id[target_id],
            "split": "R12_CAUR_PILOT",
            "target_progress": snapshot["target_progress"],
            "search_progress": progress,
            "search_stage": search_stage(progress),
            "source_iteration": snapshot["iteration"],
            "source_elapsed_wall_time": snapshot["source_elapsed_wall_time"],
            "fallback_target_set_id": fallback_id,
            "target_operation_ids": json.dumps(arm.destroyed_operations),
            "origin_rules": json.dumps(arm.origin_rules),
            "origin_families": json.dumps(arm.origin_families),
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
            "bottleneck_proxy": bottleneck_proxy,
            "critical_operation_ids": json.dumps(critical),
            "bottleneck_operation_ids": json.dumps(bottleneck),
            "labels_post_source_trajectory": True,
        }
        for seed in continuation_seeds:
            for horizon in horizons:
                candidate_prefix = arm_continuations[seed][horizon]
                fallback_prefix = fallback_continuations[seed][horizon]
                candidate_result = candidate_prefix.result
                fallback_result = fallback_prefix.result
                if candidate_result.derived_seed != fallback_result.derived_seed:
                    raise RuntimeError("candidate/fallback CRN identity mismatch")
                raw_rows.append({
                    **base,
                    "continuation_seed": seed,
                    "horizon": horizon,
                    "candidate_continuation_best_makespan": candidate_result.best_makespan,
                    "fallback_continuation_best_makespan": fallback_result.best_makespan,
                    "continuation_advantage": fallback_relative_advantage(
                        fallback_result.best_makespan,
                        candidate_result.best_makespan,
                        current.makespan,
                    ),
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
        for horizon in horizons:
            candidates = [arm_continuations[seed][horizon].result for seed in continuation_seeds]
            fallbacks = [fallback_continuations[seed][horizon].result for seed in continuation_seeds]
            advantages = [
                fallback_relative_advantage(fallback.best_makespan, candidate.best_makespan, current.makespan)
                for candidate, fallback in zip(candidates, fallbacks)
            ]
            grouped_rows.append({
                **base,
                "horizon": horizon,
                "continuation_seed_count": len(continuation_seeds),
                "continuation_advantage_mean": float(np.mean(advantages)),
                "continuation_advantage_std": float(np.std(advantages)),
                "beats_fallback": float(np.mean(np.asarray(advantages) > 0)),
                "continuation_best_makespan": float(np.mean([row.best_makespan for row in candidates])),
                "fallback_continuation_best_makespan": float(np.mean([row.best_makespan for row in fallbacks])),
                "decoder_seconds": float(sum(
                    arm_continuations[seed][horizon].decoder_seconds for seed in continuation_seeds
                )),
            })

    validate_grouped_label_records([
        row for row in grouped_rows if int(row["horizon"]) == max(horizons)
    ])
    raw_frame = pd.DataFrame(raw_rows).sort_values(
        ["target_set_id", "continuation_seed", "horizon"]
    ).reset_index(drop=True)
    grouped_frame = pd.DataFrame(grouped_rows).sort_values(
        ["target_set_id", "horizon"]
    ).reset_index(drop=True)
    atomic_parquet(raw_frame, raw_path)
    atomic_parquet(grouped_frame, grouped_path)
    replay = {
        "schema": "phase6j-caur-r12-pilot-state-replay-v1",
        "implementation_commit": git_head(),
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
        "schema": "phase6j-caur-r12-pilot-state-status-v1",
        "status": "COMPLETE",
        "implementation_commit": git_head(),
        "state_id": state_id,
        "instance_id": task["instance_id"],
        "unique_candidates": generated.unique_arm_count,
        "raw_rows": len(raw_frame),
        "grouped_rows": len(grouped_frame),
        "horizons": horizons,
        "continuation_seeds": continuation_seeds,
        "repair_decoder_evaluations": int(grouped_frame.drop_duplicates("target_set_id").repair_decoder_evaluations.sum()),
        "continuation_decoder_evaluations": int(
            raw_frame[raw_frame.horizon.eq(max(horizons))]
            .candidate_continuation_decoder_evaluations.sum()
        ),
        "decoder_seconds": float(
            raw_frame[raw_frame.horizon.eq(max(horizons))]
            .candidate_continuation_decoder_seconds.sum()
        ),
        "forced_decode_seconds": forced_runtime_seconds,
        "elapsed_seconds": time.perf_counter() - started,
        "raw_labels_sha256": digest(raw_path),
        "grouped_labels_sha256": digest(grouped_path),
        "replay_sha256": digest(replay_path),
    }
    atomic_json(status, status_path)
    return status


def all_known_states(tasks: list[dict]) -> list[tuple[dict, dict]]:
    known: list[tuple[dict, dict]] = []
    for task in tasks:
        source = valid_source(task)
        if source is not None:
            known.extend((task, snapshot) for snapshot in source["snapshots"])
    return known


def write_progress(tasks: list[dict], process_started: float) -> dict:
    known = all_known_states(tasks)
    statuses = [
        valid_state_status(state_paths(str(snapshot["state_id"]))[3])
        for _, snapshot in known
    ]
    completed = [row for row in statuses if row is not None]
    elapsed = time.perf_counter() - process_started
    measured = sum(float(row["elapsed_seconds"]) for row in completed)
    seconds_per_state = measured / len(completed) if completed else None
    remaining_seconds = (
        seconds_per_state * (27 - len(completed)) if seconds_per_state is not None else None
    )
    payload = {
        "schema": "phase6j-caur-r12-pilot-progress-v1",
        "status": "COMPLETE" if len(completed) == 27 else "RUNNING",
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_runs_complete": sum(valid_source(task) is not None for task in tasks),
        "source_runs_expected": 9,
        "states_complete": len(completed),
        "states_expected": 27,
        "unique_candidates_complete": sum(int(row["unique_candidates"]) for row in completed),
        "process_elapsed_seconds": elapsed,
        "measured_seconds_per_state": seconds_per_state,
        "measured_remaining_seconds": remaining_seconds,
        "decoder_seconds_complete": sum(float(row["decoder_seconds"]) for row in completed),
    }
    atomic_json(payload, OUT / "progress.json")
    return payload


def _spearman(left: pd.Series, right: pd.Series) -> float:
    if left.nunique() <= 1 or right.nunique() <= 1:
        return 0.0
    value = left.rank(method="average").corr(right.rank(method="average"))
    return 0.0 if pd.isna(value) else float(value)


def summarize(tasks: list[dict]) -> None:
    known = all_known_states(tasks)
    statuses = [
        valid_state_status(state_paths(str(snapshot["state_id"]))[3])
        for _, snapshot in known
    ]
    complete = len(known) == 27 and len(statuses) == 27 and all(row is not None for row in statuses)
    if not complete:
        return
    raw = pd.concat([
        pd.read_parquet(state_paths(str(snapshot["state_id"]))[0])
        for _, snapshot in known
    ], ignore_index=True)
    grouped = pd.concat([
        pd.read_parquet(state_paths(str(snapshot["state_id"]))[1])
        for _, snapshot in known
    ], ignore_index=True)
    atomic_parquet(raw, OUT / "pilot_seed_horizon_labels.parquet")
    atomic_parquet(grouped, OUT / "pilot_grouped_horizon_labels.parquet")

    horizons = sorted(grouped.horizon.unique())
    reference = grouped[grouped.horizon.eq(12)][
        ["state_id", "target_set_id", "continuation_advantage_mean"]
    ].rename(columns={"continuation_advantage_mean": "reference_advantage"})
    metric_rows = []
    per_state_rows = []
    for horizon in horizons:
        current = grouped[grouped.horizon.eq(horizon)].merge(
            reference, on=["state_id", "target_set_id"], validate="one_to_one"
        )
        for state_id, state in current.groupby("state_id"):
            predicted = state.sort_values(
                ["continuation_advantage_mean", "target_set_id"], ascending=[False, True]
            ).iloc[0]
            actual = state.sort_values(
                ["reference_advantage", "target_set_id"], ascending=[False, True]
            ).iloc[0]
            shifted = state.reference_advantage - state.reference_advantage.min()
            denominator = float(shifted.max())
            ndcg1 = 1.0 if denominator <= 0 else float(
                shifted[state.target_set_id.eq(predicted.target_set_id)].iloc[0] / denominator
            )
            per_state_rows.append({
                "state_id": state_id,
                "scale": state.scale.iloc[0],
                "CF_level": state.CF_level.iloc[0],
                "horizon": int(horizon),
                "spearman_with_h12": _spearman(
                    state.continuation_advantage_mean, state.reference_advantage
                ),
                "ndcg_at_1_with_h12": ndcg1,
                "top1_agreement_with_h12": predicted.target_set_id == actual.target_set_id,
            })
        rows = pd.DataFrame([row for row in per_state_rows if row["horizon"] == horizon])
        by_scale = rows.groupby("scale").spearman_with_h12.mean().to_dict()
        metric_rows.append({
            "horizon": int(horizon),
            "median_within_state_spearman": float(rows.spearman_with_h12.median()),
            "mean_ndcg_at_1": float(rows.ndcg_at_1_with_h12.mean()),
            "top1_agreement": float(rows.top1_agreement_with_h12.mean()),
            "mean_spearman_S": float(by_scale["S"]),
            "mean_spearman_M": float(by_scale["M"]),
            "mean_spearman_L": float(by_scale["L"]),
            "mean_spearman_by_scale": {key: float(value) for key, value in by_scale.items()},
        })
    metrics = pd.DataFrame(metric_rows)
    atomic_csv(metrics.drop(columns="mean_spearman_by_scale"), OUT / "horizon_comparison.csv")
    atomic_csv(pd.DataFrame(per_state_rows), OUT / "horizon_comparison_by_state.csv")
    selected_horizon = select_shortest_adequate_horizon({
        int(row["horizon"]): row for row in metric_rows
    })
    decision = {
        "schema": "phase6j-caur-r12-horizon-decision-v1",
        "status": "FROZEN_FROM_R12_PILOT_ONLY",
        "selected_horizon": selected_horizon,
        "rule": load_json(CONFIG_PATH)["r12_pilot"]["horizon_selection"],
        "metrics": metric_rows,
        "solver_performance_used": False,
        "r13_accessed": False,
        "r14_accessed": False,
    }
    atomic_json(decision, OUT / "horizon_decision.json")

    selected = grouped[grouped.horizon.eq(selected_horizon)].copy()
    target_metrics = []
    for state_id, state in selected.groupby("state_id"):
        target_metrics.append({
            "state_id": state_id,
            "scale": state.scale.iloc[0],
            "CF_level": state.CF_level.iloc[0],
            "spearman_immediate_vs_continuation": _spearman(
                state.immediate_utility, state.continuation_advantage_mean
            ),
            "top1_agreement": (
                state.sort_values(["immediate_utility", "target_set_id"], ascending=[False, True]).target_set_id.iloc[0]
                == state.sort_values(["continuation_advantage_mean", "target_set_id"], ascending=[False, True]).target_set_id.iloc[0]
            ),
        })
    target_frame = pd.DataFrame(target_metrics)
    atomic_csv(target_frame, OUT / "continuation_vs_immediate_by_state.csv")
    atomic_json({
        "schema": "phase6j-caur-continuation-vs-immediate-v1",
        "selected_horizon": selected_horizon,
        "median_within_state_spearman": float(target_frame.spearman_immediate_vs_continuation.median()),
        "mean_within_state_spearman": float(target_frame.spearman_immediate_vs_continuation.mean()),
        "top1_agreement": float(target_frame.top1_agreement.mean()),
        "primary_target": "fallback_relative_continuation_advantage",
        "immediate_target_role": "auxiliary_only",
    }, OUT / "continuation_vs_immediate_target_report.json")

    source_bias = selected.groupby(
        ["origin_family", "primary_origin_rule"], dropna=False
    ).agg(
        candidate_rows=("target_set_id", "size"),
        states=("state_id", "nunique"),
        mean_continuation_advantage=("continuation_advantage_mean", "mean"),
        positive_rate=("continuation_advantage_mean", lambda values: float((values > 0).mean())),
        mean_immediate_utility=("immediate_utility", "mean"),
    ).reset_index()
    atomic_csv(source_bias, OUT / "candidate_source_bias_audit.csv")

    status_rows = [row for row in statuses if row is not None]
    requested = set(grouped.requested_bank_count.astype(int))
    complete_checks = {
        "exactly_27_states": grouped.state_id.nunique() == 27,
        "true_full_bank_scope_only": set(grouped.label_scope) == {FULL_BANK_SCOPE},
        "all_requested_banks_are_24_rules": requested == {24},
        "all_unique_candidates_present": bool(
            grouped.groupby(["state_id", "horizon"]).size().eq(
                grouped.groupby(["state_id", "horizon"]).full_bank_unique_count.first()
            ).all()
        ),
        "two_crn_seeds": set(raw.continuation_seed) == set(load_json(CONFIG_PATH)["rng"]["continuation_crn_seeds"]),
        "three_horizons": set(raw.horizon) == {4, 8, 12},
        "all_feasible": bool(grouped.candidate_feasible.all()),
        "all_labels_post_source": bool(grouped.labels_post_source_trajectory.all()),
        "decoder_seconds_present": bool(raw.candidate_continuation_decoder_seconds.gt(0).all()),
        "r13_r14_locked": not (
            ROOT / "outputs/phase6j_caur/r13_selection/access_ledger.json"
        ).exists() and not (
            ROOT / "outputs/phase6j_caur/r14_holdout/access_ledger.json"
        ).exists(),
    }
    audit = {
        "schema": "phase6j-caur-r12-pilot-completeness-v1",
        "status": "PASS" if all(complete_checks.values()) else "FAIL",
        "checks": {key: bool(value) for key, value in complete_checks.items()},
        "states": int(grouped.state_id.nunique()),
        "unique_candidate_state_pairs": int(grouped[["state_id", "target_set_id"]].drop_duplicates().shape[0]),
        "raw_seed_horizon_rows": len(raw),
        "grouped_horizon_rows": len(grouped),
        "repair_decoder_evaluations": sum(int(row["repair_decoder_evaluations"]) for row in status_rows),
        "continuation_decoder_evaluations": sum(int(row["continuation_decoder_evaluations"]) for row in status_rows),
        "decoder_seconds": sum(float(row["decoder_seconds"]) for row in status_rows),
        "state_elapsed_seconds": sum(float(row["elapsed_seconds"]) for row in status_rows),
        "selected_horizon": selected_horizon,
    }
    atomic_json(audit, OUT / "full_bank_cost_completeness_audit.json")
    report = f"""# Phase 6J R12 Horizon and Full-Bank Pilot

- Status: `{audit['status']}`
- States: {audit['states']}/27
- Unique state-candidate pairs: {audit['unique_candidate_state_pairs']}
- Raw seed/horizon rows: {audit['raw_seed_horizon_rows']}
- Selected horizon: H={selected_horizon}
- Repair decoder evaluations: {audit['repair_decoder_evaluations']}
- Continuation decoder evaluations through H=12: {audit['continuation_decoder_evaluations']}
- Measured continuation decoder seconds: {audit['decoder_seconds']:.3f}
- Total measured state collection seconds: {audit['state_elapsed_seconds']:.3f}

The horizon was chosen only by agreement with H=12. R13 and R14 remained
locked. Immediate utility was retained only as an auxiliary diagnostic.
"""
    (OUT / "phase6j_caur_r12_pilot_report.md").write_text(report, encoding="utf-8")
    if audit["status"] != "PASS":
        raise RuntimeError(f"R12 pilot completeness failed: {audit}")


def load_policy(config: dict, device: str) -> FrozenLiveInference:
    policy = FrozenLiveInference(
        ROOT / config["locked_inputs"]["phase6f_experiment_freeze"],
        device=device,
        proposal_seed_namespace=int(config["rng"]["proposal_namespace"]),
        deployment_artifact=ROOT / config["locked_inputs"]["phase6h_policy"],
    )
    if policy.checkpoint_sha256 != config["locked_inputs"]["phase6f_checkpoint_sha256"]:
        raise RuntimeError("Phase 6F checkpoint hash differs from preregistration")
    if policy.deployment_artifact_sha256 != config["locked_inputs"]["phase6h_policy_sha256"]:
        raise RuntimeError("Phase 6H policy hash differs from preregistration")
    return policy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--max-new-states", type=int)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    config = load_json(CONFIG_PATH)
    validate_preregistration(config)
    tasks = build_tasks(config)
    started = time.perf_counter()
    if args.summarize_only:
        summarize(tasks)
        write_progress(tasks, started)
        print("PHASE6J_CAUR_R12_PILOT_SUMMARY_RETURNED", flush=True)
        return

    alns_config = read_alns_config(config)
    if alns_config.candidate_trials != int(config["candidate_bank"]["candidate_trials_per_target"]):
        raise RuntimeError("candidate_trials no longer means eight trials per target")
    policy = load_policy(config, args.device)
    print(f"PHASE6J_CAUR_R12_PILOT_START device={args.device} expected_states=27", flush=True)
    new_states = 0
    stop = False
    for task in tasks:
        source = valid_source(task) or run_source_trajectory(task, config, policy, alns_config)
        for snapshot in source["snapshots"]:
            status_path = state_paths(str(snapshot["state_id"]))[3]
            if valid_state_status(status_path) is not None:
                continue
            if args.max_new_states is not None and new_states >= args.max_new_states:
                stop = True
                break
            status = collect_state(task, snapshot, config, policy, alns_config)
            new_states += 1
            progress = write_progress(tasks, started)
            print(
                f"PHASE6J_CAUR_R12_STATE {progress['states_complete']}/27 "
                f"state_id={status['state_id']} candidates={status['unique_candidates']} "
                f"elapsed={status['elapsed_seconds']:.2f}s",
                flush=True,
            )
        if stop:
            break
    summarize(tasks)
    progress = write_progress(tasks, started)
    print(
        f"PHASE6J_CAUR_R12_PILOT_RETURNED status={progress['status']} "
        f"states={progress['states_complete']}/27",
        flush=True,
    )


if __name__ == "__main__":
    main()
