#!/usr/bin/env python3
"""Generate the frozen Phase 6N expanded R12 continuation dataset."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_clgri.analysis.phase6i_mr import (  # noqa: E402
    FrozenArmPrediction,
    decode_forced_candidate,
)
from rcias_clgri.analysis.phase6j_caur import (  # noqa: E402
    FULL_BANK_SCOPE,
    continue_frozen_alns_at_horizons,
    critical_and_bottleneck_operations,
    fallback_relative_advantage,
    grouped_oof_fold,
)
from rcias_clgri.analysis.phase6l_legacy_score import (  # noqa: E402
    build_score_free_candidate_source_features,
    select_score_free_fallback,
)
from rcias_clgri.data.phase6j_access import load_phase6j_instance  # noqa: E402
from rcias_clgri.env.feasibility import check_schedule  # noqa: E402
from rcias_clgri.search.alns import solve_alns  # noqa: E402
from rcias_clgri.search.common import Candidate, decode_candidate  # noqa: E402
from rcias_clgri.search.phase6c import generate_revised_target_arms  # noqa: E402
from scripts.run_phase6j_caur_pilot import read_alns_config, search_stage  # noqa: E402


CONFIG_PATH = ROOT / "configs/phase6n_candidate_conditioned_csg_v1.json"
PREREGISTRATION = (
    ROOT / "outputs/phase6n_candidate_conditioned_csg_v1/preregistration/preregistration.json"
)
DATA_PLAN = (
    ROOT
    / "outputs/phase6n_candidate_conditioned_csg_v1/preregistration/data_generation_plan.json"
)
SOURCE_HASHES = (
    ROOT / "outputs/phase6n_candidate_conditioned_csg_v1/preregistration/source_hashes.json"
)
OUT = ROOT / "outputs/phase6n_candidate_conditioned_csg_v1/data"
NEW = OUT / "new_r12_collection"
SOURCE_RUNS = NEW / "source_runs"
RAW_SHARDS = NEW / "raw_seed_labels"
GROUPED_SHARDS = NEW / "grouped_labels"
STATE_REPLAYS = NEW / "state_replays"
STATE_STATUS = NEW / "state_status"
COMBINED = OUT / "combined"
IMPLEMENTATION = OUT / "collection_implementation.json"
PROGRESS = OUT / "progress.json"
REPORT = ROOT / "docs/reports/phase6n_data_generation_report.md"
ORIGINAL_GROUPED = (
    ROOT
    / "outputs/phase6l_legacy_score_decoupling_v1/data/r12_score_free_grouped_labels.parquet"
)
ORIGINAL_RAW = (
    ROOT
    / "outputs/phase6l_legacy_score_decoupling_v1/data/r12_score_free_seed_labels.parquet"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def candidate_to_dict(candidate: Candidate) -> dict[str, list[str]]:
    return {
        "operation_order": list(candidate.operation_order),
        "island_assignment": list(candidate.island_assignment),
        "w_assignment": list(candidate.w_assignment),
        "f_assignment": list(candidate.f_assignment),
    }


def candidate_from_dict(value: dict[str, list[str]]) -> Candidate:
    return Candidate(
        tuple(value["operation_order"]),
        tuple(value["island_assignment"]),
        tuple(value["w_assignment"]),
        tuple(value["f_assignment"]),
    )


def validate_preregistration() -> tuple[dict, dict, str, str]:
    config = load_json(CONFIG_PATH)
    preregistration = load_json(PREREGISTRATION)
    data_plan = load_json(DATA_PLAN)
    require(
        config["status"] == "PREREGISTERED_BEFORE_NEW_ROLLOUT_OR_OPTIMIZER_STEP",
        "invalid Phase 6N config status",
    )
    require(
        preregistration["status"] == "FROZEN_BEFORE_NEW_ROLLOUT_OR_OPTIMIZER_STEP",
        "Phase 6N preregistration is not frozen",
    )
    require(
        preregistration["data_generation_plan_sha256"] == digest(DATA_PLAN),
        "Phase 6N data plan changed",
    )
    require(
        preregistration["source_hashes_sha256"] == digest(SOURCE_HASHES),
        "Phase 6N source hash record changed",
    )
    for relative, expected in load_json(SOURCE_HASHES)["files"].items():
        path = ROOT / relative
        require(path.is_file() and digest(path) == expected, f"locked input changed: {relative}")
    require(data_plan["new_states"] == 576, "new state count changed")
    require(data_plan["combined_states"] == 864, "combined state count changed")
    require(data_plan["historical_score_calls"] == 0, "historical scorer enabled")
    for relative in (
        "outputs/phase6j_caur/r13_selection/access_ledger.json",
        "outputs/phase6j_caur/r14_holdout/access_ledger.json",
        "outputs/phase6n_candidate_conditioned_csg_v1/r13_selection/access_ledger.json",
        "outputs/phase6n_candidate_conditioned_csg_v1/r14_holdout/access_ledger.json",
    ):
        require(not (ROOT / relative).exists(), f"forbidden holdout access: {relative}")
    require(
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", preregistration["head_at_freeze"], "HEAD"],
            cwd=ROOT,
            check=False,
        ).returncode
        == 0,
        "preregistration commit is not an ancestor of HEAD",
    )
    return config, data_plan, digest(PREREGISTRATION), digest(Path(__file__))


def freeze_implementation(preregistration_sha256: str, script_sha256: str) -> dict:
    paths = [
        Path(__file__),
        ROOT / "rcias_clgri/analysis/phase6i_mr.py",
        ROOT / "rcias_clgri/analysis/phase6j_caur.py",
        ROOT / "rcias_clgri/analysis/phase6l_legacy_score.py",
        ROOT / "rcias_clgri/data/phase6j_access.py",
        ROOT / "rcias_clgri/search/alns.py",
        ROOT / "rcias_clgri/search/common.py",
        ROOT / "rcias_clgri/search/phase6c.py",
    ]
    payload = {
        "schema": "phase6n-data-generation-implementation-v1",
        "status": "FROZEN_BEFORE_FIRST_NEW_OUTCOME",
        "implementation_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "preregistration_sha256": preregistration_sha256,
        "data_plan_sha256": digest(DATA_PLAN),
        "code_sha256": {
            str(path.relative_to(ROOT)): digest(path) for path in paths
        },
        "worker_script_sha256": script_sha256,
        "historical_score_calls": 0,
        "r13_accessed": False,
        "r14_accessed": False,
    }
    if IMPLEMENTATION.exists():
        require(load_json(IMPLEMENTATION) == payload, "data implementation changed after outcomes")
    else:
        atomic_json(IMPLEMENTATION, payload)
    return payload


def build_tasks(config: dict, data_plan: dict) -> list[dict]:
    manifest = pd.read_csv(ROOT / config["locked_inputs"]["instance_manifest"])
    fit = manifest[manifest.caur_split.eq("CAUR_FIT")].copy()
    require(len(fit) == 18 and set(fit.replicate) == {"R12"}, "R12 FIT suite changed")
    require(set(fit.cell_replicate) == {"C01", "C02"}, "R12 cell replicates changed")
    fit["scale_order"] = fit.scale.map({"S": 0, "M": 1, "L": 2})
    tasks: list[dict] = []
    for row in fit.sort_values(
        ["scale_order", "CF_level", "cell_replicate"], kind="stable"
    ).itertuples(index=False):
        for seed in data_plan["new_trajectory_seeds"]:
            tasks.append(
                {
                    "instance_id": str(row.instance_id),
                    "instance_relative_path": str(row.relative_path),
                    "instance_sha256": str(row.sha256),
                    "scale": str(row.scale),
                    "CF_level": str(row.CF_level),
                    "RI_level": str(row.RI_level),
                    "TI_level": str(row.TI_level),
                    "cell_replicate": str(row.cell_replicate),
                    "trajectory_seed": int(seed),
                }
            )
    require(len(tasks) == 72, "Phase 6N requires exactly 72 new source trajectories")
    return tasks


class ALNSSnapshotObserver:
    """Select source states by frozen wall-clock progress after the run ends."""

    def __init__(self, instance_id: str, seed: int, budget: float, anchors: list[float]):
        self.instance_id = instance_id
        self.seed = seed
        self.budget = budget
        self.anchors = anchors
        self.events: list[dict] = []

    def __call__(self, event: dict) -> None:
        current = event["current_before"]
        iteration = int(event["iteration"])
        progress = min(float(event["elapsed_time"]) / self.budget, 0.999999)
        self.events.append(
            {
                "state_id": f"{self.instance_id}__seed{self.seed}__it{iteration:07d}",
                "iteration": iteration,
                "search_progress": progress,
                "source_elapsed_wall_time": float(event["elapsed_time"]),
                "source_decoder_evaluations": int(event["decoder_evaluations"]),
                "current_makespan": float(current.makespan),
                "current_candidate": candidate_to_dict(current.candidate),
            }
        )

    def selected(self) -> list[dict]:
        require(len(self.events) >= len(self.anchors), "source trajectory has too few states")
        selected: list[dict] = []
        used: set[int] = set()
        for target in self.anchors:
            index = min(
                (position for position in range(len(self.events)) if position not in used),
                key=lambda position: (
                    round(abs(self.events[position]["search_progress"] - target), 12),
                    self.events[position]["iteration"],
                ),
            )
            used.add(index)
            selected.append({**self.events[index], "target_progress": float(target)})
        return selected


def source_path(task: dict) -> Path:
    return SOURCE_RUNS / f"{task['instance_id']}__seed{task['trajectory_seed']}.json"


def valid_source(task: dict, implementation: dict) -> dict | None:
    path = source_path(task)
    if not path.is_file():
        return None
    try:
        record = load_json(path)
    except (OSError, json.JSONDecodeError):
        return None
    checks = (
        record.get("schema") == "phase6n-alns-source-trajectory-v1",
        record.get("status") == "COMPLETE",
        record.get("instance_id") == task["instance_id"],
        record.get("trajectory_seed") == task["trajectory_seed"],
        record.get("instance_sha256") == task["instance_sha256"],
        record.get("implementation_commit") == implementation["implementation_commit"],
        record.get("worker_script_sha256") == implementation["worker_script_sha256"],
        record.get("historical_score_calls") == 0,
        len(record.get("snapshots", [])) == 8,
    )
    return record if all(checks) else None


def run_source(task: dict, config: dict, data_plan: dict, alns_config, implementation: dict) -> dict:
    instance_path = ROOT / config["locked_inputs"]["instance_root"] / task[
        "instance_relative_path"
    ]
    require(digest(instance_path) == task["instance_sha256"], "instance hash changed")
    instance = load_phase6j_instance(instance_path)
    budget = 0.25 * instance.num_operations
    observer = ALNSSnapshotObserver(
        task["instance_id"],
        task["trajectory_seed"],
        budget,
        [float(value) for value in data_plan["progress_anchors"]],
    )
    print(
        json.dumps(
            {
                "event": "source_start",
                "instance_id": task["instance_id"],
                "trajectory_seed": task["trajectory_seed"],
                "budget_seconds": budget,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    result = solve_alns(
        instance,
        budget,
        task["trajectory_seed"],
        alns_config,
        observer,
    )
    feasibility = check_schedule(instance, result.best.schedule)
    require(feasibility["feasible"], "source ALNS returned an infeasible best")
    record = {
        "schema": "phase6n-alns-source-trajectory-v1",
        "status": "COMPLETE",
        "implementation_commit": implementation["implementation_commit"],
        "worker_script_sha256": implementation["worker_script_sha256"],
        **task,
        "source_sampler": data_plan["source_sampler"],
        "budget_seconds": budget,
        "runtime_seconds": result.runtime,
        "iterations": result.iterations,
        "decoder_evaluations": result.decoder_evaluations,
        "best_makespan": result.best.makespan,
        "feasible": True,
        "historical_score_calls": 0,
        "source_completed_before_labels": True,
        "snapshots": observer.selected(),
    }
    atomic_json(source_path(task), record)
    return record


def state_paths(state_id: str) -> tuple[Path, Path, Path, Path]:
    return (
        RAW_SHARDS / f"{state_id}.parquet",
        GROUPED_SHARDS / f"{state_id}.parquet",
        STATE_REPLAYS / f"{state_id}.json",
        STATE_STATUS / f"{state_id}.json",
    )


def valid_state(state_id: str, implementation: dict) -> dict | None:
    raw_path, grouped_path, replay_path, status_path = state_paths(state_id)
    if not status_path.is_file():
        return None
    try:
        record = load_json(status_path)
    except (OSError, json.JSONDecodeError):
        return None
    checks = (
        record.get("schema") == "phase6n-expanded-state-status-v1",
        record.get("status") == "COMPLETE",
        record.get("state_id") == state_id,
        record.get("implementation_commit") == implementation["implementation_commit"],
        record.get("worker_script_sha256") == implementation["worker_script_sha256"],
        record.get("historical_score_calls") == 0,
        raw_path.is_file(),
        grouped_path.is_file(),
        replay_path.is_file(),
    )
    if not all(checks):
        return None
    if (
        record.get("raw_sha256") != digest(raw_path)
        or record.get("grouped_sha256") != digest(grouped_path)
        or record.get("replay_sha256") != digest(replay_path)
    ):
        return None
    return record


def _prediction(arm) -> FrozenArmPrediction:
    return FrozenArmPrediction(
        target_set_id=arm.target_set_id,
        arm_family=arm.arm_family,
        origin_destroy_operator=arm.origin_destroy_operator,
        origin_rules=arm.origin_rules,
        destroyed_operations=arm.destroyed_operations,
        raw_score=0.0,
        raw_probability=0.0,
        raw_utility=0.0,
        calibrated_probability=0.0,
        calibrated_utility=0.0,
    )


def collect_state(
    task: dict,
    snapshot: dict,
    config: dict,
    data_plan: dict,
    alns_config,
    implementation: dict,
) -> dict:
    state_id = str(snapshot["state_id"])
    existing = valid_state(state_id, implementation)
    if existing is not None:
        return existing
    started = time.perf_counter()
    instance_path = ROOT / config["locked_inputs"]["instance_root"] / task[
        "instance_relative_path"
    ]
    instance = load_phase6j_instance(instance_path)
    current = decode_candidate(instance, candidate_from_dict(snapshot["current_candidate"]))
    require(
        math.isclose(current.makespan, float(snapshot["current_makespan"]), abs_tol=1e-9),
        f"source replay mismatch: {state_id}",
    )
    destroy_count = min(
        max(2, round(instance.num_operations * alns_config.destroy_fraction)),
        instance.num_operations,
    )
    generated = generate_revised_target_arms(
        instance,
        current,
        state_id,
        destroy_count,
        int(data_plan["seed_namespaces"]["proposal"]),
    )
    require(generated.requested_arm_count == 24, "proposal bank is not 24-rule")
    require(
        generated.unique_arm_count + generated.duplicate_arm_count == 24,
        "proposal dedup accounting failed",
    )
    fallback = select_score_free_fallback(generated)
    critical, bottleneck, bottleneck_proxy = critical_and_bottleneck_operations(
        instance, current
    )
    features = build_score_free_candidate_source_features(
        generated,
        state_id=state_id,
        operation_count=instance.num_operations,
        critical_operations=critical,
        bottleneck_operations=bottleneck,
    )
    require(
        [row["target_set_id"] for row in features]
        == [arm.target_set_id for arm in generated.arms],
        "candidate feature order drifted",
    )
    feature_by_id = {str(row["target_set_id"]): row for row in features}

    decoded = {}
    continuations = {}
    forced_seconds = 0.0
    horizon = int(data_plan["continuation_horizon"])
    crn_seeds = [int(value) for value in data_plan["continuation_crn_seeds"]]
    for arm in generated.arms:
        repaired = decode_forced_candidate(
            instance,
            current,
            _prediction(arm),
            state_id=state_id,
            repair_seed_namespace=int(data_plan["seed_namespaces"]["repair"]),
            candidate_trials=int(data_plan["candidate_trials_per_target"]),
        )
        require(repaired.candidate.feasible, "forced candidate is infeasible")
        require(
            check_schedule(instance, repaired.candidate.schedule)["feasible"],
            "forced candidate replay is infeasible",
        )
        decoded[arm.target_set_id] = repaired
        forced_seconds += repaired.runtime_ms / 1000.0
        continuations[arm.target_set_id] = {
            seed: continue_frozen_alns_at_horizons(
                instance,
                repaired.candidate,
                state_id=state_id,
                continuation_seed=seed,
                seed_namespace=int(data_plan["seed_namespaces"]["continuation"]),
                horizons=[horizon],
                config=alns_config,
            )[horizon]
            for seed in crn_seeds
        }

    fallback_id = fallback.target_set_id
    fallback_continuations = continuations[fallback_id]
    raw_rows: list[dict] = []
    grouped_rows: list[dict] = []
    for order_index, arm in enumerate(generated.arms):
        target_id = arm.target_set_id
        repaired = decoded[target_id]
        immediate = (current.makespan - repaired.candidate.makespan) / current.makespan
        base = {
            **task,
            **feature_by_id[target_id],
            "split": "R12_CAUR_FIT",
            "phase6n_data_origin": "NEW_ALNS_EXPANSION",
            "phase6n_generator_order_index": order_index,
            "target_progress": float(snapshot["target_progress"]),
            "search_progress": float(snapshot["search_progress"]),
            "search_stage": search_stage(float(snapshot["search_progress"])),
            "source_iteration": int(snapshot["iteration"]),
            "fallback_target_set_id": fallback_id,
            "target_operation_ids": json.dumps(arm.destroyed_operations),
            "origin_rules": json.dumps(arm.origin_rules),
            "origin_families": json.dumps(arm.origin_families),
            "immediate_utility": immediate,
            "candidate_feasible": True,
            "requested_bank_count": generated.requested_arm_count,
            "full_bank_unique_count": generated.unique_arm_count,
            "duplicate_bank_count": generated.duplicate_arm_count,
            "bottleneck_proxy": bottleneck_proxy,
            "critical_operation_ids": json.dumps(critical),
            "bottleneck_operation_ids": json.dumps(bottleneck),
            "labels_post_source_trajectory": True,
        }
        advantages = []
        candidate_makespans = []
        fallback_makespans = []
        for seed in crn_seeds:
            candidate_prefix = continuations[target_id][seed]
            fallback_prefix = fallback_continuations[seed]
            candidate_result = candidate_prefix.result
            fallback_result = fallback_prefix.result
            require(
                candidate_result.derived_seed == fallback_result.derived_seed,
                "candidate/fallback CRN identity mismatch",
            )
            advantage = fallback_relative_advantage(
                fallback_result.best_makespan,
                candidate_result.best_makespan,
                current.makespan,
            )
            advantages.append(advantage)
            candidate_makespans.append(candidate_result.best_makespan)
            fallback_makespans.append(fallback_result.best_makespan)
            raw_rows.append(
                {
                    "instance_id": task["instance_id"],
                    "scale": task["scale"],
                    "CF_level": task["CF_level"],
                    "state_id": state_id,
                    "target_set_id": target_id,
                    "fallback_target_set_id": fallback_id,
                    "continuation_seed": seed,
                    "horizon": horizon,
                    "candidate_continuation_best_makespan": candidate_result.best_makespan,
                    "fallback_continuation_best_makespan": fallback_result.best_makespan,
                    "continuation_advantage": advantage,
                    "paired_derived_seed": candidate_result.derived_seed,
                    "is_fallback": target_id == fallback_id,
                    "phase6n_data_origin": "NEW_ALNS_EXPANSION",
                }
            )
        grouped_rows.append(
            {
                **base,
                "horizon": horizon,
                "continuation_seed_count": len(crn_seeds),
                "continuation_advantage_mean": float(np.mean(advantages)),
                "continuation_advantage_std": float(np.std(advantages, ddof=0)),
                "beats_fallback": float(np.mean(np.asarray(advantages) > 0)),
                "continuation_best_makespan": float(np.mean(candidate_makespans)),
                "fallback_continuation_best_makespan": float(
                    np.mean(fallback_makespans)
                ),
                "oof_fold": grouped_oof_fold(task["scale"], task["CF_level"]),
            }
        )

    grouped_frame = pd.DataFrame(grouped_rows).sort_values(
        "target_set_id", kind="stable"
    ).reset_index(drop=True)
    raw_frame = pd.DataFrame(raw_rows).sort_values(
        ["target_set_id", "continuation_seed"], kind="stable"
    ).reset_index(drop=True)
    require(len(grouped_frame) == generated.unique_arm_count, "grouped bank incomplete")
    require(len(raw_frame) == 2 * generated.unique_arm_count, "raw bank incomplete")
    require(grouped_frame.is_fallback.sum() == 1, "canonical fallback is not unique")
    raw_path, grouped_path, replay_path, status_path = state_paths(state_id)
    atomic_parquet(raw_path, raw_frame)
    atomic_parquet(grouped_path, grouped_frame)
    replay = {
        "schema": "phase6n-expanded-state-replay-v1",
        "status": "COMPLETE",
        "implementation_commit": implementation["implementation_commit"],
        "worker_script_sha256": implementation["worker_script_sha256"],
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
        "full_bank_target_ids_in_generator_order": [
            arm.target_set_id for arm in generated.arms
        ],
        "full_bank_target_operations": {
            arm.target_set_id: list(arm.destroyed_operations) for arm in generated.arms
        },
        "historical_score_calls": 0,
        "source_completed_before_labels": True,
    }
    atomic_json(replay_path, replay)
    status = {
        "schema": "phase6n-expanded-state-status-v1",
        "status": "COMPLETE",
        "implementation_commit": implementation["implementation_commit"],
        "worker_script_sha256": implementation["worker_script_sha256"],
        "state_id": state_id,
        "instance_id": task["instance_id"],
        "scale": task["scale"],
        "unique_candidates": generated.unique_arm_count,
        "raw_rows": len(raw_frame),
        "grouped_rows": len(grouped_frame),
        "horizon": horizon,
        "continuation_seeds": crn_seeds,
        "repair_decoder_evaluations": int(
            len(grouped_frame) * data_plan["candidate_trials_per_target"]
        ),
        "continuation_decoder_evaluations": int(
            len(raw_frame) * horizon * alns_config.candidate_trials
        ),
        "forced_decode_seconds": forced_seconds,
        "elapsed_seconds": time.perf_counter() - started,
        "historical_score_calls": 0,
        "raw_sha256": digest(raw_path),
        "grouped_sha256": digest(grouped_path),
        "replay_sha256": digest(replay_path),
    }
    atomic_json(status_path, status)
    return status


def progress_payload(
    *,
    implementation: dict,
    process_started: float,
    source_complete: int,
    completed: list[dict],
    active: dict | None,
    status: str,
) -> dict:
    elapsed = time.perf_counter() - process_started
    measured = sum(float(row["elapsed_seconds"]) for row in completed)
    rate = measured / len(completed) if completed else None
    remaining = 576 - len(completed)
    return {
        "schema": "phase6n-data-generation-progress-v1",
        "status": status,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "implementation_commit": implementation["implementation_commit"],
        "worker_script_sha256": implementation["worker_script_sha256"],
        "source_runs_complete": source_complete,
        "source_runs_expected": 72,
        "states_complete": len(completed),
        "states_expected": 576,
        "unique_candidates_complete": int(
            sum(int(row["unique_candidates"]) for row in completed)
        ),
        "process_elapsed_seconds": elapsed,
        "measured_state_seconds": measured,
        "measured_seconds_per_state": rate,
        "measured_remaining_state_seconds": rate * remaining if rate is not None else None,
        "projected_total_seconds_from_preregistration": 9587.016576358,
        "active": active,
        "historical_score_calls": 0,
        "r13_accessed": False,
        "r14_accessed": False,
    }


def write_progress(**kwargs) -> dict:
    payload = progress_payload(**kwargs)
    atomic_json(PROGRESS, payload)
    return payload


def existing_statuses(implementation: dict) -> list[dict]:
    rows = []
    for path in sorted(STATE_STATUS.glob("*.json")):
        row = valid_state(path.stem, implementation)
        require(row is not None, f"invalid existing state shard: {path}")
        rows.append(row)
    return rows


def add_original_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["phase6n_data_origin"] = "ORIGINAL_PHASE6J_CAUR"
    if "continuation_seed" not in result:
        result["phase6n_generator_order_index"] = -1
    return result


def summarize(tasks: list[dict], implementation: dict, completed: list[dict]) -> dict:
    require(len(completed) == 576, "cannot summarize an incomplete expansion")
    new_grouped = pd.concat(
        [pd.read_parquet(path) for path in sorted(GROUPED_SHARDS.glob("*.parquet"))],
        ignore_index=True,
    ).sort_values(["state_id", "target_set_id"], kind="stable").reset_index(drop=True)
    new_raw = pd.concat(
        [pd.read_parquet(path) for path in sorted(RAW_SHARDS.glob("*.parquet"))],
        ignore_index=True,
    ).sort_values(
        ["state_id", "target_set_id", "continuation_seed"], kind="stable"
    ).reset_index(drop=True)
    require(new_grouped.state_id.nunique() == 576, "new state count mismatch")
    require(new_grouped.groupby("instance_id").state_id.nunique().eq(32).all(), "unequal new states")
    require(new_grouped.groupby("state_id").is_fallback.sum().eq(1).all(), "fallback mismatch")
    require(set(new_grouped.label_scope) == {FULL_BANK_SCOPE}, "label scope mismatch")
    require(new_grouped.candidate_feasible.all(), "infeasible new candidate")
    require(len(new_raw) == 2 * len(new_grouped), "new raw/grouped mismatch")

    original_grouped = add_original_diagnostics(pd.read_parquet(ORIGINAL_GROUPED))
    original_raw = add_original_diagnostics(pd.read_parquet(ORIGINAL_RAW))
    combined_grouped = pd.concat([original_grouped, new_grouped], ignore_index=True)
    combined_raw = pd.concat([original_raw, new_raw], ignore_index=True)
    combined_grouped = combined_grouped.sort_values(
        ["state_id", "target_set_id"], kind="stable"
    ).reset_index(drop=True)
    combined_grouped["phase6n_model_order_index"] = combined_grouped.groupby(
        "state_id", sort=False
    ).cumcount()
    combined_raw = combined_raw.sort_values(
        ["state_id", "target_set_id", "continuation_seed"], kind="stable"
    ).reset_index(drop=True)
    require(combined_grouped.state_id.nunique() == 864, "combined state count mismatch")
    require(
        combined_grouped.groupby("instance_id").state_id.nunique().eq(48).all(),
        "combined states are not equal per instance",
    )
    require(len(combined_raw) == 2 * len(combined_grouped), "combined CRN rows mismatch")
    require(
        combined_grouped[["state_id", "target_set_id"]].duplicated().sum() == 0,
        "duplicate combined state/candidate identity",
    )
    require(
        combined_grouped.groupby("state_id").requested_bank_count.first().eq(24).all(),
        "combined full-bank rule count mismatch",
    )

    new_grouped_path = NEW / "r12_expansion_grouped_labels.parquet"
    new_raw_path = NEW / "r12_expansion_seed_labels.parquet"
    combined_grouped_path = COMBINED / "r12_expanded_grouped_labels.parquet"
    combined_raw_path = COMBINED / "r12_expanded_seed_labels.parquet"
    atomic_parquet(new_grouped_path, new_grouped)
    atomic_parquet(new_raw_path, new_raw)
    atomic_parquet(combined_grouped_path, combined_grouped)
    atomic_parquet(combined_raw_path, combined_raw)

    shard_paths = []
    for directory in (SOURCE_RUNS, RAW_SHARDS, GROUPED_SHARDS, STATE_REPLAYS, STATE_STATUS):
        shard_paths.extend(sorted(path for path in directory.glob("*") if path.is_file()))
    manifest_path = NEW / "collection_shard_manifest.csv"
    atomic_csv(
        manifest_path,
        pd.DataFrame(
            [
                {
                    "relative_path": str(path.relative_to(ROOT)),
                    "size_bytes": path.stat().st_size,
                    "sha256": digest(path),
                }
                for path in shard_paths
            ]
        ),
    )
    scale = combined_grouped.groupby("scale", sort=True).agg(
        instances=("instance_id", "nunique"),
        states=("state_id", "nunique"),
        candidates=("target_set_id", "size"),
    )
    fold = combined_grouped.groupby("oof_fold", sort=True).agg(
        instances=("instance_id", "nunique"),
        states=("state_id", "nunique"),
        candidates=("target_set_id", "size"),
    )
    integrity = {
        "schema": "phase6n-expanded-data-integrity-v1",
        "status": "PASS",
        "implementation_commit": implementation["implementation_commit"],
        "worker_script_sha256": implementation["worker_script_sha256"],
        "source_runs": len(tasks),
        "new_states": int(new_grouped.state_id.nunique()),
        "combined_states": int(combined_grouped.state_id.nunique()),
        "instances": int(combined_grouped.instance_id.nunique()),
        "states_per_instance": 48,
        "new_candidate_rows": len(new_grouped),
        "combined_candidate_rows": len(combined_grouped),
        "new_raw_crn_rows": len(new_raw),
        "combined_raw_crn_rows": len(combined_raw),
        "historical_score_calls": 0,
        "all_candidate_feasible": True,
        "full_bank_scope": FULL_BANK_SCOPE,
        "canonical_fallback_exactly_one_per_state": True,
        "scale_composition": {
            str(index): {key: int(value) for key, value in row.items()}
            for index, row in scale.to_dict("index").items()
        },
        "fold_composition": {
            str(int(index)): {key: int(value) for key, value in row.items()}
            for index, row in fold.to_dict("index").items()
        },
        "repair_decoder_evaluations": int(
            sum(row["repair_decoder_evaluations"] for row in completed)
        ),
        "continuation_decoder_evaluations": int(
            sum(row["continuation_decoder_evaluations"] for row in completed)
        ),
        "state_elapsed_seconds": float(sum(row["elapsed_seconds"] for row in completed)),
        "artifacts": {
            str(path.relative_to(ROOT)): digest(path)
            for path in (
                new_grouped_path,
                new_raw_path,
                combined_grouped_path,
                combined_raw_path,
                manifest_path,
            )
        },
        "original_grouped_sha256": digest(ORIGINAL_GROUPED),
        "original_raw_sha256": digest(ORIGINAL_RAW),
        "r13_accessed": False,
        "r14_accessed": False,
        "gurobi_run": False,
    }
    integrity_path = OUT / "data_integrity.json"
    atomic_json(integrity_path, integrity)
    REPORT.write_text(
        f"""# Phase 6N 扩展数据生成报告

状态：**`PASS`**。

按 N1 冻结计划完成 {integrity['source_runs']} 条 H1-seeded frozen ALNS source trajectories 和 {integrity['new_states']} 个新增 R12 CAUR-FIT states。新增 candidate means 为 {integrity['new_candidate_rows']:,}，two-CRN rows 为 {integrity['new_raw_crn_rows']:,}；与原始 Phase 6L 数据合并后为 {integrity['combined_states']} states、{integrity['combined_candidate_rows']:,} candidates、{integrity['combined_raw_crn_rows']:,} raw rows。18 个 instances 每个恰好 48 states。

每个 state 均生成 frozen 24-rule full bank、按原规则 dedup，并保留 generator order；canonical `operator_related` fallback 每 state 恰好一个。所有 forced candidates 通过 deterministic decoder 与 `check_schedule`。新增 source sampler、candidate features 和 labels 的 historical scorer calls 均为 0。

原始 288 states 标记为 `ORIGINAL_PHASE6J_CAUR`，新增 576 states 标记为 `NEW_ALNS_EXPANSION`。后续 common-original 与 expanded OOF 指标必须分开报告。

累计 forced-repair decoder evaluations 为 {integrity['repair_decoder_evaluations']:,}，continuation decoder evaluations 为 {integrity['continuation_decoder_evaluations']:,}，state labeling wall time 合计 {integrity['state_elapsed_seconds']:.2f} 秒。R13/R14 未访问，未运行 Gurobi。

完整哈希与 composition 位于 `outputs/phase6n_candidate_conditioned_csg_v1/data/data_integrity.json` 和 `new_r12_collection/collection_shard_manifest.csv`。
""",
        encoding="utf-8",
    )
    return integrity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new-states", type=int)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    config, data_plan, preregistration_sha256, script_sha256 = validate_preregistration()
    implementation = freeze_implementation(preregistration_sha256, script_sha256)
    tasks = build_tasks(config, data_plan)
    phase6j_config = load_json(ROOT / config["locked_inputs"]["phase6j_config"])
    alns_config = read_alns_config(phase6j_config)
    process_started = time.perf_counter()
    completed = existing_statuses(implementation)
    sources = {
        (task["instance_id"], task["trajectory_seed"]): valid_source(task, implementation)
        for task in tasks
    }
    source_complete = sum(row is not None for row in sources.values())
    write_progress(
        implementation=implementation,
        process_started=process_started,
        source_complete=source_complete,
        completed=completed,
        active=None,
        status="RUNNING",
    )
    if args.summarize_only:
        integrity = summarize(tasks, implementation, completed)
        write_progress(
            implementation=implementation,
            process_started=process_started,
            source_complete=source_complete,
            completed=completed,
            active=None,
            status="COMPLETE",
        )
        print(json.dumps(integrity, indent=2, sort_keys=True), flush=True)
        return

    new_states = 0
    for task in tasks:
        key = (task["instance_id"], task["trajectory_seed"])
        source = sources[key]
        if source is None:
            source = run_source(task, config, data_plan, alns_config, implementation)
            sources[key] = source
            source_complete += 1
        for snapshot in source["snapshots"]:
            state_id = str(snapshot["state_id"])
            current = valid_state(state_id, implementation)
            if current is None:
                active = {
                    "instance_id": task["instance_id"],
                    "trajectory_seed": task["trajectory_seed"],
                    "state_id": state_id,
                }
                write_progress(
                    implementation=implementation,
                    process_started=process_started,
                    source_complete=source_complete,
                    completed=completed,
                    active=active,
                    status="RUNNING",
                )
                current = collect_state(
                    task, snapshot, config, data_plan, alns_config, implementation
                )
                completed.append(current)
                new_states += 1
                progress = write_progress(
                    implementation=implementation,
                    process_started=process_started,
                    source_complete=source_complete,
                    completed=completed,
                    active=None,
                    status="RUNNING",
                )
                print(
                    json.dumps(
                        {
                            "event": "state_complete",
                            "state_id": state_id,
                            "states_complete": progress["states_complete"],
                            "states_expected": 576,
                            "seconds": current["elapsed_seconds"],
                            "unique_candidates": current["unique_candidates"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                if args.max_new_states is not None and new_states >= args.max_new_states:
                    print("PHASE6N_DATA_GENERATION_EARLY_RETURN", flush=True)
                    return
    integrity = summarize(tasks, implementation, completed)
    write_progress(
        implementation=implementation,
        process_started=process_started,
        source_complete=source_complete,
        completed=completed,
        active=None,
        status="COMPLETE",
    )
    print(json.dumps(integrity, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
