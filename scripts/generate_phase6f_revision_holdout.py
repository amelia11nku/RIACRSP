#!/usr/bin/env python3
"""Generate and outcome-blindly freeze the 8,100 R06 pre-action states."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_clgri.data.loader import load_instance
from rcias_clgri.data.phase6c_io import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_parquet,
    remove_partial_files,
    sha256_file,
)
from rcias_clgri.search.alns import ALNSConfig, solve_alns
from rcias_clgri.search.counterfactual import stable_seed
from scripts.run_phase6c_reservoir import Phase6CReservoirObserver, alns_config


INSTANCE_ROOT = ROOT / "instances" / "controlled" / "RCIAS-CB1-TRAIN-R06"
INSTANCE_MANIFEST = INSTANCE_ROOT / "manifests" / "revision_instance_manifest.csv"
OUT = ROOT / "outputs" / "phase6f" / "revision_holdout"
RAW = OUT / "reservoir" / "raw"
RESERVOIR_MANIFEST = OUT / "reservoir" / "reservoir_shard_manifest.csv"
STATE_MANIFEST = OUT / "state_manifest.csv"
STATE_FREEZE = OUT / "state_freeze.json"
CONFIG_PATH = ROOT / "configs" / "phase6f_revision.json"
ENVIRONMENT_FREEZE = ROOT / "outputs" / "phase6f" / "environment" / "freeze_record.json"
STAGES = ("0-20%", "20-40%", "40-60%", "60-80%", "80-100%")
FORBIDDEN_OUTCOME_COLUMNS = {
    "counterfactual_makespan",
    "relative_improvement",
    "mean_relative_improvement",
    "rank_within_state",
    "regret_to_best",
    "improved",
}


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def digest(path: Path) -> str:
    return sha256_file(path)


def canonical_hash(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def trajectory_id(instance_id: str, run: int) -> str:
    return f"{instance_id}__tr{run:02d}"


def status_path(instance_id: str, run: int) -> Path:
    return RAW / trajectory_id(instance_id, run) / "status.json"


def valid_status(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        status = load_json(path)
        states = path.parent / "states.parquet"
        environment_hash = load_json(ENVIRONMENT_FREEZE)["freeze_hash"]
        if (
            status.get("status") != "COMPLETE"
            or status.get("phase6f_environment_freeze_hash") != environment_hash
            or not states.exists()
            or digest(states) != status.get("state_sha256")
        ):
            return None
        return status
    except (KeyError, json.JSONDecodeError, OSError):
        return None


def run_trajectory(task: tuple[dict, int, dict, str]) -> dict:
    record, run, config, environment_hash = task
    started = time.perf_counter()
    instance = load_instance(INSTANCE_ROOT / record["relative_path"])
    namespaces = config["seed_namespaces"]
    trajectory_seed = stable_seed(
        instance.instance_id, run, namespace=int(namespaces["trajectory"])
    )
    state_seed = stable_seed(
        instance.instance_id, run, namespace=int(namespaces["state_sampling"])
    )
    iteration_limit = int(config["revision_holdout"]["trajectory_iteration_limit"])
    metadata = {
        "instance_id": instance.instance_id,
        "instance_relative_path": record["relative_path"],
        "training_split": "REVISION_HOLDOUT",
        "scale": record["scale"],
        "CF_level": record["CF_level"],
        "RI_level": record["RI_level"],
        "TI_level": record["TI_level"],
        "replicate": "R06",
        "trajectory_run": run,
        "trajectory_seed": trajectory_seed,
        "state_sampling_seed": state_seed,
    }
    observer = Phase6CReservoirObserver(metadata, iteration_limit, state_seed)
    base = alns_config()
    deterministic = ALNSConfig(**{
        **{field: getattr(base, field) for field in ALNSConfig.__dataclass_fields__},
        "iteration_limit": iteration_limit,
    })
    result = solve_alns(instance, 10**9, trajectory_seed, deterministic, observer)
    rows = pd.DataFrame(observer.selected()).sort_values("state_id").reset_index(drop=True)
    if FORBIDDEN_OUTCOME_COLUMNS & set(rows):
        raise RuntimeError("outcome fields leaked into R06 state reservoir")
    shard = RAW / trajectory_id(instance.instance_id, run)
    state_file = shard / "states.parquet"
    atomic_write_parquet(rows, state_file)
    status = {
        "schema": "phase6f-r06-reservoir-shard-v1",
        "phase6f_environment_freeze_hash": environment_hash,
        "phase6f_config_sha256": digest(CONFIG_PATH),
        "shard_id": trajectory_id(instance.instance_id, run),
        "instance_id": instance.instance_id,
        "split": "REVISION_HOLDOUT",
        "trajectory_run": run,
        "trajectory_seed": trajectory_seed,
        "state_sampling_seed": state_seed,
        "state_count": len(rows),
        "state_sha256": digest(state_file),
        "iterations": result.iterations,
        "decoder_evaluations": result.decoder_evaluations,
        "runtime_seconds": result.runtime,
        "wall_runtime_seconds": time.perf_counter() - started,
        "iteration_limit": iteration_limit,
        "seen_by_stage_and_stratum": {
            "|".join(key): value for key, value in observer.seen.items()
        },
        "outcome_blind": True,
        "status": "COMPLETE",
    }
    atomic_write_json(status, shard / "status.json")
    return status


def write_reservoir_manifest() -> pd.DataFrame:
    rows = []
    for path in sorted(RAW.glob("*/status.json")):
        status = valid_status(path)
        if status is not None:
            rows.append(status)
    frame = pd.DataFrame(rows)
    atomic_write_csv(frame, RESERVOIR_MANIFEST)
    return frame


def priority(state_id: str, namespace: int) -> float:
    raw = stable_seed(state_id, "phase6f_final_state_select", namespace=namespace)
    uniform = (raw + 1) / (2**64 + 1)
    return -math.log(uniform)


def select_states(config: dict) -> pd.DataFrame:
    statuses = write_reservoir_manifest()
    expected_shards = int(config["revision_holdout"]["instance_count"]) * int(
        config["revision_holdout"]["trajectory_runs_per_instance"]
    )
    if len(statuses) != expected_shards:
        raise RuntimeError(f"expected {expected_shards} reservoir shards, found {len(statuses)}")
    frames = [pd.read_parquet(path) for path in sorted(RAW.glob("*/states.parquet"))]
    available = pd.concat(frames, ignore_index=True)
    if available.state_id.duplicated().any():
        raise RuntimeError("duplicate R06 reservoir state IDs")
    if FORBIDDEN_OUTCOME_COLUMNS & set(available):
        raise RuntimeError("outcome fields leaked into R06 state selection")

    per_cell = int(config["revision_holdout"]["states_per_structural_cell"])
    per_stage, remainder = divmod(per_cell, len(STAGES))
    if remainder:
        raise RuntimeError("R06 state quota must divide evenly across five search stages")
    namespace = int(config["seed_namespaces"]["state_sampling"])
    selected = []
    group_columns = ["scale", "CF_level", "RI_level", "TI_level"]
    for cell, part in available.groupby(group_columns, sort=True):
        cell_rows = []
        for stage in STAGES:
            candidates = part[part.search_stage == stage].copy()
            candidates["_priority"] = [priority(value, namespace) for value in candidates.state_id]
            chosen = candidates.sort_values(["_priority", "state_id"]).head(per_stage)
            if len(chosen) != per_stage:
                raise RuntimeError(f"R06 cell {cell} stage {stage} has only {len(chosen)} states")
            cell_rows.append(chosen.drop(columns="_priority"))
        selected.append(pd.concat(cell_rows, ignore_index=True))
    states = pd.concat(selected, ignore_index=True).sort_values("state_id").reset_index(drop=True)
    expected_states = int(config["revision_holdout"]["state_count"])
    cell_counts = states.groupby(group_columns).size()
    stage_counts = states.groupby([*group_columns, "search_stage"]).size()
    if len(states) != expected_states or states.state_id.nunique() != expected_states:
        raise RuntimeError("R06 state manifest is not exactly 8,100 unique states")
    if len(cell_counts) != 81 or set(cell_counts) != {per_cell}:
        raise RuntimeError("R06 structural-cell state counts are imbalanced")
    if len(stage_counts) != 405 or set(stage_counts) != {per_stage}:
        raise RuntimeError("R06 search-stage state counts are imbalanced")
    atomic_write_csv(states, STATE_MANIFEST)

    freeze = {
        "schema": "phase6f-r06-state-freeze-v1",
        "status": "FROZEN_OUTCOME_BLIND",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase6f_environment_freeze_hash": load_json(ENVIRONMENT_FREEZE)["freeze_hash"],
        "phase6f_config_sha256": digest(CONFIG_PATH),
        "instance_manifest_sha256": digest(INSTANCE_MANIFEST),
        "reservoir_manifest_sha256": digest(RESERVOIR_MANIFEST),
        "state_manifest_sha256": digest(STATE_MANIFEST),
        "state_count": len(states),
        "state_id_sha256": hashlib.sha256(
            "".join(f"{value}\n" for value in sorted(states.state_id)).encode()
        ).hexdigest(),
        "structural_cell_count": len(cell_counts),
        "states_per_structural_cell": per_cell,
        "states_per_search_stage_per_cell": per_stage,
        "outcome_columns_absent": not bool(FORBIDDEN_OUTCOME_COLUMNS & set(states)),
        "outcome_blind_selection": True,
        "labels_opened": False,
        "model_config_frozen": False,
    }
    freeze["freeze_hash"] = canonical_hash(freeze)
    atomic_write_json(freeze, STATE_FREEZE)
    print("PHASE6F_R06_STATE_SELECTION_COMPLETE states=8100 cells=81")
    return states


def verify(config: dict) -> None:
    remove_partial_files(RAW)
    statuses = write_reservoir_manifest()
    expected_shards = int(config["revision_holdout"]["instance_count"]) * int(
        config["revision_holdout"]["trajectory_runs_per_instance"]
    )
    checks = {
        "reservoir_shards_complete": len(statuses) == expected_shards,
        "state_manifest_exists": STATE_MANIFEST.exists(),
        "state_freeze_exists": STATE_FREEZE.exists(),
    }
    if STATE_MANIFEST.exists():
        states = pd.read_csv(STATE_MANIFEST)
        groups = states.groupby(["scale", "CF_level", "RI_level", "TI_level"]).size()
        checks.update({
            "exactly_8100_unique_states": len(states) == 8100 and states.state_id.nunique() == 8100,
            "exactly_100_states_per_cell": len(groups) == 81 and set(groups) == {100},
            "outcome_columns_absent": not bool(FORBIDDEN_OUTCOME_COLUMNS & set(states)),
        })
    if STATE_FREEZE.exists():
        freeze = load_json(STATE_FREEZE)
        checks.update({
            "state_hash_exact": STATE_MANIFEST.exists()
            and digest(STATE_MANIFEST) == freeze["state_manifest_sha256"],
            "labels_still_unopened": not bool(freeze["labels_opened"]),
        })
    audit = {
        "schema": "phase6f-r06-state-audit-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
    }
    atomic_write_json(audit, OUT / "state_audit.json")
    if audit["status"] != "PASS":
        raise RuntimeError({key: value for key, value in checks.items() if not value})
    print("PHASE6F_R06_STATES_VERIFIED")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--shards", nargs="*")
    parser.add_argument("--select-only", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if not ENVIRONMENT_FREEZE.exists():
        raise RuntimeError("Phase 6F environment must be frozen before R06 state generation")
    config = load_json(CONFIG_PATH)
    if args.verify_only:
        verify(config)
        return
    if args.select_only:
        select_states(config)
        return
    remove_partial_files(RAW)
    manifest = pd.read_csv(INSTANCE_MANIFEST)
    requested = set(args.shards or [])
    environment_hash = load_json(ENVIRONMENT_FREEZE)["freeze_hash"]
    tasks = []
    for record in manifest.to_dict("records"):
        for run in range(1, int(config["revision_holdout"]["trajectory_runs_per_instance"]) + 1):
            shard = trajectory_id(record["instance_id"], run)
            if requested and shard not in requested:
                continue
            if valid_status(status_path(record["instance_id"], run)) is None:
                tasks.append((record, run, config, environment_hash))
    print(
        f"PHASE6F_R06_RESERVOIR_START pending={len(tasks)} workers={args.workers}",
        flush=True,
    )
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_trajectory, task) for task in tasks]
        for index, future in enumerate(as_completed(futures), 1):
            status = future.result()
            write_reservoir_manifest()
            elapsed = time.perf_counter() - started
            print(json.dumps({
                "event": "r06_reservoir_shard",
                "completed": index,
                "submitted": len(futures),
                "shard_id": status["shard_id"],
                "states": status["state_count"],
                "shards_per_minute": 60 * index / max(elapsed, 1e-9),
            }), flush=True)
    write_reservoir_manifest()
    if not requested:
        select_states(config)


if __name__ == "__main__":
    main()
