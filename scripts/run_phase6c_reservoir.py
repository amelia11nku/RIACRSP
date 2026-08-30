#!/usr/bin/env python3
"""Generate and outcome-blindly select the 100,000 Phase 6C search states."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path
import random
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_clgri.analysis.phase6a import bottleneck_proxy
from rcias_clgri.data.loader import load_instance
from rcias_clgri.data.phase6c import candidate_sha256, candidate_to_json
from rcias_clgri.data.phase6c_io import (
    atomic_write_csv, atomic_write_json, atomic_write_parquet, remove_partial_files, sha256_file,
)
from rcias_clgri.search.alns import ALNSConfig, solve_alns
from rcias_clgri.search.counterfactual import stable_seed

TRAIN = ROOT / "instances/controlled/RCIAS-CB1-TRAIN"
OUT = ROOT / "outputs/phase6c"
RAW = OUT / "reservoir/raw"
STAGES = ("0-20%", "20-40%", "40-60%", "60-80%", "80-100%")
RARE = ("F_LOGISTICS", "CROSS_RESOURCE_SYNCHRONIZATION")


class Phase6CReservoirObserver:
    """Bounded, outcome-blind reservoir sampling of pre-action states."""

    def __init__(self, metadata, iteration_limit: int, state_seed: int):
        self.metadata = dict(metadata)
        self.iteration_limit = iteration_limit
        self.rng = random.Random(state_seed)
        self.seen = {(stage, stratum): 0 for stage in STAGES for stratum in ("ALL", *RARE)}
        self.rows = {(stage, stratum): [] for stage in STAGES for stratum in ("ALL", *RARE)}

    def _insert(self, key, row, capacity):
        self.seen[key] += 1
        bucket = self.rows[key]
        if len(bucket) < capacity:
            bucket.append(row)
        else:
            replacement = self.rng.randrange(self.seen[key])
            if replacement < capacity:
                bucket[replacement] = row

    def __call__(self, event):
        progress = min(int(event["iteration"]) / self.iteration_limit, .999999)
        stage = STAGES[min(4, int(progress * 5))]
        current = event["current_before"]
        proxy = bottleneck_proxy(current.schedule)
        state_id = f"{self.metadata['instance_id']}__tr{self.metadata['trajectory_run']:02d}__it{int(event['iteration']):07d}"
        serialized = candidate_to_json(current.candidate)
        row = {
            **self.metadata,
            "state_id": state_id,
            "search_stage": stage,
            "search_progress": progress,
            "iteration": int(event["iteration"]),
            "current_makespan": current.makespan,
            "current_candidate": serialized,
            "candidate_sha256": candidate_sha256(serialized),
            "bottleneck_proxy": proxy,
            "trajectory_destroy_operator": event["destroy_operator"],
            "trajectory_repair_operator": event["repair_operator"],
        }
        self._insert((stage, "ALL"), row, 50)
        if proxy in RARE:
            self._insert((stage, proxy), row, 20)

    def selected(self):
        unique = {}
        for key in self.rows:
            for row in self.rows[key]:
                unique[row["state_id"]] = row
        return list(unique.values())


def alns_config() -> ALNSConfig:
    raw = json.loads((ROOT / "configs/phase5c_alns.json").read_text())
    return ALNSConfig(**{key: value for key, value in raw.items() if key in ALNSConfig.__dataclass_fields__})


def phase6c_config():
    return json.loads((ROOT / "configs/phase6c_counterfactual.json").read_text())


def trajectory_id(instance_id: str, run: int) -> str:
    return f"{instance_id}__tr{run:02d}"


def trajectory_status_path(instance_id: str, run: int) -> Path:
    return RAW / trajectory_id(instance_id, run) / "status.json"


def run_trajectory(task):
    record, run, namespaces = task
    instance = load_instance(TRAIN / record["relative_path"])
    trajectory_seed = stable_seed(instance.instance_id, run, namespace=int(namespaces["trajectory"]))
    state_seed = stable_seed(instance.instance_id, run, namespace=int(namespaces["state_sampling"]))
    iteration_limit = int(namespaces["trajectory_iteration_limit"])
    metadata = {
        "instance_id": instance.instance_id,
        "instance_relative_path": record["relative_path"],
        "training_split": record["training_split"],
        "scale": record["scale"],
        "CF_level": record["CF_level"],
        "RI_level": record["RI_level"],
        "TI_level": record["TI_level"],
        "replicate": record["replicate"],
        "trajectory_run": run,
        "trajectory_seed": trajectory_seed,
        "state_sampling_seed": state_seed,
    }
    observer = Phase6CReservoirObserver(metadata, iteration_limit, state_seed)
    base_config = alns_config()
    deterministic_config = ALNSConfig(**{
        **{field: getattr(base_config, field) for field in ALNSConfig.__dataclass_fields__},
        "iteration_limit": iteration_limit,
    })
    result = solve_alns(instance, 10**9, trajectory_seed, deterministic_config, observer)
    rows = pd.DataFrame(observer.selected()).sort_values("state_id").reset_index(drop=True)
    shard = RAW / trajectory_id(instance.instance_id, run)
    state_path = shard / "states.parquet"
    atomic_write_parquet(rows, state_path)
    status = {
        "schema": "phase6c-reservoir-shard-v1",
        "production_freeze_hash": namespaces["production_freeze_hash"],
        "shard_id": trajectory_id(instance.instance_id, run),
        "instance_id": instance.instance_id,
        "split": record["training_split"],
        "trajectory_run": run,
        "trajectory_seed": trajectory_seed,
        "state_sampling_seed": state_seed,
        "state_count": len(rows),
        "state_sha256": sha256_file(state_path),
        "iterations": result.iterations,
        "decoder_evaluations": result.decoder_evaluations,
        "runtime": result.runtime,
        "iteration_limit": iteration_limit,
        "seen_by_stage_and_stratum": {"|".join(key): value for key, value in observer.seen.items()},
        "status": "COMPLETE",
    }
    atomic_write_json(status, shard / "status.json")
    return status


def read_valid_status(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        status = json.loads(path.read_text())
        states = path.parent / "states.parquet"
        freeze_path = OUT / "environment/production_config_freeze.json"
        freeze_hash = json.loads(freeze_path.read_text())["freeze_hash"] if freeze_path.exists() else None
        if (status["status"] != "COMPLETE" or status.get("production_freeze_hash") != freeze_hash or
                not states.exists() or sha256_file(states) != status["state_sha256"]):
            return None
        return status
    except (KeyError, json.JSONDecodeError):
        return None


def write_reservoir_manifest() -> pd.DataFrame:
    records = []
    for status_path in sorted(RAW.glob("*/status.json")):
        status = read_valid_status(status_path)
        if status is not None:
            records.append(status)
    frame = pd.DataFrame(records)
    atomic_write_csv(frame, OUT / "manifests/reservoir_shard_manifest.csv")
    return frame


def _weighted_priority(row, namespace: int) -> float:
    raw = stable_seed(row.state_id, "final_state_select", namespace=namespace)
    uniform = (raw + 1) / (2**64 + 1)
    weight = 3.0 if row.bottleneck_proxy in RARE else 1.0
    return -math.log(uniform) / weight


def _cell_quota(total: int, cells: list[tuple[str, str, str, str]]) -> dict:
    base, remainder = divmod(total, len(cells))
    return {cell: base + (index < remainder) for index, cell in enumerate(cells)}


def _select_cell(part: pd.DataFrame, quota: int, namespace: int) -> pd.DataFrame:
    selected = []
    stage_base, stage_remainder = divmod(quota, len(STAGES))
    used = set()
    for index, stage in enumerate(STAGES):
        stage_quota = stage_base + (index < stage_remainder)
        available = part[part.search_stage == stage].copy()
        available["_priority"] = [_weighted_priority(row, namespace) for row in available.itertuples(index=False)]
        chosen = available.sort_values(["_priority", "state_id"]).head(stage_quota)
        selected.append(chosen.drop(columns="_priority"))
        used.update(chosen.state_id)
    result = pd.concat(selected, ignore_index=True) if selected else part.iloc[:0]
    if len(result) < quota:
        remaining = part[~part.state_id.isin(used)].copy()
        remaining["_priority"] = [_weighted_priority(row, namespace) for row in remaining.itertuples(index=False)]
        result = pd.concat([result, remaining.sort_values(["_priority", "state_id"]).head(quota - len(result)).drop(columns="_priority")], ignore_index=True)
    if len(result) != quota:
        raise RuntimeError(f"structural cell has only {len(result)} selectable states for quota {quota}")
    return result


def select_final_states(config: dict) -> pd.DataFrame:
    statuses = write_reservoir_manifest()
    expected = 405 * int(config["trajectory_runs_per_instance"])
    if len(statuses) != expected:
        raise RuntimeError(f"expected {expected} complete reservoir shards, found {len(statuses)}")
    frames = [pd.read_parquet(path) for path in sorted(RAW.glob("*/states.parquet"))]
    available = pd.concat(frames, ignore_index=True)
    if available.state_id.duplicated().any():
        raise RuntimeError("duplicate state IDs in the raw reservoir")
    cells = sorted(map(tuple, available[["scale", "CF_level", "RI_level", "TI_level"]].drop_duplicates().to_numpy()))
    if len(cells) != 81:
        raise RuntimeError(f"expected 81 structural cells, found {len(cells)}")
    selected = []
    namespace = int(config["seed_namespaces"]["state_sampling"])
    for split, target in config["state_targets"].items():
        quotas = _cell_quota(int(target), cells)
        split_rows = available[available.training_split == split]
        for cell, quota in quotas.items():
            mask = True
            for column, value in zip(("scale", "CF_level", "RI_level", "TI_level"), cell):
                mask &= split_rows[column] == value
            selected.append(_select_cell(split_rows[mask], quota, namespace))
    states = pd.concat(selected, ignore_index=True).sort_values(["training_split", "state_id"]).reset_index(drop=True)
    if len(states) != 100000 or states.state_id.nunique() != 100000:
        raise RuntimeError("final reservoir must contain exactly 100,000 distinct states")
    counts = states.training_split.value_counts().to_dict()
    if counts != {key: int(value) for key, value in config["state_targets"].items()}:
        raise RuntimeError(f"split count mismatch: {counts}")
    atomic_write_csv(states, OUT / "manifests/state_manifest.csv")
    split_manifest = states.groupby("training_split").agg(
        state_count=("state_id", "size"), instance_count=("instance_id", "nunique"),
        structural_cell_count=("instance_id", lambda _: 0),
    ).reset_index()
    cell_counts = {
        split: len(frame[["scale", "CF_level", "RI_level", "TI_level"]].drop_duplicates())
        for split, frame in states.groupby("training_split")
    }
    split_manifest["structural_cell_count"] = split_manifest.training_split.map(cell_counts)
    atomic_write_csv(split_manifest, OUT / "manifests/dataset_split_manifest.csv")
    print("PHASE6C_STATE_SELECTION_COMPLETE", counts)
    return states


def verify(config: dict) -> None:
    remove_partial_files(RAW)
    manifest = write_reservoir_manifest()
    expected = 405 * int(config["trajectory_runs_per_instance"])
    if len(manifest) != expected:
        raise RuntimeError(f"reservoir incomplete: {len(manifest)}/{expected}")
    state_path = OUT / "manifests/state_manifest.csv"
    if state_path.exists():
        states = pd.read_csv(state_path)
        if len(states) != 100000 or states.state_id.nunique() != 100000:
            raise RuntimeError("invalid selected state manifest")
    print(f"PHASE6C_RESERVOIR_VERIFIED shards={len(manifest)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--select-only", action="store_true")
    parser.add_argument("--shards", nargs="*")
    args = parser.parse_args()
    config = phase6c_config()
    freeze_path = OUT / "environment/production_config_freeze.json"
    if not freeze_path.exists():
        raise RuntimeError("production configuration must be frozen before reservoir generation")
    production_freeze_hash = json.loads(freeze_path.read_text())["freeze_hash"]
    if args.verify_only:
        verify(config)
        return
    if args.select_only:
        select_final_states(config)
        return
    remove_partial_files(RAW)
    manifest = pd.read_csv(TRAIN / "manifests/train_instance_manifest.csv")
    tasks = []
    requested = set(args.shards or [])
    for record in manifest.to_dict("records"):
        for run in range(1, int(config["trajectory_runs_per_instance"]) + 1):
            shard_id = trajectory_id(record["instance_id"], run)
            if requested and shard_id not in requested:
                continue
            if read_valid_status(trajectory_status_path(record["instance_id"], run)) is None:
                tasks.append((record, run, config["seed_namespaces"] | {
                    "trajectory_iteration_limit": config["trajectory_iteration_limit"],
                    "production_freeze_hash": production_freeze_hash,
                }))
    print(f"PHASE6C_RESERVOIR_START pending={len(tasks)} workers={args.workers}", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_trajectory, task) for task in tasks]
        for index, future in enumerate(as_completed(futures), 1):
            status = future.result()
            if index == 1 or index % 10 == 0 or index == len(futures):
                print(f"[{index}/{len(futures)}] {status['shard_id']} states={status['state_count']}", flush=True)
            write_reservoir_manifest()
    write_reservoir_manifest()
    if not requested:
        select_final_states(config)


if __name__ == "__main__":
    main()
