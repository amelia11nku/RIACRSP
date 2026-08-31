#!/usr/bin/env python3
"""Generate deterministic, resumable Phase 6C counterfactual dataset shards."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_clgri.analysis.phase6a import schedule_features
from rcias_clgri.data.loader import load_instance
from rcias_clgri.data.phase6c import reconstruct_state_from_instance
from rcias_clgri.data.phase6c_io import (
    atomic_write_csv, atomic_write_json, atomic_write_parquet, remove_partial_files, sha256_file,
)
from rcias_clgri.search.counterfactual import evaluate_counterfactual, stable_seed
from rcias_clgri.search.phase6c import (
    aggregate_repair_outcomes, generate_revised_target_arms, pairwise_preference, target_set_id,
)

TRAIN = ROOT / "instances/controlled/RCIAS-CB1-TRAIN"
PHASE6C = ROOT / "outputs/phase6c"
SPLIT_DIR = {
    "TRAIN": "train",
    "TRAIN_VALIDATION": "validation",
    "TRAIN_INTERNAL_HOLDOUT": "internal_holdout",
}
DATA_FILES = (
    "states.parquet",
    "repair_seed_outcomes.parquet",
    "target_set_aggregates.parquet",
    "target_membership.parquet",
    "operation_pairs.parquet",
)


def config() -> dict:
    return json.loads((ROOT / "configs/phase6c_counterfactual.json").read_text())


def collection_sha256(files: dict[str, str]) -> str:
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def repair_seed(state_id: str, target_id: str, group: int, namespace: int) -> int:
    return stable_seed(state_id, target_id, "repair_group", group, namespace=namespace)


def shard_directory(output_root: Path, split: str, instance_id: str) -> Path:
    return output_root / SPLIT_DIR.get(split, split.lower()) / instance_id


def valid_status(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        status = json.loads(path.read_text())
        files = status["file_sha256"]
        production_root = PHASE6C / "dataset"
        freeze_path = PHASE6C / "environment/production_config_freeze.json"
        freeze_hash = json.loads(freeze_path.read_text())["freeze_hash"] if freeze_path.exists() else None
        production_mismatch = production_root in path.parents and status.get("production_freeze_hash") != freeze_hash
        if status["status"] != "COMPLETE" or set(files) != set(DATA_FILES) or production_mismatch:
            return None
        if any(not (path.parent / name).exists() or sha256_file(path.parent / name) != digest for name, digest in files.items()):
            return None
        if collection_sha256(files) != status["shard_sha256"]:
            return None
        return status
    except (KeyError, json.JSONDecodeError):
        return None


def _raw_outcome(instance, state, reconstructed, arm, group: int, cfg: dict) -> dict:
    seed = repair_seed(
        state.state_id, arm.target_set_id, group, int(cfg["seed_namespaces"]["repair"]),
    )
    result = evaluate_counterfactual(
        instance, reconstructed.candidate, reconstructed.decoded, arm.destroyed_operations,
        cfg["repair_operator"], seed, int(cfg["candidate_trials"]),
    )
    return {
        "instance_id": state.instance_id,
        "training_split": state.training_split,
        "state_id": state.state_id,
        "target_set_id": arm.target_set_id,
        "repair_seed_group": group,
        "repair_seed": seed,
        "repair_operator": cfg["repair_operator"],
        "candidate_trials": int(cfg["candidate_trials"]),
        "destroy_count": len(arm.destroyed_operations),
        "counterfactual_makespan": result.counterfactual.makespan,
        "absolute_improvement": result.absolute_improvement,
        "relative_improvement": result.relative_improvement,
        "improved": result.improved,
        "decoder_evaluations": result.decoder_evaluations,
    }


def _rank_aggregates(rows: list[dict]) -> None:
    ordered = sorted(rows, key=lambda row: (-row["mean_relative_improvement"], row["target_set_id"]))
    best_mean = ordered[0]["mean_relative_improvement"]
    best_median = max(row["median_relative_improvement"] for row in ordered)
    denominator = max(1, len(ordered) - 1)
    for rank, row in enumerate(ordered, 1):
        row.update({
            "rank_within_state": rank,
            "rank_percentile": 1.0 - (rank - 1) / denominator,
            "top1": rank == 1,
            "top3": rank <= 3,
            "top5": rank <= 5,
            "regret_to_best": best_mean - row["mean_relative_improvement"],
            "robust_regret_to_best": best_median - row["median_relative_improvement"],
        })


def run_shard(task):
    if len(task) == 4:
        instance_record, states, output_root, cfg = task
        instance_root = TRAIN
    else:
        instance_record, states, output_root, cfg, instance_root = task
    started = time.perf_counter()
    instance = load_instance(Path(instance_root) / instance_record["relative_path"])
    state_rows, raw_rows, aggregate_rows, membership_rows, pair_rows = [], [], [], [], []
    arm_namespace = int(cfg["seed_namespaces"]["arm_generation"])
    for state in states.sort_values("state_id").itertuples(index=False):
        state_record = state._asdict()
        reconstructed = reconstruct_state_from_instance(instance, state_record)
        destroy_count = min(instance.num_operations, max(2, round(instance.num_operations * float(cfg["destroy_fraction"]))))
        generated = generate_revised_target_arms(
            instance, reconstructed.decoded, state.state_id, destroy_count, arm_namespace,
        )
        state_rows.append(state_record | {
            "requested_arm_count": generated.requested_arm_count,
            "unique_arm_count": generated.unique_arm_count,
            "duplicate_arm_count": generated.duplicate_arm_count,
        })
        by_target = {}
        state_aggregates = []
        for arm in generated.arms:
            outcomes = [_raw_outcome(instance, state, reconstructed, arm, group, cfg) for group in range(3)]
            raw_rows.extend(outcomes)
            aggregate = {
                "instance_id": state.instance_id,
                "training_split": state.training_split,
                "state_id": state.state_id,
                "scale": state.scale,
                "CF_level": state.CF_level,
                "RI_level": state.RI_level,
                "TI_level": state.TI_level,
                "search_stage": state.search_stage,
                "bottleneck_proxy": state.bottleneck_proxy,
                "current_makespan": state.current_makespan,
                "target_set_id": arm.target_set_id,
                "arm_family": arm.arm_family,
                "origin_destroy_operator": arm.origin_destroy_operator,
                "origin_rules": json.dumps(arm.origin_rules, separators=(",", ":")),
                "origin_families": json.dumps(arm.origin_families, separators=(",", ":")),
                "destroy_count": destroy_count,
                "destroyed_operation_ids": json.dumps(arm.destroyed_operations, separators=(",", ":")),
                "repair_operator": cfg["repair_operator"],
                "repair_seed_count": 3,
                **aggregate_repair_outcomes(outcomes),
            }
            state_aggregates.append(aggregate)
            by_target[arm.destroyed_operations] = aggregate
        _rank_aggregates(state_aggregates)
        aggregate_rows.extend(state_aggregates)

        features = schedule_features(instance, reconstructed.decoded.schedule)
        for arm in generated.arms:
            quality = by_target[arm.destroyed_operations]
            targeted = set(arm.destroyed_operations)
            for operation in instance.operations:
                membership_rows.append({
                    "instance_id": state.instance_id,
                    "training_split": state.training_split,
                    "state_id": state.state_id,
                    "target_set_id": arm.target_set_id,
                    "operation_id": operation,
                    "is_targeted": operation in targeted,
                    "target_set_mean_relative_improvement": quality["mean_relative_improvement"],
                    "target_set_improvement_probability": quality["improvement_probability"],
                    "target_set_rank": quality["rank_within_state"],
                    **features[operation],
                })

        reference = generated.canonical_related_target
        reference_aggregate = by_target[reference]
        for proposal in generated.proposals:
            if proposal.arm_family != "LOCAL_PERTURBATION":
                continue
            perturbed = by_target[proposal.destroyed_operations]
            difference = perturbed["mean_relative_improvement"] - reference_aggregate["mean_relative_improvement"]
            pair_rows.append({
                "instance_id": state.instance_id,
                "training_split": state.training_split,
                "state_id": state.state_id,
                "pair_rule": proposal.origin_rule,
                "reference_target_set_id": target_set_id(state.state_id, proposal.reference_operations),
                "perturbed_target_set_id": target_set_id(state.state_id, proposal.destroyed_operations),
                "removed_operations": json.dumps(proposal.removed_operations, separators=(",", ":")),
                "added_operations": json.dumps(proposal.added_operations, separators=(",", ":")),
                "destroy_count": destroy_count,
                "repair_operator": cfg["repair_operator"],
                "reference_mean_relative_improvement": reference_aggregate["mean_relative_improvement"],
                "perturbed_mean_relative_improvement": perturbed["mean_relative_improvement"],
                "mean_gain_difference": difference,
                "pairwise_preference": pairwise_preference(difference),
            })

    split = str(states.training_split.iloc[0])
    shard = shard_directory(Path(output_root), split, instance.instance_id)
    frames = {
        "states.parquet": pd.DataFrame(state_rows),
        "repair_seed_outcomes.parquet": pd.DataFrame(raw_rows),
        "target_set_aggregates.parquet": pd.DataFrame(aggregate_rows),
        "target_membership.parquet": pd.DataFrame(membership_rows),
        "operation_pairs.parquet": pd.DataFrame(pair_rows),
    }
    for name, frame in frames.items():
        atomic_write_parquet(frame, shard / name)
    file_hashes = {name: sha256_file(shard / name) for name in DATA_FILES}
    status = {
        "schema": "phase6c-dataset-shard-v1",
        "label_generation_version": cfg["label_generation_version"],
        "production_freeze_hash": cfg.get("production_freeze_hash"),
        "shard_id": instance.instance_id,
        "split": split,
        "structural_cells": ["|".join(map(str, states[["scale", "CF_level", "RI_level", "TI_level"]].iloc[0]))],
        "state_count": len(state_rows),
        "arm_count": len(aggregate_rows),
        "repair_seed_row_count": len(raw_rows),
        "membership_row_count": len(membership_rows),
        "operation_pair_row_count": len(pair_rows),
        "file_sha256": file_hashes,
        "shard_sha256": collection_sha256(file_hashes),
        "runtime_seconds": time.perf_counter() - started,
        "status": "COMPLETE",
    }
    atomic_write_json(status, shard / "status.json")
    return status


def status_paths(output_root: Path):
    return sorted(output_root.glob("*/*/status.json"))


def write_shard_manifest(output_root: Path, manifest_path: Path) -> pd.DataFrame:
    rows = []
    for path in status_paths(output_root):
        status = valid_status(path)
        if status is not None:
            rows.append({key: value for key, value in status.items() if key != "file_sha256"})
    frame = pd.DataFrame(rows).sort_values("shard_id").reset_index(drop=True) if rows else pd.DataFrame()
    atomic_write_csv(frame, manifest_path)
    return frame


def verify_dataset(state_manifest: Path, output_root: Path, shard_manifest: Path) -> None:
    remove_partial_files(output_root)
    states = pd.read_csv(state_manifest)
    expected = set(states.instance_id.unique())
    frame = write_shard_manifest(output_root, shard_manifest)
    complete = set(frame.shard_id) if len(frame) else set()
    if complete != expected:
        raise RuntimeError(f"dataset shards incomplete: complete={len(complete)} expected={len(expected)}")
    if int(frame.state_count.sum()) != len(states):
        raise RuntimeError("shard state counts do not match the state manifest")
    if not (frame.repair_seed_row_count == 3 * frame.arm_count).all():
        raise RuntimeError("three-seed row integrity failed")
    print(f"PHASE6C_DATASET_VERIFIED shards={len(frame)} states={int(frame.state_count.sum())} arms={int(frame.arm_count.sum())}")


def write_generation_manifest(state_manifest: Path, output_root: Path, shard_manifest: Path, cfg: dict) -> None:
    frame = pd.read_csv(shard_manifest)
    payload = {
        "schema": "phase6c-generation-manifest-v1",
        "label_generation_version": cfg["label_generation_version"],
        "production_config_sha256": sha256_file(ROOT / "configs/phase6c_counterfactual.json"),
        "state_manifest_sha256": sha256_file(state_manifest),
        "shard_manifest_sha256": sha256_file(shard_manifest),
        "state_count": int(frame.state_count.sum()),
        "arm_count": int(frame.arm_count.sum()),
        "repair_seed_row_count": int(frame.repair_seed_row_count.sum()),
        "shard_count": len(frame),
        "output_root": str(output_root.relative_to(ROOT)) if output_root.is_relative_to(ROOT) else str(output_root),
        "status": "COMPLETE",
    }
    target = PHASE6C / "manifests/generation_manifest.json" if output_root == PHASE6C / "dataset" else output_root.parent / "generation_manifest.json"
    atomic_write_json(payload, target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-manifest", type=Path, default=PHASE6C / "manifests/state_manifest.csv")
    parser.add_argument("--output-root", type=Path, default=PHASE6C / "dataset")
    parser.add_argument("--shard-manifest", type=Path)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--resume", action="store_true", help="explicitly request the default skip-complete behavior")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--shards", nargs="*")
    args = parser.parse_args()
    args.state_manifest = args.state_manifest.resolve()
    args.output_root = args.output_root.resolve()
    if args.shard_manifest is None:
        args.shard_manifest = PHASE6C / "manifests/shard_manifest.csv" if args.output_root == PHASE6C / "dataset" else args.output_root.parent / "shard_manifest.csv"
    else:
        args.shard_manifest = args.shard_manifest.resolve()
    cfg = config()
    freeze_path = PHASE6C / "environment/production_config_freeze.json"
    if args.output_root == PHASE6C / "dataset":
        if not freeze_path.exists():
            raise RuntimeError("production configuration must be frozen before dataset generation")
        cfg["production_freeze_hash"] = json.loads(freeze_path.read_text())["freeze_hash"]
    if args.verify_only:
        verify_dataset(args.state_manifest, args.output_root, args.shard_manifest)
        return
    remove_partial_files(args.output_root)
    states = pd.read_csv(args.state_manifest)
    train_manifest = pd.read_csv(TRAIN / "manifests/train_instance_manifest.csv").set_index("instance_id")
    requested = set(args.shards or [])
    tasks = []
    for instance_id, part in states.groupby("instance_id"):
        if requested and instance_id not in requested:
            continue
        record = train_manifest.loc[instance_id].to_dict() | {"instance_id": instance_id}
        status_path = shard_directory(args.output_root, str(part.training_split.iloc[0]), instance_id) / "status.json"
        if valid_status(status_path) is None:
            tasks.append((record, part, args.output_root, cfg))
    print(f"PHASE6C_DATASET_START pending={len(tasks)} workers={args.workers} output={args.output_root}", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_shard, task) for task in tasks]
        for index, future in enumerate(as_completed(futures), 1):
            status = future.result()
            write_shard_manifest(args.output_root, args.shard_manifest)
            print(f"[{index}/{len(futures)}] {status['shard_id']} states={status['state_count']} arms={status['arm_count']} runtime={status['runtime_seconds']:.1f}s", flush=True)
    write_shard_manifest(args.output_root, args.shard_manifest)
    if not requested:
        verify_dataset(args.state_manifest, args.output_root, args.shard_manifest)
        write_generation_manifest(args.state_manifest, args.output_root, args.shard_manifest, cfg)


if __name__ == "__main__":
    main()
