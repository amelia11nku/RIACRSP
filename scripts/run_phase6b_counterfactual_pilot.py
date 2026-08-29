#!/usr/bin/env python3
"""Evaluate Phase 6B counterfactual target arms, stability repeats, and swaps."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import random
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_clgri.analysis.phase6a import schedule_features
from rcias_clgri.data.loader import load_instance
from rcias_clgri.search.common import Candidate, decode_candidate
from rcias_clgri.search.counterfactual import (
    evaluate_counterfactual, generate_target_arms, stable_seed, swap_target,
)

TRAIN = ROOT / "instances/controlled/RCIAS-CB1-TRAIN"
RESERVOIR = ROOT / "outputs/phase6b/trajectory_reservoir/pilot_state_manifest.parquet"
OUT = ROOT / "outputs/phase6b/counterfactual"
MARGINAL = ROOT / "outputs/phase6b/marginal_target"
TRIALS = 8


def parse_candidate(value: str) -> Candidate:
    raw = json.loads(value)
    return Candidate(tuple(raw["operation_order"]), tuple(raw["island_assignment"]),
                     tuple(raw["w_assignment"]), tuple(raw["f_assignment"]))


def arm_seed(state_id, arm_id, group, namespace):
    return stable_seed(state_id, arm_id, "repair_group", group, namespace=int(namespace))


def evaluate_arms(instance, state, decoded, arms, repair_namespace, groups):
    rows = []
    for group in groups:
        group_rows = []
        for arm in arms:
            seed = arm_seed(state.state_id, arm.arm_id, group, repair_namespace)
            result = evaluate_counterfactual(
                instance, decoded.candidate, decoded, arm.destroyed_operations,
                "transport_aware", seed, TRIALS,
            )
            group_rows.append({
                "instance_id": state.instance_id, "training_split": state.training_split,
                "state_id": state.state_id, "scale": state.scale, "CF_level": state.CF_level,
                "RI_level": state.RI_level, "TI_level": state.TI_level,
                "search_stage": state.search_stage, "bottleneck_proxy": state.bottleneck_proxy,
                "current_makespan": state.current_makespan, "arm_id": arm.arm_id,
                "arm_family": arm.arm_family, "origin_destroy_operator": arm.origin_destroy_operator,
                "duplicate_origin_labels": json.dumps(arm.duplicate_origin_labels),
                "destroy_count": len(arm.destroyed_operations),
                "destroyed_operation_ids": json.dumps(arm.destroyed_operations),
                "repair_operator": "transport_aware", "repair_seed_group": group,
                "repair_seed": seed, "candidate_trials": TRIALS,
                "counterfactual_makespan": result.counterfactual.makespan,
                "absolute_improvement": result.absolute_improvement,
                "relative_improvement": result.relative_improvement, "improved": result.improved,
                "decoder_evaluations": result.decoder_evaluations, "runtime": result.runtime,
            })
        ordered = sorted(group_rows, key=lambda row: (-row["absolute_improvement"], row["arm_id"]))
        best = ordered[0]["absolute_improvement"]
        for rank, row in enumerate(ordered, 1):
            row.update({"rank_within_state": rank, "best_arm": rank == 1, "top3_arm": rank <= 3,
                        "regret_to_best_arm": best - row["absolute_improvement"],
                        "target_set_quality_percentile": 1.0 - (rank - 1) / max(1, len(ordered) - 1)})
        rows.extend(group_rows)
    return rows


def marginal_rows(instance, state, decoded, arms, primary_rows, repair_namespace):
    reference_arm = next(arm for arm in arms if arm.arm_id == "operator_related" or "operator_related" in arm.duplicate_origin_labels)
    reference_result = next(row for row in primary_rows if row["destroyed_operation_ids"] == json.dumps(reference_arm.destroyed_operations))
    reference = reference_arm.destroyed_operations
    outside = sorted(set(instance.operations) - set(reference))
    rng = random.Random(stable_seed(state.state_id, "marginal_swaps", namespace=664000000))
    removed = rng.sample(list(reference), min(6, len(reference)))
    added = rng.sample(outside, min(6, len(outside)))
    rows = []
    for index, (removed_in, added_out) in enumerate(zip(removed, added)):
        target = swap_target(reference, removed_in, added_out)
        seed = int(reference_result["repair_seed"])
        result = evaluate_counterfactual(instance, decoded.candidate, decoded, target, "transport_aware", seed, TRIALS)
        rows.append({
            "instance_id": state.instance_id, "state_id": state.state_id, "scale": state.scale,
            "CF_level": state.CF_level, "RI_level": state.RI_level, "TI_level": state.TI_level,
            "search_stage": state.search_stage, "bottleneck_proxy": state.bottleneck_proxy,
            "removed_in_operation": removed_in, "added_out_operation": added_out,
            "reference_destroyed_operation_ids": json.dumps(reference),
            "swap_destroyed_operation_ids": json.dumps(target),
            "repair_operator": "transport_aware", "repair_seed": seed, "candidate_trials": TRIALS,
            "reference_absolute_improvement": reference_result["absolute_improvement"],
            "swap_absolute_improvement": result.absolute_improvement,
            "marginal_swap_gain": result.absolute_improvement - reference_result["absolute_improvement"],
            "swap_improved": result.improved, "decoder_evaluations": result.decoder_evaluations,
            "runtime": result.runtime,
        })
    return rows


def run_instance(task):
    record, states = task
    instance = load_instance(TRAIN / record["relative_path"])
    arm_namespace = int(record["counterfactual_arm_seed_namespace"])
    repair_namespace = int(record["repair_seed_namespace"])
    arm_rows, target_rows, swap_rows = [], [], []
    marginal_ids = set(sorted(states.state_id, key=lambda value: stable_seed(value, "marginal_select"))[:15])
    for state in states.itertuples(index=False):
        candidate = parse_candidate(state.current_candidate); decoded = decode_candidate(instance, candidate)
        if decoded.makespan != state.current_makespan:
            raise RuntimeError(f"state reconstruction mismatch: {state.state_id}")
        count = max(2, round(instance.num_operations * .15))
        arms = generate_target_arms(instance, decoded, state.state_id, count, arm_namespace)
        stability = stable_seed(state.state_id, "stability_select") % 10 == 0
        groups = (0, 1, 2) if stability else (0,)
        evaluated = evaluate_arms(instance, state, decoded, arms, repair_namespace, groups)
        arm_rows.extend(evaluated)
        primary = [row for row in evaluated if row["repair_seed_group"] == 0]
        features = schedule_features(instance, decoded.schedule)
        for arm in arms:
            targeted = set(arm.destroyed_operations)
            for operation in instance.operations:
                target_rows.append({
                    "instance_id": state.instance_id, "state_id": state.state_id,
                    "arm_id": arm.arm_id, "operation_id": operation,
                    "is_targeted": operation in targeted, **features[operation],
                })
        if state.state_id in marginal_ids:
            swap_rows.extend(marginal_rows(instance, state, decoded, arms, primary, repair_namespace))
    shard = OUT / "shards" / record["instance_id"]; shard.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(arm_rows).to_parquet(shard / "arms.parquet", index=False)
    pd.DataFrame(target_rows).to_parquet(shard / "targets.parquet", index=False)
    pd.DataFrame(swap_rows).to_parquet(shard / "swaps.parquet", index=False)
    summary = {"instance_id": record["instance_id"], "states": len(states), "arms": len(arm_rows),
               "target_rows": len(target_rows), "swaps": len(swap_rows)}
    (shard / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def combine():
    shards = sorted((OUT / "shards").glob("*/summary.json"))
    if len(shards) != 81: raise RuntimeError(f"expected 81 shards, found {len(shards)}")
    arms = pd.concat([pd.read_parquet(path.parent / "arms.parquet") for path in shards], ignore_index=True)
    targets = pd.concat([pd.read_parquet(path.parent / "targets.parquet") for path in shards], ignore_index=True)
    swaps = pd.concat([pd.read_parquet(path.parent / "swaps.parquet") for path in shards], ignore_index=True)
    primary = arms[arms.repair_seed_group == 0]
    if primary.state_id.nunique() < 10000 or primary.groupby("state_id").size().min() < 2:
        raise RuntimeError("counterfactual arm coverage gate failed")
    OUT.mkdir(parents=True, exist_ok=True); MARGINAL.mkdir(parents=True, exist_ok=True)
    arms.to_parquet(OUT / "counterfactual_arm_results.parquet", index=False)
    targets.to_parquet(OUT / "counterfactual_target_rows.parquet", index=False)
    swaps.to_parquet(MARGINAL / "marginal_swap_results.parquet", index=False)
    pd.DataFrame([json.loads(path.read_text()) for path in shards]).to_csv(OUT / "counterfactual_shard_summary.csv", index=False)
    print(f"COUNTERFACTUAL_PILOT_COMPLETE states={primary.state_id.nunique()} primary_arms={len(primary)} stability_rows={len(arms)-len(primary)} swaps={len(swaps)}")


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--workers", type=int, default=1); parser.add_argument("--combine-only", action="store_true"); args = parser.parse_args()
    if args.combine_only: combine(); return
    states = pd.read_parquet(RESERVOIR); manifest = pd.read_csv(TRAIN / "manifests/train_instance_manifest.csv")
    by_id = manifest.set_index("instance_id")
    tasks = []
    for instance_id, part in states.groupby("instance_id"):
        if not (OUT / "shards" / instance_id / "summary.json").exists():
            tasks.append((by_id.loc[instance_id].to_dict() | {"instance_id": instance_id}, part))
    print(f"PHASE6B_COUNTERFACTUAL_START pending_instances={len(tasks)} workers={args.workers}", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_instance, task) for task in tasks]
        for index, future in enumerate(as_completed(futures), 1):
            summary = future.result(); print(f"[{index}/{len(tasks)}] {summary['instance_id']} arms={summary['arms']} swaps={summary['swaps']}", flush=True)
    combine()


if __name__ == "__main__": main()
