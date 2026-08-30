#!/usr/bin/env python3
"""Integrity, throughput, and label audit for a Phase 6C scale gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_clgri.data.phase6c_io import atomic_write_json


def load_shards(dataset: Path, name: str) -> pd.DataFrame:
    paths = sorted(dataset.glob(f"*/*/{name}"))
    if not paths:
        raise RuntimeError(f"no {name} shards under {dataset}")
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def mean_state_spearman(raw: pd.DataFrame, aggregate: pd.DataFrame) -> float:
    aggregate_rank = aggregate.set_index(["state_id", "target_set_id"])["rank_within_state"]
    values = []
    for (state_id, group), part in raw.groupby(["state_id", "repair_seed_group"]):
        if len(part) < 2:
            continue
        seed_rank = part.relative_improvement.rank(ascending=False, method="average")
        mean_rank = part.target_set_id.map(lambda target: aggregate_rank.loc[(state_id, target)])
        correlation = seed_rank.corr(mean_rank, method="spearman")
        if pd.notna(correlation):
            values.append(float(correlation))
    return float(np.mean(values))


def pairwise_stability(raw: pd.DataFrame, pairs: pd.DataFrame) -> float:
    indexed = raw.set_index(["state_id", "target_set_id", "repair_seed_group"])["relative_improvement"]
    agreements = []
    for pair in pairs.itertuples(index=False):
        aggregate_sign = int(pair.pairwise_preference)
        for group in range(3):
            reference = indexed.loc[(pair.state_id, pair.reference_target_set_id, group)]
            perturbed = indexed.loc[(pair.state_id, pair.perturbed_target_set_id, group)]
            difference = perturbed - reference
            seed_sign = 1 if difference > 0 else -1 if difference < 0 else 0
            agreements.append(seed_sign == aggregate_sign)
    return float(np.mean(agreements))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", choices=("gate_a", "gate_b"), required=True)
    args = parser.parse_args()
    gate = ROOT / "outputs/phase6c/gates" / args.gate
    dataset = gate / "dataset"
    states = load_shards(dataset, "states.parquet")
    raw = load_shards(dataset, "repair_seed_outcomes.parquet")
    aggregate = load_shards(dataset, "target_set_aggregates.parquet")
    pairs = load_shards(dataset, "operation_pairs.parquet")
    status = [json.loads(path.read_text()) for path in sorted(dataset.glob("*/*/status.json"))]
    group_size = raw.groupby(["state_id", "target_set_id"]).size()
    seed_count = raw.groupby(["state_id", "target_set_id"]).repair_seed.nunique()
    state_destroy_counts = aggregate.groupby("state_id").destroy_count.nunique()
    state_arm_counts = aggregate.groupby("state_id").size()
    seed0 = raw[raw.repair_seed_group == 0]
    aggregate_sign = aggregate.set_index(["state_id", "target_set_id"]).mean_relative_improvement.gt(0)
    seed_sign = seed0.set_index(["state_id", "target_set_id"]).relative_improvement.gt(0)
    audit = {
        "schema": "phase6c-scale-gate-audit-v1",
        "gate": args.gate,
        "integrity": {
            "distinct_state_count": int(states.state_id.nunique()),
            "state_ids_unique": not states.state_id.duplicated().any(),
            "all_target_sets_have_three_rows": bool((group_size == 3).all()),
            "all_target_sets_have_three_distinct_seeds": bool((seed_count == 3).all()),
            "fixed_destroy_count_within_state": bool((state_destroy_counts == 1).all()),
            "primary_repair_is_transport_aware": bool(raw.repair_operator.eq("transport_aware").all()),
            "requested_arm_count_is_24": bool(states.requested_arm_count.eq(24).all()),
            "deduplication_accounting_valid": bool((states.unique_arm_count + states.duplicate_arm_count == 24).all()),
            "rank_is_complete": bool(aggregate.rank_within_state.notna().all()),
        },
        "counts": {
            "shards": len(status),
            "states": len(states),
            "unique_target_sets": len(aggregate),
            "repair_seed_rows": len(raw),
            "conditional_pair_rows": len(pairs),
            "mean_unique_arms_per_state": float(state_arm_counts.mean()),
            "min_unique_arms_per_state": int(state_arm_counts.min()),
            "max_unique_arms_per_state": int(state_arm_counts.max()),
        },
        "labels": {
            "mean_aggregate_positive_arm_fraction": float(aggregate.mean_relative_improvement.gt(0).mean()),
            "robust_2_of_3_positive_arm_fraction": float(aggregate.positive_under_2_of_3.mean()),
            "states_with_positive_arm_fraction": float(aggregate.groupby("state_id").mean_relative_improvement.max().gt(0).mean()),
            "mean_improvement_probability": float(aggregate.improvement_probability.mean()),
            "single_seed_vs_aggregate_sign_agreement": float((seed_sign == aggregate_sign).mean()),
            "single_seed_vs_aggregate_rank_spearman": mean_state_spearman(raw, aggregate),
            "conditional_pair_preference_stability": pairwise_stability(raw, pairs),
        },
        "throughput": {
            "summed_shard_runtime_seconds": float(sum(item["runtime_seconds"] for item in status)),
            "repair_seed_rows_per_cpu_second": float(len(raw) / sum(item["runtime_seconds"] for item in status)),
            "dataset_bytes": sum(path.stat().st_size for path in dataset.rglob("*") if path.is_file()),
            "bytes_per_state": sum(path.stat().st_size for path in dataset.rglob("*") if path.is_file()) / len(states),
        },
        "phase6b_reference": {
            "positive_arm_fraction": 0.06308883179004864,
            "states_with_positive_arm_fraction": 0.23955555555555555,
            "single_seed_rank_spearman": 0.6436327213707315,
            "single_seed_sign_agreement": 0.9323642237028062,
        },
        "production_config_change_justified": False,
    }
    if not all(audit["integrity"].values()):
        raise RuntimeError(audit["integrity"])
    atomic_write_json(audit, gate / "gate_audit.json")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
