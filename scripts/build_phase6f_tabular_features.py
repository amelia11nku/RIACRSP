#!/usr/bin/env python3
"""Build R06 pre-action target-set features for the frozen Phase 6C baseline."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "outputs/phase6f/revision_holdout/sealed_labels/revision_holdout"
OUTPUT = ROOT / "outputs/phase6f/evaluation/r06_target_set_preaction_features.parquet"
SUMMARY = ROOT / "outputs/phase6f/audit/r06_tabular_feature_audit.json"


def main() -> None:
    frames = []
    paths = sorted(DATASET_ROOT.glob("*/target_membership.parquet"))
    if len(paths) != 81:
        raise ValueError(f"expected 81 R06 membership shards, found {len(paths)}")
    for index, path in enumerate(paths, start=1):
        membership = pd.read_parquet(path)
        targeted = membership[membership["is_targeted"]]
        frames.append(targeted.groupby(["state_id", "target_set_id"]).agg(
            target_mean_criticality=("criticality_score", "mean"),
            target_max_criticality=("criticality_score", "max"),
            target_mean_slack=("operation_slack", "mean"),
            target_min_slack=("operation_slack", "min"),
            target_mean_W_delay=("W_waiting_or_delay_contribution", "mean"),
            target_mean_F_delay=("F_waiting_or_delay_contribution", "mean"),
            target_mean_island_load=("island_relative_load", "mean"),
            target_mean_reconfiguration=("local_reconfiguration_contribution", "mean"),
            target_mean_eligible_islands=("eligible_island_count", "mean"),
            target_mean_sync_delay=("synchronization_wait_contribution", "mean"),
            target_critical_fraction=("is_on_processing_critical_path", "mean"),
            target_resource_critical_fraction=("is_on_resource_critical_chain", "mean"),
            target_product_diversity=("product_id", "nunique"),
            target_island_diversity=("assigned_island", "nunique"),
        ).reset_index())
        print(json.dumps({
            "event": "tabular_feature_shard",
            "completed_shards": index,
            "total_shards": len(paths),
            "instance_id": path.parent.name,
            "feature_rows": len(frames[-1]),
        }), flush=True)
    features = pd.concat(frames, ignore_index=True)
    checks = {
        "exact_81_shards": len(paths) == 81,
        "exact_8100_states": features["state_id"].nunique() == 8_100,
        "exact_191416_actions": len(features) == 191_416,
        "unique_state_action_keys": not features[
            ["state_id", "target_set_id"]
        ].duplicated().any(),
        "no_missing_features": not features.isna().any().any(),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".parquet.partial")
    features.to_parquet(temporary, index=False)
    temporary.replace(OUTPUT)
    result = {
        "schema": "phase6f-r06-tabular-feature-audit-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": {key: bool(value) for key, value in checks.items()},
        "shard_count": len(paths),
        "state_count": int(features["state_id"].nunique()),
        "action_count": len(features),
        "feature_count": len(features.columns) - 2,
        "output_path": str(OUTPUT.resolve()),
    }
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    temporary_summary = SUMMARY.with_suffix(".json.partial")
    temporary_summary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary_summary.replace(SUMMARY)
    print(json.dumps({"event": "tabular_features_complete", **result}), flush=True)
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
