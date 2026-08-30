#!/usr/bin/env python3
"""Freeze Phase 6C state, action, split, aggregation, and shard identities."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_clgri.data.phase6c_io import atomic_write_json, sha256_file

OUT = ROOT / "outputs/phase6c"


def line_digest(lines) -> str:
    digest = hashlib.sha256()
    for line in lines:
        digest.update(str(line).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def main():
    integrity = json.loads((OUT / "audit/counterfactual_integrity.json").read_text())
    if not integrity["COUNTERFACTUAL_INTEGRITY_PASSED"]:
        raise RuntimeError("integrity audit must pass before freezing")
    states = pd.read_csv(OUT / "manifests/state_manifest.csv", usecols=["state_id", "training_split"])
    shards = pd.read_csv(OUT / "manifests/shard_manifest.csv")
    target_ids = set()
    for path in sorted((OUT / "dataset").glob("*/*/target_set_aggregates.parquet")):
        values = pd.read_parquet(path, columns=["target_set_id"]).target_set_id
        if target_ids.intersection(values):
            raise RuntimeError(f"duplicate target-set ID detected in {path}")
        target_ids.update(values)
    config = json.loads((ROOT / "configs/phase6c_counterfactual.json").read_text())
    record = {
        "schema": "phase6c-dataset-freeze-v1",
        "label_generation_version": config["label_generation_version"],
        "state_count": len(states),
        "state_id_sha256": line_digest(sorted(states.state_id)),
        "split_membership_sha256": line_digest(
            f"{row.state_id}|{row.training_split}" for row in states.sort_values("state_id").itertuples(index=False)
        ),
        "target_set_count": len(target_ids),
        "target_set_id_sha256": line_digest(sorted(target_ids)),
        "repair_seed_count": config["repair_seed_count"],
        "repair_operator": config["repair_operator"],
        "aggregation_rule": "mean_relative_improvement",
        "production_config_sha256": sha256_file(ROOT / "configs/phase6c_counterfactual.json"),
        "production_config_freeze_sha256": sha256_file(OUT / "environment/production_config_freeze.json"),
        "state_manifest_sha256": sha256_file(OUT / "manifests/state_manifest.csv"),
        "shard_manifest_sha256": sha256_file(OUT / "manifests/shard_manifest.csv"),
        "shard_collection_sha256": line_digest(
            f"{row.shard_id}|{row.shard_sha256}" for row in shards.sort_values("shard_id").itertuples(index=False)
        ),
        "shard_count": len(shards),
        "status": "FROZEN",
    }
    record["freeze_hash"] = hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    atomic_write_json(record, OUT / "audit/dataset_freeze_record.json")
    recommendation_path = OUT / "diagnostics/phase6d_recommendation.json"
    recommendation = json.loads(recommendation_path.read_text())
    recommendation["DATASET_FROZEN"] = True
    atomic_write_json(recommendation, recommendation_path)
    print("PHASE6C_DATASET_FROZEN", record["freeze_hash"])


if __name__ == "__main__":
    main()
