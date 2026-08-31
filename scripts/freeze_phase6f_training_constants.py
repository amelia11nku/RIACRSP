#!/usr/bin/env python3
"""Derive and freeze Phase 6F loss scaling constants from TRAIN only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "outputs" / "phase6e" / "tensorization" / "cache" / "cache_manifest.csv"
OUTPUT = ROOT / "outputs" / "phase6f" / "audit" / "training_constants.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    import torch

    manifest = pd.read_csv(MANIFEST)
    train = manifest[
        manifest["status"].eq("COMPLETE")
        & manifest["training_split"].eq("TRAIN")
    ].sort_values("instance_id")
    if train.empty or set(train["training_split"]) != {"TRAIN"}:
        raise ValueError("loss constants must be derived from TRAIN only")

    utility_chunks: list[np.ndarray] = []
    gap_chunks: list[np.ndarray] = []
    state_count = 0
    action_count = 0
    for record in train.to_dict("records"):
        payload = torch.load(record["cache_path"], map_location="cpu", weights_only=False)
        samples = payload["samples"]
        for sample in samples:
            utility = sample.actions.utility.detach().float().numpy().astype(np.float64)
            utility_chunks.append(np.abs(utility))
            if len(utility) > 1:
                first, second = np.triu_indices(len(utility), k=1)
                gaps = np.abs(utility[first] - utility[second])
                nonzero = gaps[gaps > 0]
                if len(nonzero):
                    gap_chunks.append(nonzero)
            state_count += 1
            action_count += len(utility)

    absolute_utility = np.concatenate(utility_chunks)
    nonzero_gaps = np.concatenate(gap_chunks)
    utility_clip = float(np.quantile(absolute_utility, 0.99))
    rank_gap_scale = float(np.median(nonzero_gaps))
    positive = int(train["positive_count"].sum())
    positive_weight = max((action_count - positive) / max(positive, 1), 1.0)
    checks = {
        "source_split_exactly_train": set(train["training_split"]) == {"TRAIN"},
        "state_count_matches_manifest": state_count == int(train["state_count"].sum()),
        "action_count_matches_manifest": action_count == int(train["action_count"].sum()),
        "constants_finite_positive": bool(
            np.isfinite(utility_clip) and utility_clip > 0
            and np.isfinite(rank_gap_scale) and rank_gap_scale > 0
            and np.isfinite(positive_weight) and positive_weight >= 1
        ),
    }
    payload = {
        "schema": "phase6f-training-constants-v1",
        "status": "FROZEN_TRAIN_ONLY" if all(checks.values()) else "FAIL",
        "checks": checks,
        "source_split": "TRAIN",
        "train_shard_count": len(train),
        "train_state_count": state_count,
        "train_action_count": action_count,
        "positive_count": positive,
        "positive_weight": positive_weight,
        "utility_clip_absolute_train_p99": utility_clip,
        "rank_gap_scale_train_median_nonzero_pairwise": rank_gap_scale,
        "rank_weight_min": 0.25,
        "rank_weight_max": 4.0,
        "cache_manifest_sha256": sha256_file(MANIFEST),
        "train_internal_holdout_accessed": False,
        "revision_holdout_accessed": False,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if payload["status"] == "FAIL":
        raise SystemExit(1)
    print(json.dumps({
        "event": "phase6f_training_constants_frozen",
        "states": state_count,
        "actions": action_count,
        "utility_clip": utility_clip,
        "rank_gap_scale": rank_gap_scale,
    }))


if __name__ == "__main__":
    main()
