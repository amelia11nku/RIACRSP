#!/usr/bin/env python3
"""Freeze Phase 6I-MR head-training choices before the first optimizer step."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.prepare_phase6i_mr_training_data import (  # noqa: E402
    atomic_json,
    digest,
    load_json,
)


CONFIG_PATH = ROOT / "configs/phase6i_mr_live_utility_revision.json"
CONSTANTS_PATH = ROOT / "outputs/phase6i_mr/training_data/training_constants.json"
DATA_FREEZE_PATH = ROOT / "outputs/phase6i_mr/training_data/training_data_freeze.json"
CACHE_ROOT = ROOT / "outputs/phase6i_mr/embedding_cache"
CACHE_INTEGRITY_PATH = CACHE_ROOT / "embedding_cache_integrity.json"
CACHE_MANIFEST_PATH = CACHE_ROOT / "embedding_cache_manifest.csv"
OUTPUT = ROOT / "outputs/phase6i_mr/frozen/training_protocol.json"


def source_frame(source: str) -> pd.DataFrame:
    paths = sorted((CACHE_ROOT / source).glob("*.parquet"))
    if not paths:
        raise RuntimeError(f"missing frozen embedding source: {source}")
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def target_constants(frame: pd.DataFrame, column: str) -> dict[str, float | int]:
    absolute = frame[column].abs().to_numpy(dtype=float)
    gap_chunks = []
    positive_stds = []
    for _, group in frame.groupby("state_id", sort=False):
        values = group[column].to_numpy(dtype=float)
        first, second = np.triu_indices(len(values), k=1)
        gaps = np.abs(values[first] - values[second])
        if np.any(gaps > 0):
            gap_chunks.append(gaps[gaps > 0])
        standard_deviation = float(np.std(values))
        if standard_deviation > 0:
            positive_stds.append(standard_deviation)
    positive_count = int(frame[column].gt(0).sum())
    negative_count = int(len(frame) - positive_count)
    return {
        "absolute_p99_scale": float(np.quantile(absolute, 0.99)),
        "pair_gap_scale": float(np.median(np.concatenate(gap_chunks))),
        "listwise_temperature": max(float(np.median(positive_stds)), 0.001),
        "positive_class_weight": max(negative_count / max(positive_count, 1), 1.0),
        "positive_count": positive_count,
        "negative_count": negative_count,
    }


def main() -> None:
    config = load_json(CONFIG_PATH)
    constants = load_json(CONSTANTS_PATH)
    data_freeze = load_json(DATA_FREEZE_PATH)
    cache_integrity = load_json(CACHE_INTEGRITY_PATH)
    if not all([
        data_freeze.get("status") == "FROZEN_BEFORE_MODEL_FIT_AND_R10_ACCESS",
        data_freeze.get("r10_accessed") is False,
        data_freeze.get("r11_accessed") is False,
        cache_integrity.get("status") == "PASS",
        cache_integrity.get("r10_accessed") is False,
        cache_integrity.get("r11_accessed") is False,
    ]):
        raise RuntimeError("cannot freeze training protocol from failed or accessed inputs")
    r09 = source_frame("r09")
    continuation = source_frame("continuation")
    immediate = target_constants(r09, "decoded_immediate_utility")
    continuation_constants = target_constants(continuation, "continuation_value")
    if not np.isclose(
        immediate["absolute_p99_scale"],
        constants["immediate_utility_p99_absolute"],
        rtol=0,
        atol=1e-12,
    ):
        raise RuntimeError("immediate target p99 does not match the frozen R09 constant")
    immediate["listwise_temperature"] = constants["listwise_temperature"]
    immediate["positive_class_weight"] = constants["positive_weight"]
    code_paths = [
        "rcias_clgri/ni/phase6i_heads.py",
        "scripts/train_phase6i_mr_heads.py",
    ]
    objective = config["objective"]
    payload = {
        "schema": "phase6i-mr-training-protocol-v1.2",
        "status": "FROZEN_BEFORE_FIRST_MODEL_OPTIMIZER_STEP_AND_R10_R11_ACCESS",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "assumptions": {
            "per_instance_spearman": "mean finite within-state Spearman per instance, then arithmetic mean over instances",
            "state_balance": "each loss is averaged within state and then across states; the R09 design is exactly balanced by scale, CF, stage, and candidate role",
            "old_train_scope": "all 540 frozen Phase6F TRAIN-only states may enter every grouped-OOF fit because they are disjoint instances; only R09 fold instances are held out",
        },
        "model_candidates": {
            "U1": "128->128 GELU dropout(0.1)->1 immediate head",
            "U2": "19->32 GELU context encoder with 128-d sigmoid gate and tanh residual into the U1-shaped head",
            "U2_H_CONTINUATION": "a separate U2-shaped fixed-horizon continuation head; immediate and continuation targets never merge",
            "U3": "conditional config-defined last-block adaptation, activated only by the frozen grouped-OOF rule and trained only in the one-round hard-aggregation variant",
        },
        "parameter_counts": {"U1": 16641, "U2": 25729, "U2_H_CONTINUATION": 25729},
        "training_seeds": config["seeds"]["MODEL_TRAINING"],
        "grouped_oof_folds": config["hard_state_aggregation"]["grouped_oof_folds"],
        "context_columns": constants["context_output_columns"],
        "data_variants": ["R09_ONLY", "MIXED_OLD_NEW"],
        "source_loss_weights": {"earlier_train": 0.25, "r09_live": 0.75},
        "mixed_state_batch_order": ["earlier_train", "r09_live"],
        "targets": {
            "immediate": immediate,
            "continuation": continuation_constants,
        },
        "objective": {
            "pairwise_weight": objective["pairwise_margin_weight"],
            "listwise_weight": objective["listwise_weight"],
            "huber_weight": objective["huber_weight"],
            "positive_consistency_weight": objective["positive_consistency_weight"],
            "pair_gap_weight_clip": objective["pairwise_gap_weight_clip"],
            "pair_margin_clip": objective["pairwise_margin_clip"],
            "huber_delta": objective["huber_delta"],
            "listwise_prediction_temperature": 1.0,
            "cross_state_target": "raw relative utility divided by the R09-only absolute p99 and clipped to [-1,1]",
        },
        "training_budgets": {
            "immediate": {
                "optimizer": "AdamW",
                "epochs": 40,
                "state_batch_size": 64,
                "mixed_state_batch_counts": [16, 48],
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
                "gradient_norm_clip": 5.0,
                "dropout": 0.1,
                "early_stopping": False,
                "numeric_precision": "float32",
                "training_device": "CPU",
                "cpu_threads": 4,
            },
            "continuation": {
                "optimizer": "AdamW",
                "epochs": 160,
                "state_batch_size": 18,
                "mixed_state_batch_counts": [0, 18],
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
                "gradient_norm_clip": 5.0,
                "dropout": 0.1,
                "early_stopping": False,
                "numeric_precision": "float32",
                "training_device": "CPU",
                "cpu_threads": 4,
            },
            "full_base_refit": "same target-specific optimizer, epochs, initialization, and source ratios as grouped OOF",
            "one_round_hard_refit": {
                "epochs": 40,
                "state_batch_size": 80,
                "state_batch_counts": {"earlier_train": 16, "r09_live": 48, "r09_hard_state": 16},
                "source_loss_weights": {"earlier_train": 0.20, "r09_live": 0.60, "r09_hard_state": 0.20},
                "rounds": 1,
            },
            "conditional_u3": {
                "epochs": 20,
                "state_batch_size": 16,
                "learning_rates": {"last_relation_block": 0.00001, "action_projection": 0.00001, "utility_head": 0.001},
                "score_stability_weight": objective["u3_score_stability_weight"],
                "maximum_trainable_parameters": config["model_candidates"]["U3"]["maximum_trainable_parameters"],
            },
        },
        "hard_state_aggregation": config["hard_state_aggregation"],
        "u3_activation_rule": config["model_candidates"]["U3"]["activation"],
        "contrastive_activation_rule": config["contrastive_branch"],
        "u2h_gate": "apply frozen Phase6H probability/support eligibility, rank by continuation value, then require selected immediate utility >= its selected threshold and continuation value >= its selected threshold",
        "threshold_grids": {
            "probability": config["live_policy"]["probability_threshold_grid"],
            "immediate": config["live_policy"]["immediate_utility_threshold_grid"],
            "continuation": config["live_policy"]["continuation_threshold_grid"],
        },
        "family_artifact_rule": config["training_seed_robustness"]["family_artifact_rule"],
        "single_best_seed_selection": "FORBIDDEN",
        "r10_selection": config["r10_selection"],
        "negative_transfer_rule": config["training_data"]["negative_transfer_rule"],
        "code_hashes": {path: digest(ROOT / path) for path in code_paths},
        "input_hashes": {
            "config": digest(CONFIG_PATH),
            "training_constants": digest(CONSTANTS_PATH),
            "training_data_freeze": digest(DATA_FREEZE_PATH),
            "embedding_cache_integrity": digest(CACHE_INTEGRITY_PATH),
            "embedding_cache_manifest": digest(CACHE_MANIFEST_PATH),
        },
        "protected_split_ledger": {
            "R09": "labels accessed for fitting, normalization, OOF, calibration, and one hard-state round",
            "R10": "NOT_ACCESSED",
            "R11": "NOT_ACCESSED",
        },
        "r10_accessed": False,
        "r11_accessed": False,
    }
    atomic_json(payload, OUTPUT)
    print({
        "status": payload["status"],
        "output": str(OUTPUT),
        "sha256": digest(OUTPUT),
        "immediate_pair_gap_scale": immediate["pair_gap_scale"],
        "continuation_pair_gap_scale": continuation_constants["pair_gap_scale"],
    })


if __name__ == "__main__":
    main()
