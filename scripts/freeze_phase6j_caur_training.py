#!/usr/bin/env python3
"""Freeze the Phase 6J R12 training boundary before any optimizer step."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_clgri.analysis.phase6j_caur import grouped_oof_fold  # noqa: E402
from rcias_clgri.ni.encoder import NIModelConfig  # noqa: E402
from rcias_clgri.ni.phase6j_caur_model import CAURModel  # noqa: E402
from rcias_clgri.ni.scorer import CSGTargetSetScorer  # noqa: E402
from rcias_clgri.ni.tensorize import CSGTensorizer  # noqa: E402
from scripts.run_phase6j_caur_pilot import atomic_json, digest, load_json  # noqa: E402
from scripts.train_phase6j_caur import (  # noqa: E402
    CATEGORICAL_COLUMNS,
    FAMILIES,
    NUMERIC_COLUMNS,
    fit_feature_transform,
)


CONFIG_PATH = ROOT / "configs/phase6j_caur.json"
COLLECTION = ROOT / "outputs/phase6j_caur/r12_collection"
SOURCE_PATH = COLLECTION / "r12_grouped_labels.parquet"
CACHE = ROOT / "outputs/phase6j_caur/tensor_cache"
HORIZON_FREEZE = ROOT / "outputs/phase6j_caur/frozen/r12_horizon_freeze.json"
OUTPUT = ROOT / "outputs/phase6j_caur/frozen/r12_training_protocol.json"
R13_LEDGER = ROOT / "outputs/phase6j_caur/r13_selection/access_ledger.json"
R14_LEDGER = ROOT / "outputs/phase6j_caur/r14_holdout/access_ledger.json"

CODE_PATHS = (
    "rcias_clgri/analysis/phase6j_caur.py",
    "rcias_clgri/data/phase6j_access.py",
    "rcias_clgri/ni/phase6j_caur_model.py",
    "scripts/build_phase6j_caur_tensor_cache.py",
    "scripts/freeze_phase6j_caur_training.py",
    "scripts/train_phase6j_caur.py",
)


def load_base_model(path: Path) -> CSGTargetSetScorer:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = CSGTargetSetScorer(
        CSGTensorizer(), NIModelConfig(**checkpoint["model_config"])
    )
    model.load_state_dict(checkpoint["model_state"])
    return model


def positive_pair_gap_median(frame: pd.DataFrame) -> tuple[float, int]:
    gaps = []
    for _, group in frame.groupby("state_id", sort=True):
        values = group.continuation_advantage_mean.to_numpy(dtype=float)
        absolute = np.abs(values[:, None] - values[None, :])
        gaps.append(absolute[absolute > 1e-12])
    combined = np.concatenate(gaps)
    if not len(combined) or not np.isfinite(combined).all():
        raise RuntimeError("R12 contains no finite nonzero continuation gaps")
    return float(np.median(combined)), int(len(combined))


def validate_existing(protocol: dict) -> None:
    checks = {
        "schema": protocol.get("schema")
        == "phase6j-caur-r12-training-protocol-v1",
        "status": protocol.get("status")
        == "FROZEN_BEFORE_FIRST_OPTIMIZER_STEP",
        "config": protocol.get("config_sha256") == digest(CONFIG_PATH),
        "r12_grouped_labels": protocol.get("input_hashes", {}).get(
            "r12_grouped_labels"
        )
        == digest(SOURCE_PATH),
        "tensor_cache_integrity": protocol.get("input_hashes", {}).get(
            "tensor_cache_integrity"
        )
        == digest(CACHE / "tensor_cache_integrity.json"),
        "r13_locked": protocol.get("r13_accessed") is False
        and not R13_LEDGER.exists(),
        "r14_locked": protocol.get("r14_accessed") is False
        and not R14_LEDGER.exists(),
    }
    for relative, expected in protocol.get("code_hashes", {}).items():
        checks[f"code:{relative}"] = digest(ROOT / relative) == expected
    if not all(checks.values()):
        raise RuntimeError(f"existing R12 training freeze is invalid: {checks}")


def main() -> None:
    if OUTPUT.exists():
        existing = load_json(OUTPUT)
        validate_existing(existing)
        print(json.dumps(existing, indent=2, sort_keys=True))
        return
    if R13_LEDGER.exists() or R14_LEDGER.exists():
        raise RuntimeError("cannot freeze R12 training after R13/R14 access")

    config = load_json(CONFIG_PATH)
    collection = load_json(COLLECTION / "collection_integrity.json")
    tensor = load_json(CACHE / "tensor_cache_integrity.json")
    horizon = load_json(HORIZON_FREEZE)
    source_sha256 = digest(SOURCE_PATH)
    checks = {
        "collection_pass": collection.get("status") == "PASS"
        and all(collection.get("checks", {}).values()),
        "collection_complete": collection.get("states") == 288
        and collection.get("grouped_rows") == 6809,
        "tensor_cache_pass": tensor.get("status") == "PASS"
        and all(tensor.get("checks", {}).values()),
        "tensor_cache_complete": tensor.get("states") == 288
        and tensor.get("actions") == 6809,
        "source_hash_chain": source_sha256
        == collection.get("r12_grouped_labels_sha256")
        == tensor.get("source_sha256"),
        "horizon_hash_chain": digest(HORIZON_FREEZE)
        == collection.get("r12_horizon_freeze_sha256"),
        "selected_horizon": horizon.get("selected_horizon")
        == collection.get("selected_horizon")
        == 4,
        "r13_locked": not R13_LEDGER.exists(),
        "r14_locked": not R14_LEDGER.exists(),
    }
    if not all(checks.values()):
        raise RuntimeError(f"R12 training freeze prerequisites failed: {checks}")

    frame = pd.read_parquet(SOURCE_PATH)
    frame["oof_fold"] = [
        grouped_oof_fold(str(scale), str(cf))
        for scale, cf in zip(frame.scale, frame.CF_level)
    ]
    expected_folds = {0, 1, 2}
    if set(frame.oof_fold.unique()) != expected_folds:
        raise RuntimeError("R12 grouped folds are incomplete")
    fold_rows = []
    fold_transforms = []
    for held_fold in sorted(expected_folds):
        train = frame[frame.oof_fold.ne(held_fold)]
        transform = fit_feature_transform(train)
        fold_transforms.append(transform)
        fold_rows.append({
            "held_fold": held_fold,
            "training_instances": int(train.instance_id.nunique()),
            "training_states": int(train.state_id.nunique()),
            "training_actions": int(len(train)),
            "categorical_vocabulary_sizes": {
                column: len(transform.vocabularies[column])
                for column in CATEGORICAL_COLUMNS
            },
        })
    vocabulary_sizes = {
        tuple(len(transform.vocabularies[column]) for column in CATEGORICAL_COLUMNS)
        for transform in fold_transforms
    }
    if vocabulary_sizes != {(24, 7, 5)}:
        raise RuntimeError(f"fold vocabularies are not stable: {vocabulary_sizes}")
    categorical_sizes = tuple(size + 1 for size in next(iter(vocabulary_sizes)))

    phase6f_freeze_path = ROOT / config["locked_inputs"]["phase6f_experiment_freeze"]
    phase6f_freeze = load_json(phase6f_freeze_path)
    checkpoint_path = Path(phase6f_freeze["selected_checkpoint_path"])
    if digest(checkpoint_path) != config["locked_inputs"]["phase6f_checkpoint_sha256"]:
        raise RuntimeError("Phase 6F checkpoint hash does not match Phase 6J config")
    families = {}
    for family in FAMILIES:
        model = CAURModel(
            load_base_model(checkpoint_path), categorical_sizes, family=family
        )
        total, trainable = model.parameter_counts()
        caps = config["model_families"][family]
        if total > int(caps["total_parameter_cap"]):
            raise RuntimeError(f"{family} exceeds its total parameter cap")
        if trainable > int(caps["trainable_parameter_cap"]):
            raise RuntimeError(f"{family} exceeds its trainable parameter cap")
        families[family] = {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "total_parameter_cap": int(caps["total_parameter_cap"]),
            "trainable_parameter_cap": int(caps["trainable_parameter_cap"]),
            "trainable_modules": caps["trainable"],
            "frozen_modules": caps["frozen"],
        }

    pair_gap_scale, positive_pair_count = positive_pair_gap_median(frame)
    immediate = frame.immediate_utility.to_numpy(dtype=float)
    immediate_delta = float(np.quantile(immediate, 0.75) - np.quantile(immediate, 0.25))
    if immediate_delta <= 0 or not np.isfinite(immediate_delta):
        raise RuntimeError("R12 immediate-utility IQR is not positive")

    training = config["training"]
    calibration = config["selection_aware_calibration"]
    gate = config["gate"]
    payload = {
        "schema": "phase6j-caur-r12-training-protocol-v1",
        "status": "FROZEN_BEFORE_FIRST_OPTIMIZER_STEP",
        "optimizer_steps_started": False,
        "r13_accessed": False,
        "r14_accessed": False,
        "config_sha256": digest(CONFIG_PATH),
        "freeze_implementation_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip(),
        "input_hashes": {
            "r12_horizon_freeze": digest(HORIZON_FREEZE),
            "r12_collection_integrity": digest(COLLECTION / "collection_integrity.json"),
            "r12_grouped_labels": source_sha256,
            "tensor_cache_integrity": digest(CACHE / "tensor_cache_integrity.json"),
            "tensor_manifest": digest(CACHE / "tensor_manifest.csv"),
        },
        "code_hashes": {
            relative: digest(ROOT / relative) for relative in CODE_PATHS
        },
        "prerequisite_checks": checks,
        "base_checkpoint": {
            "path": str(checkpoint_path.relative_to(ROOT)),
            "sha256": digest(checkpoint_path),
            "phase6f_experiment_freeze": str(phase6f_freeze_path.relative_to(ROOT)),
            "phase6f_experiment_freeze_sha256": digest(phase6f_freeze_path),
        },
        "feature_schema": {
            "categorical_columns": list(CATEGORICAL_COLUMNS),
            "numeric_columns": list(NUMERIC_COLUMNS),
            "categorical_sizes_with_unknown": list(categorical_sizes),
            "unknown_category_index": 0,
            "fold_training_audit": fold_rows,
            "normalization_fit": "two R12 training folds only",
            "numeric_normalization": "median/IQR, denominator floor 1e-6",
            "robust_z_clip": [-8.0, 8.0],
            "outcome_features_forbidden": config["candidate_features"][
                "outcome_fields_forbidden"
            ],
        },
        "families": families,
        "objective": {
            "weights": training["losses"],
            "pair_gap_scale": pair_gap_scale,
            "pair_gap_scale_rule": "median positive absolute within-state R12 label gap",
            "positive_ordered_pair_gaps": positive_pair_count,
            "immediate_huber_delta": immediate_delta,
            "immediate_huber_delta_rule": "full R12 immediate-utility IQR",
            "gap_weight_clip": training["gap_weight_clip"],
            "state_balanced": True,
            "ranking_standardization": "within-state only for pairwise and ListNet terms",
        },
        "grouped_oof": config["grouped_oof"],
        "training": {
            "seeds": config["rng"]["model_seeds"],
            "optimizer": training["optimizer"],
            "learning_rate": {
                family: training["learning_rate"][family] for family in FAMILIES
            },
            "weight_decay": training["weight_decay"],
            "state_groups_per_batch": training["state_groups_per_batch"],
            "gradient_norm_clip": training["gradient_norm_clip"],
            "deterministic_torch_algorithms": True,
            "maximum_epochs": 120,
            "patience": 12,
            "early_stopping_metric": "held-fold grouped-bootstrap selected-lift LCB",
            "early_stopping_protocol": (
                "for each outer held fold, tune epoch on one inner fold using the "
                "remaining inner fold for fitting; then reinitialize and fit both "
                "outer-training folds for the selected fixed epoch count"
            ),
            "inner_fold_rule": "validation=(outer_held+1)%3; fit=remaining fold",
            "minimum_lcb_improvement": 1e-6,
            "r13_refit": False,
        },
        "bootstrap": {
            "seed": config["rng"]["grouped_bootstrap_seed"],
            "resamples": 2000,
            "unit": "instance mean",
            "confidence_interval": 0.95,
        },
        "calibration": {
            "fit_rows": "R12 grouped-OOF argmax-selected winners only",
            "candidate_methods": ["PLATT", "ISOTONIC"],
            "isotonic_minimum_selected_rows": calibration[
                "isotonic_minimum_selected_rows"
            ],
            "selection": calibration["selection"],
            "ece_max": calibration["ece_max"],
        },
        "gate": {
            "p_min_grid": gate["p_min_grid"],
            "lcb_lambda_grid": gate["lcb_lambda_grid"],
            "delta_min_grid": gate["delta_min_grid"],
            "immediate_harm_floor": gate["immediate_harm_floor"],
            "minimum_direct_interventions_per_scale": gate[
                "minimum_direct_interventions_per_scale"
            ],
            "forced_abstention_minimum": 40,
            "selection_order": gate["threshold_selection"],
        },
        "j3_activation": {
            "pairwise_accuracy_min": 0.60,
            "activate_if_any_scale_mean_spearman_negative": True,
            "source_rule": config["model_families"]["J3_CONT_RELATIONAL"][
                "activation"
            ],
        },
        "r12_acceptance": config["r12_acceptance"],
        "latency": {
            "p90_neural_decision_ms_max": config["runtime"][
                "p90_neural_decision_ms_max"
            ],
            "p90_total_decision_ms_max": config["runtime"][
                "p90_total_decision_ms_max"
            ],
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(payload, OUTPUT)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
