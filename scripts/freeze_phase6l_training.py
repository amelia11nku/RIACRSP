#!/usr/bin/env python3
"""Freeze the Phase 6L dataset/model/training boundary before optimization."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_clgri.analysis.phase6l_legacy_score import (  # noqa: E402
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
)
from rcias_clgri.ni.encoder import NIModelConfig  # noqa: E402
from rcias_clgri.ni.phase6l_score_free_model import ScoreFreeCAURModel  # noqa: E402
from rcias_clgri.ni.scorer import CSGTargetSetScorer  # noqa: E402
from rcias_clgri.ni.tensorize import CSGTensorizer  # noqa: E402


CONFIG = ROOT / "configs/phase6l_legacy_score_decoupling_v1.json"
PREREG = ROOT / "outputs/phase6l_legacy_score_decoupling_v1/preregistration/preregistration.json"
DATA = ROOT / "outputs/phase6l_legacy_score_decoupling_v1/data"
SOURCE = DATA / "r12_score_free_grouped_labels.parquet"
OUTPUT = ROOT / "outputs/phase6l_legacy_score_decoupling_v1/training/training_protocol.json"
CODE_PATHS = (
    "rcias_clgri/analysis/phase6l_legacy_score.py",
    "rcias_clgri/ni/phase6l_score_free_model.py",
    "scripts/build_phase6l_score_free_dataset.py",
    "scripts/freeze_phase6l_training.py",
    "scripts/train_phase6l_score_free.py",
)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != text:
        raise RuntimeError("refusing to replace frozen Phase 6L training protocol")
    if not path.exists():
        path.write_text(text)


def load_base(path: Path) -> CSGTargetSetScorer:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = CSGTargetSetScorer(CSGTensorizer(), NIModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state"])
    return model


def transform(frame: pd.DataFrame) -> dict:
    return {
        "vocabularies": {
            column: sorted(frame[column].astype(str).unique().tolist())
            for column in CATEGORICAL_COLUMNS
        },
        "medians": {
            column: float(np.median(frame[column].to_numpy(dtype=float)))
            for column in NUMERIC_COLUMNS
        },
        "iqrs": {
            column: max(float(
                np.quantile(frame[column].to_numpy(dtype=float), 0.75)
                - np.quantile(frame[column].to_numpy(dtype=float), 0.25)
            ), 1e-6) for column in NUMERIC_COLUMNS
        },
    }


def positive_pair_gap_median(frame: pd.DataFrame) -> tuple[float, int]:
    gaps = []
    for _, group in frame.groupby("state_id", sort=True):
        values = group.continuation_advantage_mean.to_numpy(dtype=float)
        absolute = np.abs(values[:, None] - values[None, :])
        gaps.append(absolute[absolute > 1e-12])
    combined = np.concatenate(gaps)
    return float(np.median(combined)), int(len(combined))


def main() -> None:
    if OUTPUT.exists():
        existing = json.loads(OUTPUT.read_text())
        for relative, expected in existing["code_hashes"].items():
            if digest(ROOT / relative) != expected:
                raise RuntimeError(f"frozen training code changed: {relative}")
        print(json.dumps(existing, indent=2, sort_keys=True))
        return
    config = json.loads(CONFIG.read_text())
    prereg = json.loads(PREREG.read_text())
    manifest = json.loads((DATA / "dataset_manifest.json").read_text())
    checks = {
        "preregistration_frozen": prereg["status"]
        == "FROZEN_BEFORE_PHASE6L_DATASET_DERIVATION_OR_OPTIMIZER_STEP",
        "dataset_pass": manifest["status"] == "PASS",
        "dataset_hash": digest(SOURCE) == manifest["grouped_sha256"],
        "states": manifest["derivation"]["states"] == 288,
        "candidates": manifest["derivation"]["candidates"] == 6809,
        "reanchored_states": manifest["derivation"]["reanchored_states"] == 10,
        "live_feature_parity": manifest["live_builder_parity"]["states_checked"] == 288,
        "zero_historical_forward": manifest["online_historical_score_forward_calls"] == 0,
        "r13_locked": not (ROOT / "outputs/phase6j_caur/r13_selection/access_ledger.json").exists(),
        "r14_locked": not (ROOT / "outputs/phase6j_caur/r14_holdout/access_ledger.json").exists(),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Phase 6L training freeze prerequisites failed: {checks}")
    frame = pd.read_parquet(SOURCE)
    fold_audit = []
    vocabulary_sizes = set()
    for held in (0, 1, 2):
        training = frame[frame.oof_fold.ne(held)]
        record = transform(training)
        sizes = tuple(len(record["vocabularies"][column]) for column in CATEGORICAL_COLUMNS)
        vocabulary_sizes.add(sizes)
        fold_audit.append({
            "held_fold": held,
            "training_instances": int(training.instance_id.nunique()),
            "training_states": int(training.state_id.nunique()),
            "training_actions": len(training),
            "categorical_vocabulary_sizes": dict(zip(CATEGORICAL_COLUMNS, sizes)),
        })
    if vocabulary_sizes != {(24, 7, 5)}:
        raise RuntimeError(f"unexpected Phase 6L vocabularies: {vocabulary_sizes}")
    categorical_sizes = tuple(value + 1 for value in next(iter(vocabulary_sizes)))
    checkpoint = ROOT / config["locked_inputs"]["phase6f_base_checkpoint"]
    if digest(checkpoint) != config["locked_inputs"]["phase6f_base_checkpoint_sha256"]:
        raise RuntimeError("base checkpoint changed")
    model = ScoreFreeCAURModel(load_base(checkpoint), categorical_sizes, family=config["primary_model"]["model_id"])
    total, trainable = model.parameter_counts()
    if total > config["primary_model"]["total_parameter_cap"] or trainable > config["primary_model"]["trainable_parameter_cap"]:
        raise RuntimeError("Phase 6L parameter cap failed")
    pair_gap, pair_count = positive_pair_gap_median(frame)
    immediate = frame.immediate_utility.to_numpy(dtype=float)
    immediate_delta = float(np.quantile(immediate, 0.75) - np.quantile(immediate, 0.25))
    payload = {
        "schema": "phase6l-score-free-training-protocol-v1",
        "status": "FROZEN_BEFORE_FIRST_OPTIMIZER_STEP",
        "optimizer_steps_started": False,
        "freeze_implementation_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "config_sha256": digest(CONFIG),
        "preregistration_sha256": digest(PREREG),
        "input_hashes": {
            "dataset_manifest": digest(DATA / "dataset_manifest.json"),
            "score_free_grouped_labels": digest(SOURCE),
            "score_free_seed_labels": digest(DATA / "r12_score_free_seed_labels.parquet"),
            "feature_schema": digest(DATA / "feature_schema.json"),
            "feature_provenance": digest(DATA / "feature_provenance.json"),
            "normalization_manifest": digest(DATA / "normalization_manifest.json"),
            "phase6j_tensor_cache_integrity": digest(ROOT / "outputs/phase6j_caur/tensor_cache/tensor_cache_integrity.json"),
        },
        "code_hashes": {relative: digest(ROOT / relative) for relative in CODE_PATHS},
        "prerequisite_checks": checks,
        "base_checkpoint": {
            "path": str(checkpoint.relative_to(ROOT)), "sha256": digest(checkpoint)
        },
        "feature_schema": {
            "categorical_columns": list(CATEGORICAL_COLUMNS),
            "numeric_columns": list(NUMERIC_COLUMNS),
            "categorical_sizes_with_unknown": list(categorical_sizes),
            "fold_training_audit": fold_audit,
            "normalization_fit": "training folds only",
            "numeric_normalization": "median/IQR, denominator floor 1e-6",
            "robust_z_clip": [-8.0, 8.0],
        },
        "families": {
            config["primary_model"]["model_id"]: {
                "total_parameters": total,
                "trainable_parameters": trainable,
                "total_parameter_cap": config["primary_model"]["total_parameter_cap"],
                "trainable_parameter_cap": config["primary_model"]["trainable_parameter_cap"],
            }
        },
        "objective": {
            "weights": {
                "pairwise_logistic_advantage": 1.0,
                "listnet_state_list": 0.75,
                "huber_advantage": 0.5,
                "bce_beats_fallback": 0.25,
                "huber_immediate_utility_auxiliary": 0.1,
            },
            "pair_gap_scale": pair_gap,
            "positive_ordered_pair_gaps": pair_count,
            "immediate_huber_delta": immediate_delta,
            "gap_weight_clip": [0.25, 4.0],
            "state_balanced": True,
            "ranking_standardization": "within-state only for pairwise and ListNet terms",
        },
        "grouped_oof": config["grouped_oof"],
        "training": {
            "seeds": config["training"]["seeds"],
            "optimizer": "AdamW",
            "learning_rate": {config["primary_model"]["model_id"]: 0.0003},
            "weight_decay": 0.0001,
            "state_groups_per_batch": 8,
            "gradient_norm_clip": 1.0,
            "deterministic_torch_algorithms": True,
            "maximum_epochs": 120,
            "patience": 12,
            "minimum_lcb_improvement": 1e-8,
            "r13_refit": False,
        },
        "bootstrap": config["bootstrap"],
        "calibration": config["calibration"],
        "gate": config["gate"],
        "r12_acceptance": config["quality_gates"],
        "latency": config["runtime"],
        "r13_accessed": False,
        "r14_accessed": False,
    }
    atomic_json(OUTPUT, payload)
    print(json.dumps({
        "status": payload["status"], "total_parameters": total,
        "trainable_parameters": trainable, "pair_gap_scale": pair_gap,
        "immediate_huber_delta": immediate_delta,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
