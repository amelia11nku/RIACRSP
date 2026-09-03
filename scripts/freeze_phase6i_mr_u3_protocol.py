#!/usr/bin/env python3
"""Freeze the conditionally activated Phase 6I U3 training protocol."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_clgri.ni.encoder import NIModelConfig  # noqa: E402
from rcias_clgri.ni.scorer import CSGTargetSetScorer  # noqa: E402
from rcias_clgri.ni.tensorize import CSGTensorizer  # noqa: E402
from scripts.prepare_phase6i_mr_training_data import atomic_json, digest, load_json  # noqa: E402


TRAINING_PROTOCOL = ROOT / "outputs/phase6i_mr/frozen/training_protocol.json"
ACTIVATION = ROOT / "outputs/phase6i_mr/model_training/u3_activation_decision.json"
PROBE = ROOT / "outputs/phase6i_mr/model_training/representation_probe.json"
TENSOR_INTEGRITY = ROOT / "outputs/phase6i_mr/u3_tensor_cache/u3_tensor_integrity.json"
PHASE6F_FREEZE = ROOT / "outputs/phase6f/audit/experiment_freeze.json"
OUTPUT = ROOT / "outputs/phase6i_mr/frozen/u3_training_protocol.json"


def main() -> None:
    training = load_json(TRAINING_PROTOCOL)
    activation = load_json(ACTIVATION)
    probe = load_json(PROBE)
    tensor = load_json(TENSOR_INTEGRITY)
    phase6f = load_json(PHASE6F_FREEZE)
    if not all([
        training.get("r10_accessed") is False,
        training.get("r11_accessed") is False,
        activation.get("u3_activated") is True,
        probe.get("contrastive_activated") is False,
        tensor.get("status") == "PASS",
        tensor.get("r10_accessed") is False,
        tensor.get("r11_accessed") is False,
    ]):
        raise RuntimeError("U3 was not activated from complete R09-only evidence")
    checkpoint_path = Path(phase6f["selected_checkpoint_path"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = CSGTargetSetScorer(
        CSGTensorizer(), NIModelConfig(**checkpoint["model_config"])
    )
    model.load_state_dict(checkpoint["model_state"])
    prefixes = [
        "state_encoder.layers.1",
        "action_encoder.projection",
        "utility_head",
    ]
    trainable = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name.startswith(tuple(prefixes))
    )
    code_paths = [
        "scripts/build_phase6i_mr_u3_tensor_cache.py",
        "scripts/train_phase6i_mr_u3.py",
        "rcias_clgri/ni/phase6i_heads.py",
    ]
    base_budget = training["training_budgets"]["conditional_u3"]
    payload = {
        "schema": "phase6i-mr-u3-training-protocol-v1.2",
        "status": "FROZEN_BEFORE_FIRST_U3_OPTIMIZER_STEP_AND_R10_R11_ACCESS",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "activation_evidence": {
            "rule": training["u3_activation_rule"],
            "decision_sha256": digest(ACTIVATION),
            "best_u1_u2_oof_spearman": activation["best_candidate"]["mean_overall_spearman"],
        },
        "contrastive": {
            "activated": False,
            "probe_sha256": digest(PROBE),
            "reason": "the two preregistered activation conditions were not both satisfied",
        },
        "base_checkpoint_path": str(checkpoint_path),
        "base_checkpoint_sha256": digest(checkpoint_path),
        "total_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_prefixes": prefixes,
        "trainable_parameter_count": trainable,
        "maximum_trainable_parameters": base_budget["maximum_trainable_parameters"],
        "utility_head_initialization": "seed-specific Xavier uniform weights and zero biases; all other trainable tensors warm-start from frozen Phase6F",
        "training_seeds": training["training_seeds"],
        "grouped_oof_folds": training["grouped_oof_folds"],
        "variant": "ONE_ROUND_HARD_AGG_ONLY",
        "source_state_counts": {"earlier_train": 540, "r09_live": 1620, "r09_hard_state": 360},
        "source_loss_weights": {"earlier_train": 0.20, "r09_live": 0.60, "r09_hard_state": 0.20},
        "state_batch_counts": {"earlier_train": 3, "r09_live": 10, "r09_hard_state": 3},
        "epochs": base_budget["epochs"],
        "optimizer": "AdamW",
        "learning_rates": base_budget["learning_rates"],
        "weight_decay": 0.0001,
        "gradient_norm_clip": 5.0,
        "device": "CUDA",
        "numeric_precision": "float32",
        "early_stopping": False,
        "target": training["targets"]["immediate"],
        "objective": training["objective"],
        "score_stability_weight": base_budget["score_stability_weight"],
        "score_stability": "state-balanced MSE between within-state standardized adapted and frozen-reference score-head outputs",
        "score_head_parameters_trainable": False,
        "steps_per_epoch": "ceil(number of eligible R09 live training states / 10); other sources sampled cyclically to their fixed per-batch counts",
        "family_artifact_rule": training["family_artifact_rule"],
        "code_hashes": {path: digest(ROOT / path) for path in code_paths},
        "input_hashes": {
            "training_protocol": digest(TRAINING_PROTOCOL),
            "u3_tensor_integrity": digest(TENSOR_INTEGRITY),
            "hard_state_integrity": digest(ROOT / "outputs/phase6i_mr/hard_state_round1/hard_state_integrity.json"),
            "old_train_selection": digest(ROOT / "outputs/phase6i_mr/training_data/old_train_state_selection.csv"),
        },
        "r10_accessed": False,
        "r11_accessed": False,
    }
    if trainable > payload["maximum_trainable_parameters"]:
        raise RuntimeError("U3 exceeds the frozen trainable-parameter cap")
    atomic_json(payload, OUTPUT)
    print({
        "status": payload["status"],
        "sha256": digest(OUTPUT),
        "total_parameters": payload["total_parameter_count"],
        "trainable_parameters": trainable,
        "contrastive": False,
        "r10_accessed": False,
        "r11_accessed": False,
    })


if __name__ == "__main__":
    main()
