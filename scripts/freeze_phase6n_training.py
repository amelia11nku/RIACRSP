#!/usr/bin/env python3
"""Freeze the implemented Phase 6N model and N4 training contract."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_clgri.ni.encoder import NIModelConfig  # noqa: E402
from rcias_clgri.ni.phase6n_candidate_conditioned import (  # noqa: E402
    BOUNDARY_FAMILIES,
    DIRECTIONS,
    FAMILY,
    CandidateConditionedCSGModel,
)
from rcias_clgri.ni.scorer import CSGTargetSetScorer  # noqa: E402
from rcias_clgri.ni.tensorize import CSGTensorizer  # noqa: E402
from scripts import train_phase6n_candidate_conditioned as training  # noqa: E402


CONFIG = ROOT / "configs/phase6n_candidate_conditioned_csg_v1.json"
PREREG = ROOT / "outputs/phase6n_candidate_conditioned_csg_v1/preregistration/preregistration.json"
SOURCE_HASHES = ROOT / "outputs/phase6n_candidate_conditioned_csg_v1/preregistration/source_hashes.json"
DATA_INTEGRITY = ROOT / "outputs/phase6n_candidate_conditioned_csg_v1/data/data_integrity.json"
SOURCE = ROOT / "outputs/phase6n_candidate_conditioned_csg_v1/data/combined/r12_expanded_grouped_labels.parquet"
CACHE_INTEGRITY = ROOT / "outputs/phase6n_candidate_conditioned_csg_v1/tensor_cache/tensor_cache_integrity.json"
OUT = ROOT / "outputs/phase6n_candidate_conditioned_csg_v1/training/training_protocol.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    config = load_json(CONFIG)
    prereg = load_json(PREREG)
    source_hashes = load_json(SOURCE_HASHES)
    data = load_json(DATA_INTEGRITY)
    cache = load_json(CACHE_INTEGRITY)
    for relative, expected in source_hashes["files"].items():
        if training.digest(ROOT / relative) != expected:
            raise RuntimeError(f"Phase 6N locked predecessor changed: {relative}")
    checks = (
        config.get("primary_family") == FAMILY,
        config.get("promotable_families") == [FAMILY],
        prereg.get("status") == "FROZEN_BEFORE_NEW_ROLLOUT_OR_OPTIMIZER_STEP",
        prereg.get("optimizer_steps_started") is False,
        data.get("status") == "PASS",
        data.get("combined_states") == 864,
        data.get("combined_candidate_rows") == 20441,
        cache.get("status") == "PASS",
        cache.get("states") == 864,
        cache.get("actions") == 20441,
        training.digest(SOURCE) == data["artifacts"].get(str(SOURCE.relative_to(ROOT))),
        data.get("historical_score_calls") == cache.get("historical_score_calls") == 0,
        not (ROOT / "outputs/phase6j_caur/r13_selection/access_ledger.json").exists(),
        not (ROOT / "outputs/phase6j_caur/r14_holdout/access_ledger.json").exists(),
        not (ROOT / "outputs/phase6n_candidate_conditioned_csg_v1/r13_selection/access_ledger.json").exists(),
        not (ROOT / "outputs/phase6n_candidate_conditioned_csg_v1/r14_holdout/access_ledger.json").exists(),
    )
    if not all(checks):
        raise RuntimeError("Phase 6N training freeze prerequisite failed")

    frame = pd.read_parquet(SOURCE)
    transform = training.fit_feature_transform(frame)
    checkpoint_path = ROOT / config["locked_inputs"]["phase6f_base_checkpoint"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    base = CSGTargetSetScorer(
        CSGTensorizer(), NIModelConfig(**checkpoint["model_config"])
    )
    base.load_state_dict(checkpoint["model_state"])
    model = CandidateConditionedCSGModel(
        base,
        tuple(
            len(transform.vocabularies[column]) + 1
            for column in training.CATEGORICAL_COLUMNS
        ),
        family=FAMILY,
    )
    total, trainable = model.parameter_counts()
    final_block = sum(
        parameter.numel() for parameter in model.state_encoder.layers[-1].parameters()
    )
    new_modules = trainable - final_block
    if total > 8_000_000 or trainable > 4_000_000:
        raise RuntimeError("Phase 6N parameter cap exceeded")

    fold_audit = []
    for held_fold in range(3):
        held = frame[frame.oof_fold.eq(held_fold)]
        outer = frame[~frame.oof_fold.eq(held_fold)]
        fold_audit.append({
            "held_fold": held_fold,
            "held_instances": int(held.instance_id.nunique()),
            "held_states": int(held.state_id.nunique()),
            "held_candidates": len(held),
            "training_instances": int(outer.instance_id.nunique()),
            "training_states": int(outer.state_id.nunique()),
            "training_candidates": len(outer),
            "instance_overlap": sorted(
                set(held.instance_id.astype(str)) & set(outer.instance_id.astype(str))
            ),
        })
    if any(row["instance_overlap"] for row in fold_audit):
        raise RuntimeError("Phase 6N whole-instance fold isolation failed")

    code_paths = (
        "rcias_clgri/ni/phase6n_candidate_conditioned.py",
        "scripts/train_phase6n_candidate_conditioned.py",
        "scripts/freeze_phase6n_training.py",
    )
    train_config = config["training"]
    payload = {
        "schema": "phase6n-training-protocol-v1",
        "status": "FROZEN_BEFORE_FIRST_OPTIMIZER_STEP",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "freeze_implementation_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "preregistration_sha256": training.digest(PREREG),
        "source_hashes_sha256": training.digest(SOURCE_HASHES),
        "input_hashes": {
            "config": training.digest(CONFIG),
            "data_integrity": training.digest(DATA_INTEGRITY),
            "expanded_grouped_labels": training.digest(SOURCE),
            "tensor_cache_integrity": training.digest(CACHE_INTEGRITY),
        },
        "code_hashes": {
            relative: training.digest(ROOT / relative) for relative in code_paths
        },
        "base_checkpoint": {
            "path": str(checkpoint_path.relative_to(ROOT)),
            "sha256": training.digest(checkpoint_path),
        },
        "model": {
            "family": FAMILY,
            "total_parameters": total,
            "trainable_parameters": trainable,
            "trainable_final_relation_block_parameters": final_block,
            "trainable_new_module_parameters": new_modules,
            "frozen_modules": [
                "all state-encoder input projections",
                "state-encoder relation block 0",
                "state-encoder graph projection",
            ],
            "trainable_modules": [
                "state-encoder relation block 1",
                "candidate target/relation/critical pooler",
                "fallback-relative fusion",
                "continuation advantage head",
                "beats-fallback head",
            ],
            "boundary_family_order": list(BOUNDARY_FAMILIES),
            "boundary_direction_order": list(DIRECTIONS),
            "boundary_node_semantics": "unique (candidate, heterogeneous external node) set",
            "boundary_count_denominator": "log1p(total heterogeneous CSG node count in state)",
            "critical_sync_semantics": "unique external nodes on binding temporal relations incident to candidate-critical OP nodes",
            "graph_encoder_executions_per_state": 1,
            "per_candidate_graph_rerun": False,
            "historical_score_calls": 0,
            "precision": "FP32_ONLY",
        },
        "feature_schema": {
            "categorical_columns": list(training.CATEGORICAL_COLUMNS),
            "numeric_columns": list(training.NUMERIC_COLUMNS),
            "categorical_sizes_with_unknown": [
                len(transform.vocabularies[column]) + 1
                for column in training.CATEGORICAL_COLUMNS
            ],
            "normalization": "training-fold median/IQR, denominator floor 1e-6, clip [-8,8]",
            "outcome_inputs": [],
        },
        "fold_audit": fold_audit,
        "training": {
            "seeds": train_config["seeds"],
            "maximum_epochs": train_config["maximum_epochs"],
            "patience": train_config["patience"],
            "state_groups_per_batch": train_config["state_groups_per_batch"],
            "learning_rate_new_modules": train_config["learning_rate_new_modules"],
            "learning_rate_final_relation_block": train_config["learning_rate_final_relation_block"],
            "weight_decay": train_config["weight_decay"],
            "gradient_norm_clip": train_config["gradient_norm_clip"],
            "objective_weights": train_config["objective"],
            "pair_gap_scale": "training-fold median positive within-state absolute gap",
            "huber_delta": "training-fold MAD clipped [0.005,0.05]",
            "epoch_selection": "maximize validation grouped raw selected-lift LCB; tie by lower joint loss then earlier epoch",
            "deterministic_torch_algorithms": True,
        },
        "bootstrap": config["raw_representation_gate"]["bootstrap"],
        "optimizer_steps_started": False,
        "outer_oof_generated": False,
        "historical_score_calls": 0,
        "r13_accessed": False,
        "r14_accessed": False,
    }
    training.metrics.atomic_json(payload, OUT)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
