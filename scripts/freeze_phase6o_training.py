#!/usr/bin/env python3
"""Freeze the implemented Phase 6O Route B training contract."""

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

from rcias_clgri.ni.phase6n_candidate_conditioned import (  # noqa: E402
    BOUNDARY_FAMILIES,
    DIRECTIONS,
)
from rcias_clgri.ni.phase6o_top_utility import FAMILY, TopUtilityCSGModel  # noqa: E402
from scripts import train_phase6o_top_utility as training  # noqa: E402


CONFIG = ROOT / "configs/phase6o_neural_shortlist_v1.json"
PREREG = ROOT / "outputs/phase6o_neural_shortlist_v1/preregistration/preregistration.json"
SOURCE_HASHES = ROOT / "outputs/phase6o_neural_shortlist_v1/preregistration/source_hashes.json"
COMPLETION = ROOT / "outputs/phase6o_neural_shortlist_v1/relabeling/completion_audit.json"
CACHE_INTEGRITY = ROOT / "outputs/phase6n_candidate_conditioned_csg_v1/tensor_cache/tensor_cache_integrity.json"
OUT = ROOT / "outputs/phase6o_neural_shortlist_v1/training/training_protocol.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    config = load_json(CONFIG)
    prereg = load_json(PREREG)
    source_hashes = load_json(SOURCE_HASHES)
    completion = load_json(COMPLETION)
    cache = load_json(CACHE_INTEGRITY)
    require(config["route"]["primary_family"] == FAMILY, "Phase 6O family changed")
    require(config["route"]["decision"] == "PROCEED_TOP_UTILITY_RETRAIN", "Phase 6O route changed")
    require(prereg["status"] == "FROZEN_BEFORE_TARGETED_RELABEL_OR_OPTIMIZER_STEP", "Phase 6O preregistration changed")
    require(completion["status"] == "PASS", "Phase 6O relabeling did not pass")
    require(completion["states"] == 864 and completion["full_bank_candidates"] == 20441, "Phase 6O relabeling scope changed")
    require(completion["targeted_five_seed_candidates"] == 4786, "Phase 6O five-seed scope changed")
    require(completion["scientific_calculation_changed_by_recovery"] is False, "recovery changed scientific calculation")
    require(completion["historical_score_calls"] == 0, "historical scorer entered relabeling")
    for relative, expected in completion["artifacts"].items():
        require(training.digest(ROOT / relative) == expected, f"relabel artifact changed: {relative}")
    require(cache["status"] == "PASS" and cache["states"] == 864 and cache["actions"] == 20441, "Phase 6N cache cannot be reused")
    require(source_hashes["config_sha256"] == training.digest(CONFIG), "Phase 6O config hash changed")
    for relative, expected in source_hashes["sources"].items():
        require(training.digest(ROOT / relative) == expected, f"protected predecessor changed: {relative}")
    for relative in (
        "outputs/phase6j_caur/r13_selection/access_ledger.json",
        "outputs/phase6j_caur/r14_holdout/access_ledger.json",
        "outputs/phase6o_neural_shortlist_v1/r13_selection/access_ledger.json",
        "outputs/phase6o_neural_shortlist_v1/r14_holdout/access_ledger.json",
    ):
        require(not (ROOT / relative).exists(), f"forbidden holdout access: {relative}")

    frame = training.attach_candidate_noise(
        pd.read_parquet(training.SOURCE), pd.read_parquet(training.RAW)
    )
    transform = training.phase6n.fit_feature_transform(frame)
    base = training.phase6n.load_base_model({
        "base_checkpoint": {
            "path": config["locked_inputs"]["phase6f_base_checkpoint"],
            "sha256": training.digest(
                ROOT / config["locked_inputs"]["phase6f_base_checkpoint"]
            ),
        }
    })
    model = TopUtilityCSGModel(
        base,
        tuple(
            len(transform.vocabularies[column]) + 1
            for column in training.phase6n.CATEGORICAL_COLUMNS
        ),
    )
    total, trainable = model.parameter_counts()
    encoder_parameters = sum(
        parameter.numel() for parameter in model.state_encoder.parameters()
    )
    require(not any(parameter.requires_grad for parameter in model.state_encoder.parameters()), "Phase 6F encoder is trainable")
    require(trainable == total - encoder_parameters, "Phase 6O trainable boundary changed")

    fold_audit = []
    for held_fold in range(3):
        held = frame[frame.oof_fold.eq(held_fold)]
        outer = frame[~frame.oof_fold.eq(held_fold)]
        outer_folds = sorted(outer.oof_fold.unique().tolist())
        directions = []
        for train_fold, validation_fold in (
            (outer_folds[0], outer_folds[1]),
            (outer_folds[1], outer_folds[0]),
        ):
            fitted = frame[frame.oof_fold.eq(train_fold)]
            validated = frame[frame.oof_fold.eq(validation_fold)]
            contract = training.fit_fold_contract(fitted)
            directions.append({
                "training_fold": int(train_fold),
                "validation_fold": int(validation_fold),
                "training_instances": int(fitted.instance_id.nunique()),
                "validation_instances": int(validated.instance_id.nunique()),
                "instance_overlap": sorted(
                    set(fitted.instance_id.astype(str))
                    & set(validated.instance_id.astype(str))
                ),
                "contract": contract.to_dict(),
            })
        fold_audit.append({
            "held_fold": held_fold,
            "held_instances": int(held.instance_id.nunique()),
            "held_states": int(held.state_id.nunique()),
            "held_candidates": len(held),
            "outer_training_instances": int(outer.instance_id.nunique()),
            "outer_training_states": int(outer.state_id.nunique()),
            "outer_training_candidates": len(outer),
            "held_training_instance_overlap": sorted(
                set(held.instance_id.astype(str)) & set(outer.instance_id.astype(str))
            ),
            "symmetric_inner_directions": directions,
            "outer_contract": training.fit_fold_contract(outer).to_dict(),
        })
    require(all(not row["held_training_instance_overlap"] for row in fold_audit), "outer fold isolation failed")
    require(all(not direction["instance_overlap"] for row in fold_audit for direction in row["symmetric_inner_directions"]), "inner fold isolation failed")

    code_paths = (
        "rcias_clgri/ni/batching.py",
        "rcias_clgri/ni/cache.py",
        "rcias_clgri/ni/encoder.py",
        "rcias_clgri/ni/scorer.py",
        "rcias_clgri/ni/tensorize.py",
        "rcias_clgri/ni/phase6n_candidate_conditioned.py",
        "rcias_clgri/ni/phase6o_top_utility.py",
        "scripts/train_phase6j_caur.py",
        "scripts/train_phase6n_candidate_conditioned.py",
        "scripts/train_phase6o_top_utility.py",
        "scripts/freeze_phase6o_training.py",
    )
    train_config = config["training"]
    checkpoint_path = ROOT / config["locked_inputs"]["phase6f_base_checkpoint"]
    payload = {
        "schema": "phase6o-top-utility-training-protocol-v1",
        "status": "FROZEN_BEFORE_FIRST_OPTIMIZER_STEP",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "freeze_implementation_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "preregistration_sha256": training.digest(PREREG),
        "source_hashes_sha256": training.digest(SOURCE_HASHES),
        "input_hashes": {
            "config": training.digest(CONFIG),
            "relabel_completion_audit": training.digest(COMPLETION),
            "training_grouped_labels": training.digest(training.SOURCE),
            "combined_seed_labels": training.digest(training.RAW),
            "additional_seed_labels": training.digest(
                ROOT / "outputs/phase6o_neural_shortlist_v1/relabeling/additional_seed_labels.parquet"
            ),
            "tensor_cache_integrity": training.digest(CACHE_INTEGRITY),
            "tensor_manifest": training.digest(training.CACHE / "tensor_manifest.csv"),
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
            "frozen_encoder_parameters": encoder_parameters,
            "frozen_modules": ["entire Phase 6F RT-HGT state encoder"],
            "trainable_modules": [
                "candidate-conditioned pooler",
                "fallback-relative fusion",
                "continuation advantage head",
                "beats-fallback head",
            ],
            "boundary_family_order": list(BOUNDARY_FAMILIES),
            "boundary_direction_order": list(DIRECTIONS),
            "graph_encoder_executions_per_state": 1,
            "per_candidate_graph_rerun": False,
            "precision": "FP32_ONLY",
        },
        "feature_schema": {
            "categorical_columns": list(training.phase6n.CATEGORICAL_COLUMNS),
            "numeric_columns": list(training.phase6n.NUMERIC_COLUMNS),
            "categorical_sizes_with_unknown": [
                len(transform.vocabularies[column]) + 1
                for column in training.phase6n.CATEGORICAL_COLUMNS
            ],
            "normalization": "training-fold median/IQR, denominator floor 1e-6, clip [-8,8]",
            "outcome_inputs": [],
        },
        "fold_audit": fold_audit,
        "training": {
            "seeds": train_config["seeds"],
            "maximum_epochs": train_config["maximum_epochs"],
            "state_groups_per_batch": train_config["state_groups_per_batch"],
            "learning_rate": train_config["learning_rate"],
            "weight_decay": train_config["weight_decay"],
            "gradient_norm_clip": train_config["gradient_norm_clip"],
            "source_balancing": train_config["source_balancing"],
            "noise_margin": train_config["noise_margin"],
            "opportunity_weight": train_config["opportunity_weight"],
            "fold_scales": train_config["fold_scales"],
            "objective": train_config["objective"],
            "objective_weights": {
                "top_set_cross_entropy": 1.0,
                "near_best_vs_rest_regret_logistic": 1.0,
                "soft_expected_utility": 1.0,
                "advantage_huber": 0.1,
                "beats_fallback_bce": 0.1,
            },
            "inner_selection": train_config["inner_selection"],
            "rng": {
                "initialization_seed": "training seed",
                "epoch_shuffle_seed": "seed + epoch * 1000003 + direction-specific frozen salt",
                "inner_direction_salts": "held_fold * 10007 + direction_index * 1009 + 101",
                "outer_refit_salt": "held_fold * 10007 + 303",
            },
            "deterministic_torch_algorithms": True,
        },
        "bootstrap": config["oof_quality_gate"]["paired_bootstrap"],
        "fold_isolation": "whole instance",
        "symmetric_two_way_inner_validation": True,
        "optimizer_steps_started": False,
        "outer_oof_generated": False,
        "historical_score_calls": 0,
        "gurobi_run": False,
        "r13_accessed": False,
        "r14_accessed": False,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    training.metrics.atomic_json(payload, OUT)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
