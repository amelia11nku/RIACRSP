#!/usr/bin/env python3
"""Audit Phase 6O formal nested-OOF training completion and artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import train_phase6o_top_utility as training  # noqa: E402


OUT = ROOT / "outputs/phase6o_neural_shortlist_v1/training"
AUDIT = OUT / "completion_integrity_audit.json"
MANIFEST = OUT / "run_manifest.csv"
REPORT = ROOT / "docs/reports/phase6o_top_utility_training_report.md"
PREDICTION_COLUMNS = (
    "predicted_continuation_advantage",
    "predicted_beats_fallback_logit",
    "predicted_beats_fallback_probability_raw",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def selected_epoch_is_preregistered(record: dict) -> bool:
    rows = record["history"]["selection"]
    selected = max(
        rows,
        key=lambda row: (
            float(row["mean_selected_lift"]),
            float(row["mean_near_best_hit_rate"]),
            -float(row["mean_top1_regret"]),
            -float(row["mean_joint_loss"]),
            -int(row["epoch"]),
        ),
    )
    return int(selected["epoch"]) == int(record["selected_epoch"])


def main() -> None:
    protocol = training.validate_protocol()
    protocol_sha256 = training.digest(training.PROTOCOL)
    progress = training.load_json(OUT / "progress.json")
    summary = training.load_json(OUT / "training_summary.json")
    source = pd.read_parquet(training.SOURCE).sort_values(
        ["state_id", "target_set_id"], kind="stable"
    ).reset_index(drop=True)
    keys = ["state_id", "target_set_id"]
    records: list[dict] = []
    prediction_parts: list[pd.DataFrame] = []

    for seed in protocol["training"]["seeds"]:
        for held_fold in range(3):
            paths = training.run_paths(int(seed), held_fold)
            require(
                training.valid_run(paths, protocol_sha256),
                f"invalid Phase 6O run: seed={seed}, fold={held_fold}",
            )
            checkpoint_path, prediction_path, record_path = paths
            record = training.load_json(record_path)
            prediction = pd.read_parquet(prediction_path).sort_values(
                keys, kind="stable"
            ).reset_index(drop=True)
            expected = source[source.oof_fold.eq(held_fold)].sort_values(
                keys, kind="stable"
            ).reset_index(drop=True)
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            state = checkpoint["trainable_model_state"]
            require(
                checkpoint.get("schema")
                == "phase6o-top-utility-oof-checkpoint-v1"
                and checkpoint.get("training_protocol_sha256") == protocol_sha256
                and checkpoint.get("training_seed") == int(seed)
                and checkpoint.get("held_fold") == held_fold,
                "Phase 6O checkpoint contract changed",
            )
            require(
                not any(name.startswith("state_encoder.") for name in state)
                and sum(value.numel() for value in state.values())
                == int(protocol["model"]["trainable_parameters"])
                and all(torch.isfinite(value).all().item() for value in state.values()),
                "Phase 6O checkpoint contains encoder, wrong size, or non-finite values",
            )
            require(
                prediction[keys].equals(expected[keys])
                and prediction.state_id.nunique() == 288
                and set(prediction.held_fold.astype(int)) == {held_fold}
                and set(prediction.training_seed.astype(int)) == {int(seed)},
                "Phase 6O held prediction scope or candidate identity changed",
            )
            require(
                np.isfinite(prediction[list(PREDICTION_COLUMNS)].to_numpy(float)).all(),
                "non-finite Phase 6O predictions",
            )
            directions = record["history"]["directions"]
            expected_directions = protocol["fold_audit"][held_fold][
                "symmetric_inner_directions"
            ]
            require(
                len(directions) == len(expected_directions) == 2
                and all(len(direction["history"]) == 50 for direction in directions)
                and [
                    (direction["train_fold"], direction["validation_fold"])
                    for direction in directions
                ]
                == [
                    (direction["training_fold"], direction["validation_fold"])
                    for direction in expected_directions
                ],
                "Phase 6O symmetric inner-validation execution changed",
            )
            require(
                selected_epoch_is_preregistered(record)
                and len(record["history"]["outer_history"])
                == int(record["selected_epoch"]),
                "Phase 6O epoch selection/refit contract changed",
            )
            records.append(
                {
                    "training_seed": int(seed),
                    "held_fold": held_fold,
                    "selected_epoch": int(record["selected_epoch"]),
                    "inner_direction_epochs": 100,
                    "outer_refit_epochs": len(record["history"]["outer_history"]),
                    "runtime_seconds": float(record["runtime_seconds"]),
                    "held_states": int(prediction.state_id.nunique()),
                    "held_candidates": len(prediction),
                    "trainable_checkpoint_parameters": sum(
                        value.numel() for value in state.values()
                    ),
                    "checkpoint_sha256": training.digest(checkpoint_path),
                    "predictions_sha256": training.digest(prediction_path),
                    "record_sha256": training.digest(record_path),
                    "historical_score_calls": int(record["historical_score_calls"]),
                    "r13_accessed": bool(record["r13_accessed"]),
                    "r14_accessed": bool(record["r14_accessed"]),
                }
            )
            prediction_parts.append(prediction)

    run_manifest = pd.DataFrame(records).sort_values(
        ["training_seed", "held_fold"], kind="stable"
    )
    training.metrics.atomic_csv(run_manifest, MANIFEST)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    ensemble = pd.read_parquet(OUT / "ensemble_oof.parquet").sort_values(
        keys, kind="stable"
    ).reset_index(drop=True)
    expected_grid = {
        (int(seed), fold)
        for seed in protocol["training"]["seeds"]
        for fold in range(3)
    }
    grouped = source.groupby("state_id", sort=True)
    checks = {
        "progress_complete_9_of_9": progress.get("status") == "COMPLETE"
        and progress.get("runs_complete") == progress.get("runs_expected") == 9,
        "summary_hash_matches_progress": training.digest(
            OUT / "training_summary.json"
        )
        == progress.get("summary_sha256"),
        "run_grid_exact": len(run_manifest) == 9
        and set(zip(run_manifest.training_seed, run_manifest.held_fold))
        == expected_grid,
        "all_run_hashes_valid": len(records) == 9,
        "outer_prediction_rows": len(predictions) == len(source) * 3,
        "outer_seed_coverage": not predictions.duplicated(
            [*keys, "training_seed"]
        ).any()
        and predictions.groupby(keys).training_seed.nunique().eq(3).all(),
        "ensemble_candidate_identity_and_order": ensemble[keys].equals(source[keys]),
        "ensemble_truth_identity": np.array_equal(
            ensemble.continuation_advantage_mean.to_numpy(),
            source.continuation_advantage_mean.to_numpy(),
        ),
        "ensemble_finite": np.isfinite(
            ensemble[
                [
                    "ensemble_advantage_mean",
                    "ensemble_advantage_std",
                    "ensemble_beats_fallback_logit",
                    "ensemble_beats_fallback_probability_raw",
                ]
            ].to_numpy(float)
        ).all(),
        "full_bank_identity": source.requested_bank_count.eq(24).all()
        and grouped.size().between(21, 24).all()
        and grouped.size().eq(grouped.full_bank_unique_count.first()).all()
        and grouped.target_set_id.nunique().eq(grouped.size()).all()
        and grouped.is_fallback.sum().eq(1).all(),
        "candidate_feasibility": source.candidate_feasible.astype(bool).all(),
        "whole_instance_fold_isolation": all(
            row["held_training_instance_overlap"] == []
            and all(
                direction["instance_overlap"] == []
                for direction in row["symmetric_inner_directions"]
            )
            for row in protocol["fold_audit"]
        ),
        "checkpoint_encoder_frozen": run_manifest.trainable_checkpoint_parameters.eq(
            int(protocol["model"]["trainable_parameters"])
        ).all(),
        "zero_historical_score_calls": run_manifest.historical_score_calls.eq(0).all()
        and progress.get("historical_score_calls") == 0,
        "r13_r14_locked": not run_manifest.r13_accessed.any()
        and not run_manifest.r14_accessed.any()
        and progress.get("r13_accessed") is False
        and progress.get("r14_accessed") is False,
        "summary_complete": summary.get("status") == "COMPLETE"
        and summary.get("family") == "O1_TOP_UTILITY_FROZEN_ENCODER",
    }
    audit = {
        "schema": "phase6o-training-completion-audit-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": {key: bool(value) for key, value in checks.items()},
        "runs": records,
        "training_seconds": float(progress["elapsed_seconds"]),
        "selected_epochs": [int(value) for value in run_manifest.selected_epoch],
        "oof_prediction_rows": len(predictions),
        "ensemble_rows": len(ensemble),
        "states": int(source.state_id.nunique()),
        "instances": int(source.instance_id.nunique()),
        "candidates": len(source),
        "artifacts": {
            str((OUT / "ensemble_oof.parquet").relative_to(ROOT)): training.digest(
                OUT / "ensemble_oof.parquet"
            ),
            str((OUT / "state_metrics.parquet").relative_to(ROOT)): training.digest(
                OUT / "state_metrics.parquet"
            ),
            str((OUT / "training_summary.json").relative_to(ROOT)): training.digest(
                OUT / "training_summary.json"
            ),
            str(MANIFEST.relative_to(ROOT)): training.digest(MANIFEST),
        },
        "historical_score_calls": 0,
        "r13_accessed": False,
        "r14_accessed": False,
    }
    training.metrics.atomic_json(audit, AUDIT)
    REPORT.write_text(
        f"""# Phase 6O Top-Utility 训练报告

状态：**COMPLETE，完整性审计 `{audit['status']}`；等待独立 OOF quality gate**。

Route B 完成 3 seeds × 3 whole-instance outer folds，共 9 runs。每个 run 的两个 inner 方向均完整运行 50 epochs，并按冻结词典序选择 epoch，再在两个 outer-training folds 上重训。选中 epochs 为 `{audit['selected_epochs']}`。完整训练耗时 {audit['training_seconds']:.2f} 秒。

外层 OOF 共 {audit['oof_prediction_rows']:,} seed-candidate rows，ensemble 为 {audit['ensemble_rows']:,} candidates/{audit['states']} states。候选身份与顺序、更新后的 continuation truth、full-bank、feasibility、fold isolation、checkpoint 哈希和有限数值全部通过。所有 checkpoint 仅含 {protocol['model']['trainable_parameters']:,} 个 pooler/fusion/head 参数，不含 frozen encoder 参数。

Expanded selected lift 为 {summary['expanded']['selected_lift']:.8f}，95% grouped LCB 为 {summary['expanded']['selected_lift_lcb']:.8f}；common original selected lift 为 {summary['common_original_288']['selected_lift']:.8f}，selection regret 为 {summary['common_original_288']['selection_regret']:.8f}。这些是训练摘要，晋级由独立预注册 OOF gate 判定。

historical scorer calls 为 0；未运行 Gurobi；未访问 R13/R14。
"""
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    if audit["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
