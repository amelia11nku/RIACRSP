#!/usr/bin/env python3
"""Audit Phase 6N N4 training completion before representation qualification."""

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

from scripts import train_phase6n_candidate_conditioned as training  # noqa: E402


OUT = ROOT / "outputs/phase6n_candidate_conditioned_csg_v1/training"
AUDIT = OUT / "completion_integrity_audit.json"
MANIFEST = OUT / "run_manifest.csv"
REPORT = ROOT / "docs/reports/phase6n_training_report.md"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    protocol = training.validate_protocol()
    protocol_sha256 = training.digest(training.PROTOCOL)
    progress = training.load_json(OUT / "progress.json")
    source = pd.read_parquet(training.SOURCE).sort_values(
        ["state_id", "target_set_id"], kind="stable"
    ).reset_index(drop=True)
    records = []
    prediction_parts = []
    inner_parts = []
    for seed in protocol["training"]["seeds"]:
        for held_fold in range(3):
            paths = training.run_paths(int(seed), held_fold)
            require(
                training.valid_run(paths, protocol_sha256),
                f"invalid Phase 6N run: seed={seed}, fold={held_fold}",
            )
            checkpoint_path, prediction_path, inner_path, record_path = paths
            record = training.load_json(record_path)
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            prediction = pd.read_parquet(prediction_path)
            inner = pd.read_parquet(inner_path)
            require(
                checkpoint.get("schema") == "phase6n-oof-checkpoint-v1"
                and checkpoint.get("training_protocol_sha256") == protocol_sha256,
                "Phase 6N checkpoint contract changed",
            )
            require(
                set(prediction.oof_fold.astype(int)) == {held_fold}
                and prediction.state_id.nunique() == 288,
                "Phase 6N outer held prediction scope changed",
            )
            expected_inner_fold = (held_fold + 1) % 3
            require(
                set(inner.oof_fold.astype(int)) == {expected_inner_fold}
                and inner.state_id.nunique() == 288,
                "Phase 6N inner validation prediction scope changed",
            )
            require(
                np.isfinite(
                    prediction[
                        [
                            "predicted_continuation_advantage",
                            "predicted_beats_fallback_logit",
                            "predicted_beats_fallback_probability_raw",
                        ]
                    ].to_numpy(dtype=float)
                ).all(),
                "non-finite Phase 6N outer predictions",
            )
            require(
                np.isfinite(
                    inner[
                        [
                            "predicted_continuation_advantage",
                            "predicted_beats_fallback_logit",
                            "predicted_beats_fallback_probability_raw",
                        ]
                    ].to_numpy(dtype=float)
                ).all(),
                "non-finite Phase 6N inner predictions",
            )
            records.append({
                "training_seed": int(seed),
                "held_fold": held_fold,
                "best_epoch": int(record["best_epoch"]),
                "inner_epochs_run": int(record["inner_epochs_run"]),
                "outer_epochs_run": int(record["outer_final_epochs_run"]),
                "runtime_seconds": float(record["runtime_seconds"]),
                "held_states": int(prediction.state_id.nunique()),
                "held_candidates": len(prediction),
                "inner_validation_fold": expected_inner_fold,
                "inner_validation_states": int(inner.state_id.nunique()),
                "checkpoint_sha256": training.digest(checkpoint_path),
                "predictions_sha256": training.digest(prediction_path),
                "inner_predictions_sha256": training.digest(inner_path),
                "record_sha256": training.digest(record_path),
                "historical_score_calls": int(record["historical_score_calls"]),
                "r13_accessed": bool(record["r13_accessed"]),
                "r14_accessed": bool(record["r14_accessed"]),
            })
            prediction_parts.append(prediction)
            inner_parts.append(inner)

    run_manifest = pd.DataFrame(records).sort_values(
        ["training_seed", "held_fold"], kind="stable"
    )
    training.metrics.atomic_csv(run_manifest, MANIFEST)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    inner_predictions = pd.concat(inner_parts, ignore_index=True)
    stored_predictions = pd.read_parquet(OUT / "oof_predictions.parquet")
    stored_inner = pd.read_parquet(OUT / "inner_validation_predictions.parquet")
    ensemble = pd.read_parquet(OUT / "ensemble_oof.parquet").sort_values(
        ["state_id", "target_set_id"], kind="stable"
    ).reset_index(drop=True)
    summary = training.load_json(OUT / "oof_summary.json")
    keys = ["state_id", "target_set_id"]
    checks = {
        "progress_complete_9_of_9": progress.get("status") == "COMPLETE"
        and progress.get("completed_runs") == progress.get("expected_runs") == 9,
        "run_grid_exact": len(run_manifest) == 9
        and set(zip(run_manifest.training_seed, run_manifest.held_fold))
        == {(seed, fold) for seed in protocol["training"]["seeds"] for fold in range(3)},
        "all_run_hashes_valid": len(records) == 9,
        "outer_prediction_rows": len(predictions) == len(source) * 3
        and len(stored_predictions) == len(predictions),
        "inner_prediction_rows": len(inner_predictions) == len(source) * 3
        and len(stored_inner) == len(inner_predictions),
        "outer_seed_coverage": not predictions.duplicated([*keys, "training_seed"]).any()
        and predictions.groupby(keys).training_seed.nunique().eq(3).all(),
        "inner_seed_coverage": not inner_predictions.duplicated(
            [*keys, "training_seed"]
        ).any()
        and inner_predictions.groupby(keys).training_seed.nunique().eq(3).all(),
        "ensemble_candidate_identity": ensemble[keys].equals(source[keys]),
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
            ].to_numpy(dtype=float)
        ).all(),
        "full_bank_and_fallback": source.groupby("state_id").size().between(21, 24).all()
        and source.groupby("state_id").is_fallback.sum().eq(1).all(),
        "candidate_feasibility": source.candidate_feasible.astype(bool).all(),
        "whole_instance_isolation": all(
            row["instance_overlap"] == [] for row in protocol["fold_audit"]
        ),
        "zero_historical_score_calls": run_manifest.historical_score_calls.eq(0).all()
        and progress.get("historical_score_calls") == 0,
        "r13_r14_locked": not run_manifest.r13_accessed.any()
        and not run_manifest.r14_accessed.any()
        and progress.get("r13_accessed") is False
        and progress.get("r14_accessed") is False,
        "summary_complete": summary.get("status") == "COMPLETE"
        and summary.get("family") == "N1_CANDIDATE_CONDITIONED_CSG",
    }
    audit = {
        "schema": "phase6n-training-completion-audit-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": {key: bool(value) for key, value in checks.items()},
        "runs": records,
        "training_seconds": float(progress["elapsed_seconds"]),
        "best_epochs": [int(value) for value in run_manifest.best_epoch],
        "oof_prediction_rows": len(predictions),
        "inner_validation_prediction_rows": len(inner_predictions),
        "ensemble_rows": len(ensemble),
        "states": int(source.state_id.nunique()),
        "instances": int(source.instance_id.nunique()),
        "artifacts": {
            str((OUT / "oof_predictions.parquet").relative_to(ROOT)): training.digest(
                OUT / "oof_predictions.parquet"
            ),
            str((OUT / "inner_validation_predictions.parquet").relative_to(ROOT)): training.digest(
                OUT / "inner_validation_predictions.parquet"
            ),
            str((OUT / "ensemble_oof.parquet").relative_to(ROOT)): training.digest(
                OUT / "ensemble_oof.parquet"
            ),
            str((OUT / "oof_summary.json").relative_to(ROOT)): training.digest(
                OUT / "oof_summary.json"
            ),
            str(MANIFEST.relative_to(ROOT)): training.digest(MANIFEST),
        },
        "historical_score_calls": 0,
        "r13_accessed": False,
        "r14_accessed": False,
    }
    training.metrics.atomic_json(audit, AUDIT)
    report = f"""# Phase 6N N4 训练报告

状态：**`{audit['status']}`**。

正式 whole-instance nested OOF 已完成 3 seeds × 3 held folds，共 9 runs。完整训练 wall time 为 {audit['training_seconds']:.2f} 秒；每个 held fold 包含 6 instances、288 states，训练侧包含 12 instances、576 states，instance overlap 为 0。best epochs 为 `{audit['best_epochs']}`。

外层 OOF 共 {audit['oof_prediction_rows']:,} seed-candidate rows，inner-validation 亦为 {audit['inner_validation_prediction_rows']:,} rows；两者对每个 state/candidate 均恰好覆盖三个 seeds。ensemble 为 {audit['ensemble_rows']:,} candidates/864 states，身份、顺序和 continuation truth 与冻结组合数据完全一致。全部 checkpoint、outer prediction、inner-validation prediction 与 run record 哈希通过。

Expanded OOF：Spearman {summary['expanded']['spearman']:.6f}，pairwise accuracy {summary['expanded']['pairwise_accuracy']:.6f}，NDCG@1 {summary['expanded']['ndcg_at_1']:.6f}，raw selected lift {summary['expanded']['selected_lift']:.6f}，grouped-bootstrap LCB {summary['expanded']['selected_lift_lcb']:.6f}。

Common original 288：Spearman {summary['common_original_288']['spearman']:.6f}，pairwise accuracy {summary['common_original_288']['pairwise_accuracy']:.6f}，NDCG@1 {summary['common_original_288']['ndcg_at_1']:.6f}，raw selected lift {summary['common_original_288']['selected_lift']:.6f}，LCB {summary['common_original_288']['selected_lift_lcb']:.6f}。

历史 scorer calls 为 0；所有候选标签均为 feasible；R13/R14 未访问。N4 仅证明训练和 OOF 工件完整，是否进入 calibration 由独立 N5 raw representation gate 决定。
"""
    REPORT.write_text(report)
    print(json.dumps(audit, indent=2, sort_keys=True))
    if audit["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
