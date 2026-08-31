#!/usr/bin/env python3
"""Audit frozen Phase 6E controlled-model training before holdout access."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_VARIANTS = {"FLAT_SET", "STATIC_CSG", "NO_EDGE_FEATURES"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training-root",
        type=Path,
        default=ROOT / "outputs/phase6e/ablations/training",
    )
    parser.add_argument(
        "--evaluation-config",
        type=Path,
        default=ROOT / "configs/phase6e_evaluation.json",
    )
    parser.add_argument(
        "--final-training-audit",
        type=Path,
        default=ROOT / "outputs/phase6e/audit/final_training_audit.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/phase6e/audit/ablation_training_audit.json",
    )
    args = parser.parse_args()

    completion_path = args.training_root / "ablation_training_completion.json"
    manifest_path = args.training_root / "checkpoint_manifest.json"
    summary_path = args.training_root / "ablation_training_summary.csv"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = pd.read_csv(summary_path)
    final_audit = json.loads(args.final_training_audit.read_text(encoding="utf-8"))
    checkpoint_hashes = {
        str(record["variant"]): file_sha256(Path(str(record["checkpoint_path"])))
        for record in manifest["checkpoints"]
    }
    recorded_hashes = {
        str(record["variant"]): str(record["checkpoint_sha256"])
        for record in manifest["checkpoints"]
    }
    variants = set(summary["variant"].astype(str))
    metric_columns = [
        "validation_roc_auc", "validation_pr_auc",
        "validation_pairwise_accuracy", "validation_within_state_spearman",
        "validation_ndcg", "validation_mean_selected_utility",
        "validation_mean_selected_regret", "validation_objective",
    ]
    checks = {
        "completion_marker_complete": completion.get("status") == "COMPLETE",
        "checkpoint_manifest_complete": manifest.get("status") == "COMPLETE",
        "exact_controlled_variants": variants == EXPECTED_VARIANTS,
        "one_checkpoint_per_variant": {
            str(record["variant"]) for record in manifest["checkpoints"]
        } == EXPECTED_VARIANTS and len(manifest["checkpoints"]) == 3,
        "all_training_summaries_complete": summary["status"].eq("COMPLETE").all(),
        "all_validation_states_scored": summary["validation_state_count"].eq(20_000).all(),
        "all_validation_actions_scored": summary["validation_action_count"].eq(472_685).all(),
        "all_metrics_finite": all(
            math.isfinite(float(value))
            for column in metric_columns for value in summary[column]
        ),
        "all_checkpoint_hashes_exact": checkpoint_hashes == recorded_hashes,
        "evaluation_plan_unchanged": manifest.get("evaluation_plan_sha256")
        == file_sha256(args.evaluation_config),
        "final_training_audit_passed": final_audit.get("status") == "PASS",
        "holdout_not_observed": completion.get("internal_holdout_metrics_observed") is False
        and manifest.get("internal_holdout_metrics_observed") is False,
    }
    result = {
        "schema": "phase6e-ablation-training-audit-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": {key: bool(value) for key, value in checks.items()},
        "variants": sorted(variants),
        "checkpoint_manifest_sha256": file_sha256(manifest_path),
        "completion_marker_sha256": file_sha256(completion_path),
        "evaluation_plan_sha256": file_sha256(args.evaluation_config),
        "final_training_audit_sha256": file_sha256(args.final_training_audit),
        "internal_holdout_metrics_observed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
