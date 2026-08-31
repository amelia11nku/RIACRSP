#!/usr/bin/env python3
"""Audit three final Phase 6E seeds before controlled evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


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
        default=ROOT / "outputs/phase6e/training/final_seeds",
    )
    parser.add_argument(
        "--experiment-freeze",
        type=Path,
        default=ROOT / "outputs/phase6e/audit/experiment_freeze.json",
    )
    parser.add_argument(
        "--evaluation-config",
        type=Path,
        default=ROOT / "configs/phase6e_evaluation.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/phase6e/audit/final_training_audit.json",
    )
    args = parser.parse_args()
    summary = json.loads(
        (args.training_root / "final_seeds_summary.json").read_text(encoding="utf-8")
    )
    checkpoint_manifest_path = args.training_root / "checkpoint_manifest.json"
    manifest = json.loads(checkpoint_manifest_path.read_text(encoding="utf-8"))
    seed_summary = pd.read_csv(args.training_root / "model_seed_summary.csv")
    validation = pd.read_csv(args.training_root / "validation_metrics.csv")
    expected_seeds = {660201, 660202, 660203}
    checkpoint_hashes = {
        int(record["seed"]): file_sha256(Path(record["checkpoint_path"]))
        for record in manifest["checkpoints"]
    }
    recorded_hashes = {
        int(record["seed"]): record["checkpoint_sha256"]
        for record in manifest["checkpoints"]
    }
    core = [
        "validation_roc_auc", "validation_pr_auc", "validation_pairwise_accuracy",
        "validation_within_state_spearman", "validation_ndcg",
        "validation_mean_selected_utility", "validation_mean_selected_regret",
    ]
    checks = {
        "final_summary_complete": summary["status"] == "COMPLETE",
        "exact_three_seeds": set(seed_summary["seed"].astype(int)) == expected_seeds,
        "six_epochs_per_seed": validation.groupby("seed").size().eq(6).all(),
        "all_best_epochs_valid": seed_summary["best_epoch"].between(1, 6).all(),
        "common_config_fingerprint": seed_summary["config_fingerprint"].nunique() == 1,
        "all_validation_states_scored": seed_summary["validation_state_count"].eq(20_000).all(),
        "all_validation_actions_scored": seed_summary["validation_action_count"].eq(472_685).all(),
        "all_metrics_finite": all(
            math.isfinite(float(value)) for column in core for value in seed_summary[column]
        ),
        "all_checkpoint_hashes_exact": checkpoint_hashes == recorded_hashes,
        "experiment_freeze_unchanged": manifest["experiment_freeze_sha256"] == file_sha256(
            args.experiment_freeze
        ),
        "holdout_not_observed": summary["internal_holdout_metrics_observed"] is False,
    }
    stability = {
        column: {
            "mean": float(seed_summary[column].mean()),
            "std": float(seed_summary[column].std(ddof=1)),
        }
        for column in core
    }
    result = {
        "schema": "phase6e-final-training-audit-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": {key: bool(value) for key, value in checks.items()},
        "seeds": sorted(expected_seeds),
        "selected_candidate": summary["selected_candidate"],
        "config_fingerprint": str(seed_summary.iloc[0]["config_fingerprint"]),
        "stability": stability,
        "checkpoint_manifest_sha256": file_sha256(checkpoint_manifest_path),
        "experiment_freeze_sha256": file_sha256(args.experiment_freeze),
        "evaluation_plan_sha256": file_sha256(args.evaluation_config),
        "internal_holdout_metrics_observed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
