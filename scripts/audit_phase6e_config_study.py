#!/usr/bin/env python3
"""Audit Phase 6E config selection before final-seed training."""

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
        "--study-root",
        type=Path,
        default=ROOT / "outputs/phase6e/training/config_study",
    )
    parser.add_argument(
        "--freeze",
        type=Path,
        default=ROOT / "outputs/phase6e/audit/experiment_freeze.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/phase6e/audit/config_study_audit.json",
    )
    args = parser.parse_args()
    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    candidate_summary = pd.read_csv(args.study_root / "candidate_summary.csv")
    validation = pd.read_csv(args.study_root / "validation_metrics.csv")
    candidates = candidate_summary["candidate"].tolist()
    launch_records = [
        json.loads((args.study_root / candidate / "launch_record.json").read_text())
        for candidate in candidates
    ]
    best_from_history = (
        validation.groupby("candidate")["validation_objective"].max().to_dict()
    )
    best_from_summary = candidate_summary.set_index("candidate")[
        "best_validation_objective"
    ].to_dict()
    selected = str(
        candidate_summary.sort_values(
            ["best_validation_objective", "parameter_count", "candidate"],
            ascending=[False, True, True],
            kind="stable",
        ).iloc[0]["candidate"]
    )
    finite_columns = [
        "validation_pairwise_accuracy", "validation_ndcg", "validation_roc_auc",
        "validation_pr_auc", "validation_mean_selected_utility",
        "validation_mean_selected_regret", "validation_objective",
    ]
    checks = {
        "study_complete": json.loads(
            (args.study_root / "study_summary.json").read_text()
        )["status"] == "COMPLETE",
        "exact_candidate_count": len(candidate_summary) == 3,
        "three_epochs_per_candidate": validation.groupby("candidate").size().eq(3).all(),
        "single_development_seed": {int(record["seed"]) for record in launch_records} == {660150},
        "full_train_coverage": all(int(record["train_states"]) == 60_000 for record in launch_records),
        "full_validation_coverage": all(
            int(record["validation_states"]) == 20_000 for record in launch_records
        ),
        "no_holdout_shards_loaded": all(
            int(record["internal_holdout_shards_loaded"]) == 0 for record in launch_records
        ),
        "validation_all_actions_scored": validation["validation_action_count"].eq(472_685).all(),
        "validation_all_states_scored": validation["validation_state_count"].eq(20_000).all(),
        "core_metrics_finite": all(
            math.isfinite(float(value))
            for column in finite_columns for value in validation[column]
        ),
        "summary_matches_history": all(
            abs(best_from_history[name] - best_from_summary[name]) < 1e-12
            for name in candidates
        ),
        "selected_candidate_is_objective_max": freeze["selected_candidate"] == selected,
        "freeze_precedes_holdout": (
            freeze["status"] == "FROZEN_BEFORE_INTERNAL_HOLDOUT"
            and freeze["internal_holdout_metrics_observed"] is False
            and freeze["internal_holdout_checkpoint_selection"] is False
        ),
        "all_best_checkpoints_exist": all(
            (args.study_root / candidate / "checkpoint_best.pt").exists()
            for candidate in candidates
        ),
    }
    result = {
        "schema": "phase6e-config-study-audit-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": {key: bool(value) for key, value in checks.items()},
        "selected_candidate": selected,
        "selected_validation_objective": float(best_from_summary[selected]),
        "candidate_objectives": best_from_summary,
        "experiment_freeze_sha256": file_sha256(args.freeze),
        "internal_holdout_metrics_observed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
