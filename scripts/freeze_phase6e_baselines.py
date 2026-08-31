#!/usr/bin/env python3
"""Freeze validation-selected B2 and TRAIN-only Phase 6C B3 before holdout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rcias_clgri.ni.baselines import (  # noqa: E402
    Phase6CTabularDiagnostic,
    choose_best_fixed_original,
)
from rcias_clgri.ni.cache import file_sha256  # noqa: E402


AGGREGATE_COLUMNS = [
    "state_id", "target_set_id", "training_split", "scale", "CF_level",
    "RI_level", "TI_level", "search_stage", "bottleneck_proxy",
    "current_makespan", "arm_family", "origin_destroy_operator",
    "origin_rules", "destroy_count", "mean_relative_improvement", "regret_to_best",
    "positive_under_2_of_3",
]


def load_split(dataset_root: Path, directory: str) -> pd.DataFrame:
    paths = sorted(dataset_root.glob(f"{directory}/*/target_set_aggregates.parquet"))
    if not paths:
        raise ValueError(f"no Phase 6C aggregates found for {directory}")
    return pd.concat(
        [pd.read_parquet(path, columns=AGGREGATE_COLUMNS) for path in paths],
        ignore_index=True,
    )


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root", type=Path, default=ROOT / "outputs/phase6c/dataset"
    )
    parser.add_argument(
        "--target-features",
        type=Path,
        default=ROOT / "outputs/phase6c/diagnostics/target_set_preaction_features.parquet",
    )
    parser.add_argument(
        "--evaluation-config",
        type=Path,
        default=ROOT / "configs/phase6e_evaluation.json",
    )
    parser.add_argument(
        "--ablation-audit",
        type=Path,
        default=ROOT / "outputs/phase6e/audit/ablation_training_audit.json",
    )
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "outputs/phase6e/baselines"
    )
    parser.add_argument(
        "--freeze-output",
        type=Path,
        default=ROOT / "outputs/phase6e/audit/baseline_freeze.json",
    )
    args = parser.parse_args()

    audit = json.loads(args.ablation_audit.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS" or audit.get("internal_holdout_metrics_observed") is not False:
        raise ValueError("ablation audit must pass before baseline freeze")
    if args.freeze_output.exists():
        existing = json.loads(args.freeze_output.read_text(encoding="utf-8"))
        if existing.get("status") == "FROZEN_BEFORE_INTERNAL_HOLDOUT":
            print(json.dumps({"event": "baseline_freeze_exists", **existing}))
            return

    print(json.dumps({"event": "load_validation_aggregates"}), flush=True)
    validation = load_split(args.dataset_root, "validation")
    best_operator, operator_summary = choose_best_fixed_original(validation)
    args.output_root.mkdir(parents=True, exist_ok=True)
    operator_summary.to_csv(args.output_root / "validation_fixed_operator_summary.csv", index=False)

    print(json.dumps({"event": "sample_train_for_b3"}), flush=True)
    train = load_split(args.dataset_root, "train")
    train_sample = train.sample(n=min(250_000, len(train)), random_state=670001)
    needed_ids = pd.concat([
        train_sample[["state_id", "target_set_id"]],
        validation[["state_id", "target_set_id"]],
    ], ignore_index=True)
    features = pd.read_parquet(args.target_features)
    selected_features = features.merge(
        needed_ids, on=["state_id", "target_set_id"], how="inner", validate="one_to_one"
    )
    del features
    modeled_train = train_sample.merge(
        selected_features,
        on=["state_id", "target_set_id"],
        how="inner",
        validate="one_to_one",
    )
    if len(modeled_train) != len(train_sample):
        raise ValueError("B3 train feature coverage mismatch")
    model = Phase6CTabularDiagnostic(sample_limit=250_000, random_state=670001)
    model.fit(modeled_train)
    model_path = args.output_root / "phase6c_tabular_diagnostic.joblib"
    temporary_model = model_path.with_suffix(model_path.suffix + ".partial")
    joblib.dump(model, temporary_model)
    temporary_model.replace(model_path)

    modeled_validation = validation.merge(
        selected_features,
        on=["state_id", "target_set_id"],
        how="inner",
        validate="one_to_one",
    )
    if len(modeled_validation) != len(validation):
        raise ValueError("B3 validation feature coverage mismatch")
    validation_score = model.score(modeled_validation)
    robust_label = modeled_validation["positive_under_2_of_3"].astype(int)
    validation_metrics = {
        "robust_label_roc_auc": float(roc_auc_score(robust_label, validation_score)),
        "robust_label_pr_auc": float(average_precision_score(robust_label, validation_score)),
        "action_count": len(modeled_validation),
        "state_count": int(modeled_validation["state_id"].nunique()),
    }
    published = pd.read_csv(ROOT / "outputs/phase6c/summaries/predictability_summary.csv")
    published_row = published[
        published["model"].eq("SET_STRUCTURE_CONTEXT_LOGISTIC")
        & published["evaluation_split"].eq("TRAIN_VALIDATION")
        & published["regime_dimension"].eq("ALL")
    ].iloc[0]
    validation_metrics["published_roc_auc"] = float(published_row["roc_auc"])
    validation_metrics["published_pr_auc"] = float(published_row["pr_auc"])
    validation_metrics["matches_published_phase6c"] = bool(
        abs(validation_metrics["robust_label_roc_auc"] - validation_metrics["published_roc_auc"]) < 1e-10
        and abs(validation_metrics["robust_label_pr_auc"] - validation_metrics["published_pr_auc"]) < 1e-10
    )
    if not validation_metrics["matches_published_phase6c"]:
        raise ValueError("B3 reproduction does not match frozen Phase 6C validation metrics")

    freeze = {
        "schema": "phase6e-baseline-freeze-v1",
        "status": "FROZEN_BEFORE_INTERNAL_HOLDOUT",
        "B2_best_fixed_original_operator": best_operator,
        "B2_selection_split": "TRAIN_VALIDATION",
        "B3_training_split": "TRAIN",
        "B3_training_label": model.training_label,
        "B3_training_sample_count": model.training_sample_count,
        "B3_training_sample_sha256": model.training_sample_sha256,
        "B3_feature_columns": model.feature_columns,
        "B3_model_path": str(model_path.resolve()),
        "B3_model_sha256": file_sha256(model_path),
        "B3_validation_reproduction": validation_metrics,
        "evaluation_plan_sha256": file_sha256(args.evaluation_config),
        "ablation_training_audit_sha256": file_sha256(args.ablation_audit),
        "internal_holdout_metrics_observed": False,
    }
    write_json(args.freeze_output, freeze)
    print(json.dumps({"event": "baseline_freeze_complete", **freeze}), flush=True)


if __name__ == "__main__":
    main()
