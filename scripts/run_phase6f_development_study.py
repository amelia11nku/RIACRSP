#!/usr/bin/env python3
"""Run the pre-registered Phase 6F TRAIN/TRAIN_VALIDATION study."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "phase6f_revision.json"
RUN_ROOT = ROOT / "outputs" / "phase6f" / "development" / "runs"
SANITY = ROOT / "outputs" / "phase6f" / "audit" / "mandatory_sanity.json"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_name(objective: str, model: str, weight: float, seed: int) -> str:
    return f"{objective}__{model}__D{weight:g}__seed{seed}"


def stream_command(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        return_code = process.wait()
    if return_code:
        raise RuntimeError(f"candidate training failed with return code {return_code}")


def ensure_run(
    *, objective: str, model: str, weight: float, seed: int, epochs: int, device: str
) -> Path:
    name = run_name(objective, model, weight, seed)
    directory = RUN_ROOT / name
    summary_path = directory / "training_summary.json"
    valid = False
    if summary_path.exists() and (directory / "checkpoint_best.pt").exists():
        launch = read_json(directory / "launch_record.json")
        summary = read_json(summary_path)
        valid = (
            summary.get("status") == "COMPLETE"
            and launch.get("objective") == objective
            and launch.get("model_candidate") == model
            and float(launch.get("distillation_weight")) == weight
            and int(launch.get("seed")) == seed
            and int(launch["training"]["epochs"]) == epochs
            and not launch.get("train_internal_holdout_accessed")
            and not launch.get("revision_holdout_accessed")
        )
    if not valid:
        command = [
            sys.executable,
            str(ROOT / "scripts" / "train_phase6f_revision.py"),
            "--objective", objective,
            "--model", model,
            "--distillation-weight", str(weight),
            "--seed", str(seed),
            "--epochs", str(epochs),
            "--run-name", name,
            "--output-root", str(RUN_ROOT),
            "--device", device,
        ]
        checkpoint = directory / "checkpoint_last.pt"
        if checkpoint.exists():
            command.extend(["--resume", str(checkpoint)])
        stream_command(command, RUN_ROOT / f"{name}.log")
    return directory


def summarize_run(directory: Path) -> dict[str, object]:
    summary = read_json(directory / "training_summary.json")
    history = pd.read_csv(directory / "training_history.csv")
    best_epoch = int(summary["best_epoch"])
    row = history[history["epoch"].eq(best_epoch)].iloc[0]
    launch = read_json(directory / "launch_record.json")
    return {
        "objective": launch["objective"],
        "model_candidate": launch["model_candidate"],
        "distillation_weight": float(launch["distillation_weight"]),
        "seed": int(launch["seed"]),
        "best_epoch": best_epoch,
        "parameter_count": int(summary["parameter_count"]),
        "mean_selected_utility": float(row["validation_mean_selected_utility"]),
        "mean_selected_regret": float(row["validation_mean_selected_regret"]),
        "selected_positive_fraction": float(
            row["validation_selected_positive_fraction"]
        ),
        "selected_top3_fraction": float(row["validation_selected_top3_fraction"]),
        "pairwise_accuracy": float(row["validation_pairwise_accuracy"]),
        "ndcg": float(row["validation_ndcg"]),
        "checkpoint_path": str((directory / "checkpoint_best.pt").resolve()),
        "checkpoint_sha256": sha256_file(directory / "checkpoint_best.pt"),
        "validation_scores_path": str(
            (directory / f"validation_scores_epoch_{best_epoch:02d}.parquet").resolve()
        ),
        "train_internal_holdout_accessed": False,
        "revision_holdout_accessed": False,
    }


def select(frame: pd.DataFrame) -> pd.Series:
    return frame.sort_values(
        ["mean_selected_utility", "mean_selected_regret", "selected_positive_fraction",
         "parameter_count", "objective", "model_candidate", "distillation_weight"],
        ascending=[False, True, False, True, True, True, True],
        kind="stable",
    ).iloc[0]


def calibrate_selected(record: pd.Series, config: dict) -> dict[str, object]:
    from rcias_clgri.ni.calibration import (
        calibration_metrics,
        fit_probability_calibrator,
        fit_utility_calibrator,
        reliability_table,
    )
    from rcias_clgri.ni.metrics import evaluate_action_scores
    from rcias_clgri.ni.selective_policy import threshold_study

    scores = pd.read_parquet(record["validation_scores_path"])
    if "origin_rules" not in scores:
        metadata_paths = sorted(
            (ROOT / "outputs/phase6c/dataset/validation").glob(
                "*/target_set_aggregates.parquet"
            )
        )
        metadata = pd.concat([
            pd.read_parquet(
                path,
                columns=["state_id", "target_set_id", "origin_rules", "origin_families"],
            )
            for path in metadata_paths
        ], ignore_index=True)
        scores = scores.merge(
            metadata,
            on=["state_id", "target_set_id"],
            how="inner",
            validate="one_to_one",
        )
        if len(scores) != len(metadata):
            raise ValueError("TRAIN_VALIDATION origin metadata coverage mismatch")
    positive = scores["mean_relative_improvement"].gt(0).astype(int)
    raw_metrics = evaluate_action_scores(scores)
    rows = []
    models = {}
    calibrated_frames = {}
    for method in config["calibration"]["candidate_methods"]:
        model = fit_probability_calibrator(scores["score"], positive, method)
        calibrated = scores.copy()
        calibrated["calibrated_probability"] = model.predict(scores["score"])
        rank_metrics = evaluate_action_scores(calibrated, score="calibrated_probability")
        metrics = calibration_metrics(
            calibrated["calibrated_probability"], positive,
            bins=int(config["calibration"]["ece_bins"]),
        )
        ranking_preserved = (
            rank_metrics["mean_selected_utility"]
            >= raw_metrics["mean_selected_utility"] - 1e-12
        )
        rows.append({
            "method": method,
            **metrics,
            "mean_selected_utility": rank_metrics["mean_selected_utility"],
            "ranking_preserved": ranking_preserved,
        })
        models[method] = model
        calibrated_frames[method] = calibrated
    calibration = pd.DataFrame(rows)
    eligible = calibration[calibration["ranking_preserved"]]
    if eligible.empty:
        raise RuntimeError("no probability calibrator preserves validation ranking")
    selected_method = str(eligible.sort_values(
        ["brier_score", "expected_calibration_error", "method"],
        kind="stable",
    ).iloc[0]["method"])
    calibration["selected"] = calibration["method"].eq(selected_method)
    output_root = ROOT / "outputs" / "phase6f" / "calibration"
    output_root.mkdir(parents=True, exist_ok=True)
    calibration.to_csv(output_root / "calibration_summary.csv", index=False)

    selected_frame = calibrated_frames[selected_method]
    utility_predictor = (
        selected_frame["predicted_utility"]
        if "predicted_utility" in selected_frame else selected_frame["score"]
    )
    utility_model = fit_utility_calibrator(
        utility_predictor, selected_frame["mean_relative_improvement"]
    )
    selected_frame["calibrated_utility"] = utility_model.predict(utility_predictor)
    reliability_table(
        selected_frame["calibrated_probability"], positive,
        bins=int(config["calibration"]["ece_bins"]),
    ).to_csv(output_root / "reliability_bins.csv", index=False)
    selected_frame.to_parquet(output_root / "selected_validation_scores.parquet", index=False)

    policy, thresholds = threshold_study(
        selected_frame,
        minimum_coverage=float(config["calibration"]["minimum_intervention_coverage"]),
    )
    policy.to_csv(output_root / "selective_policy_summary.csv", index=False)
    selected_policy = policy[policy["selected"]].iloc[0].to_dict()
    calibration_record = {
        "schema": "phase6f-development-calibration-v1",
        "status": "COMPLETE",
        "fit_split": "TRAIN_VALIDATION",
        "selected_probability_method": selected_method,
        "probability_calibrator": models[selected_method].to_dict(),
        "utility_calibrator": utility_model.to_dict(),
        "thresholds": {
            "confidence": thresholds.confidence,
            "predicted_utility": thresholds.predicted_utility,
            "decision_margin": thresholds.decision_margin,
        },
        "selected_policy_metrics": selected_policy,
        "minimum_coverage": config["calibration"]["minimum_intervention_coverage"],
        "fallback": config["calibration"]["fallback_primary"],
        "train_internal_holdout_accessed": False,
        "revision_holdout_accessed": False,
    }
    (output_root / "development_calibration.json").write_text(
        json.dumps(calibration_record, indent=2) + "\n", encoding="utf-8"
    )
    return calibration_record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = read_json(CONFIG)
    sanity = read_json(SANITY)
    if sanity.get("status") != "PASS":
        raise ValueError("Phase 6F mandatory sanity tests must pass before model comparison")
    protocol = config["development_protocol"]
    seed = int(protocol["study_seed"])
    objective_epochs = int(protocol["objective_study_epochs"])
    compact_epochs = int(protocol["compact_study_epochs"])
    objectives = list(config["objective_candidates"])
    models = list(config["compact_model_candidates"])
    if len(objectives) > 3 or len(models) > 3:
        raise ValueError("Phase 6F candidate budget exceeded")
    # The smallest frozen candidate is the deterministic objective-study anchor.
    anchor = "C2_H96_L2"
    objective_rows = []
    for objective in objectives:
        directory = ensure_run(
            objective=objective, model=anchor, weight=0.0, seed=seed,
            epochs=objective_epochs, device=args.device,
        )
        objective_rows.append(summarize_run(directory))
    objective_summary = pd.DataFrame(objective_rows)
    selected_objective = str(select(objective_summary)["objective"])
    objective_root = ROOT / "outputs" / "phase6f" / "objectives"
    objective_root.mkdir(parents=True, exist_ok=True)
    objective_summary.assign(
        selected=objective_summary["objective"].eq(selected_objective)
    ).to_csv(objective_root / "objective_validation_summary.csv", index=False)

    compact_rows = []
    for model in models:
        directory = ensure_run(
            objective=selected_objective, model=model, weight=0.0, seed=seed,
            epochs=compact_epochs, device=args.device,
        )
        compact_rows.append(summarize_run(directory))
    compact_summary = pd.DataFrame(compact_rows)
    selected_model = str(select(compact_summary)["model_candidate"])
    compact_root = ROOT / "outputs" / "phase6f" / "compact_models"
    compact_root.mkdir(parents=True, exist_ok=True)
    compact_summary.assign(
        selected=compact_summary["model_candidate"].eq(selected_model)
    ).to_csv(compact_root / "compact_model_validation_summary.csv", index=False)

    distillation_rows = []
    for weight in config["distillation"]["candidate_weights"]:
        directory = ensure_run(
            objective=selected_objective, model=selected_model, weight=float(weight),
            seed=seed, epochs=compact_epochs, device=args.device,
        )
        distillation_rows.append(summarize_run(directory))
    distillation_summary = pd.DataFrame(distillation_rows)
    selected_distillation = float(select(distillation_summary)["distillation_weight"])
    distillation_root = ROOT / "outputs" / "phase6f" / "distillation"
    distillation_root.mkdir(parents=True, exist_ok=True)
    distillation_summary.assign(
        selected=np.isclose(
            distillation_summary["distillation_weight"], selected_distillation
        )
    ).to_csv(distillation_root / "distillation_summary.csv", index=False)

    selected_run = select(distillation_summary)
    calibration = calibrate_selected(selected_run, config)
    summary = {
        "schema": "phase6f-development-study-summary-v1",
        "status": "COMPLETE",
        "objective_candidate_count": len(objectives),
        "compact_model_candidate_count": len(models),
        "distillation_candidate_count": len(distillation_summary),
        "selected_objective": selected_objective,
        "selected_model_candidate": selected_model,
        "selected_distillation_weight": selected_distillation,
        "selected_development_checkpoint": selected_run["checkpoint_path"],
        "selected_development_checkpoint_sha256": selected_run["checkpoint_sha256"],
        "selected_probability_calibration_method": calibration[
            "selected_probability_method"
        ],
        "selection_split": "TRAIN_VALIDATION",
        "train_internal_holdout_accessed": False,
        "revision_holdout_accessed": False,
        "phase6f_config_sha256": sha256_file(CONFIG),
    }
    output = ROOT / "outputs" / "phase6f" / "development" / "study_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "phase6f_development_study_complete", **summary}))


if __name__ == "__main__":
    main()
