#!/usr/bin/env python3
"""Train three final Phase 6F seeds and freeze one deployment checkpoint."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "phase6f_revision.json"
DEVELOPMENT = ROOT / "outputs" / "phase6f" / "development" / "study_summary.json"
DEVELOPMENT_CALIBRATION = (
    ROOT / "outputs" / "phase6f" / "calibration" / "development_calibration.json"
)
OUTPUT_ROOT = ROOT / "outputs" / "phase6f" / "training" / "final_seeds"
EXPERIMENT_FREEZE = ROOT / "outputs" / "phase6f" / "audit" / "experiment_freeze.json"
SEALED_LABEL_FREEZE = (
    ROOT / "outputs" / "phase6f" / "revision_holdout" / "sealed_label_freeze.json"
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


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
        raise RuntimeError(f"final-seed training failed with return code {return_code}")


def ensure_seed(
    *,
    seed: int,
    objective: str,
    model: str,
    distillation_weight: float,
    epochs: int,
    device: str,
) -> Path:
    directory = OUTPUT_ROOT / f"seed_{seed}"
    summary_path = directory / "training_summary.json"
    valid = False
    if summary_path.exists() and (directory / "checkpoint_best.pt").exists():
        launch = read_json(directory / "launch_record.json")
        summary = read_json(summary_path)
        valid = (
            summary.get("status") == "COMPLETE"
            and launch.get("objective") == objective
            and launch.get("model_candidate") == model
            and float(launch.get("distillation_weight")) == distillation_weight
            and int(launch.get("seed")) == seed
            and int(launch["training"]["epochs"]) == epochs
            and not launch.get("revision_holdout_accessed")
        )
    if not valid:
        command = [
            sys.executable,
            str(ROOT / "scripts" / "train_phase6f_revision.py"),
            "--objective", objective,
            "--model", model,
            "--distillation-weight", str(distillation_weight),
            "--seed", str(seed),
            "--epochs", str(epochs),
            "--run-name", f"seed_{seed}",
            "--output-root", str(OUTPUT_ROOT),
            "--device", device,
        ]
        checkpoint = directory / "checkpoint_last.pt"
        if checkpoint.exists():
            command.extend(["--resume", str(checkpoint)])
        stream_command(command, OUTPUT_ROOT / f"seed_{seed}.log")
    return directory


def calibrate_seed(
    directory: Path,
    *,
    seed: int,
    calibration_method: str,
    config: dict,
) -> dict[str, object]:
    from rcias_clgri.ni.calibration import (
        calibration_metrics,
        fit_probability_calibrator,
        fit_utility_calibrator,
    )
    from rcias_clgri.ni.selective_policy import threshold_study

    summary = read_json(directory / "training_summary.json")
    history = pd.read_csv(directory / "training_history.csv")
    best_epoch = int(summary["best_epoch"])
    row = history[history["epoch"].eq(best_epoch)].iloc[0]
    score_path = directory / f"validation_scores_epoch_{best_epoch:02d}.parquet"
    frame = pd.read_parquet(score_path)
    required = {"origin_rules", "origin_families", "predicted_utility"}
    if not required.issubset(frame.columns):
        raise ValueError(f"final seed {seed} validation metadata is incomplete")
    positive = frame["mean_relative_improvement"].gt(0).astype(int)
    probability_model = fit_probability_calibrator(
        frame["score"], positive, calibration_method
    )
    frame["calibrated_probability"] = probability_model.predict(frame["score"])
    utility_model = fit_utility_calibrator(
        frame["predicted_utility"], frame["mean_relative_improvement"]
    )
    frame["calibrated_utility"] = utility_model.predict(frame["predicted_utility"])
    policy, thresholds = threshold_study(
        frame,
        minimum_coverage=float(config["calibration"]["minimum_intervention_coverage"]),
    )
    selected = policy[policy["selected"]].iloc[0]
    calibration = calibration_metrics(
        frame["calibrated_probability"], positive,
        bins=int(config["calibration"]["ece_bins"]),
    )
    record = {
        "schema": "phase6f-final-seed-validation-v1",
        "status": "COMPLETE",
        "seed": seed,
        "best_epoch": best_epoch,
        "checkpoint_path": str((directory / "checkpoint_best.pt").resolve()),
        "checkpoint_sha256": sha256_file(directory / "checkpoint_best.pt"),
        "parameter_count": int(summary["parameter_count"]),
        "raw_mean_selected_utility": float(row["validation_mean_selected_utility"]),
        "raw_mean_selected_regret": float(row["validation_mean_selected_regret"]),
        "raw_selected_positive_fraction": float(
            row["validation_selected_positive_fraction"]
        ),
        "hybrid_selected_utility": float(selected["hybrid_selected_utility"]),
        "hybrid_mean_regret": float(selected["hybrid_mean_regret"]),
        "hybrid_selected_positive_fraction": float(
            selected["hybrid_selected_positive_fraction"]
        ),
        "intervention_coverage": float(selected["coverage"]),
        "incremental_utility_vs_fallback": float(
            selected["incremental_utility_vs_fallback"]
        ),
        "probability_calibration": calibration,
        "probability_calibrator": probability_model.to_dict(),
        "utility_calibrator": utility_model.to_dict(),
        "thresholds": {
            "confidence": thresholds.confidence,
            "predicted_utility": thresholds.predicted_utility,
            "decision_margin": thresholds.decision_margin,
        },
        "selection_split": "TRAIN_VALIDATION",
        "revision_holdout_accessed": False,
    }
    (directory / "validation_calibration.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    policy.to_csv(directory / "selective_policy_summary.csv", index=False)
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = read_json(CONFIG)
    development = read_json(DEVELOPMENT)
    development_calibration = read_json(DEVELOPMENT_CALIBRATION)
    sealed = read_json(SEALED_LABEL_FREEZE)
    if development.get("status") != "COMPLETE":
        raise ValueError("Phase 6F development study is incomplete")
    if sealed.get("status") != "SEALED_COMPLETE" or sealed.get(
        "label_files_opened_for_analysis"
    ):
        raise ValueError("R06 labels must remain sealed during final-seed selection")

    objective = str(development["selected_objective"])
    model = str(development["selected_model_candidate"])
    distillation_weight = float(development["selected_distillation_weight"])
    calibration_method = str(development["selected_probability_calibration_method"])
    epochs = int(config["final_training"]["epochs"])
    records = []
    for seed in config["final_training"]["seeds"]:
        directory = ensure_seed(
            seed=int(seed),
            objective=objective,
            model=model,
            distillation_weight=distillation_weight,
            epochs=epochs,
            device=args.device,
        )
        records.append(calibrate_seed(
            directory,
            seed=int(seed),
            calibration_method=calibration_method,
            config=config,
        ))
        print(json.dumps({
            "event": "phase6f_final_seed_complete",
            "seed": seed,
            "hybrid_selected_utility": records[-1]["hybrid_selected_utility"],
            "hybrid_mean_regret": records[-1]["hybrid_mean_regret"],
        }), flush=True)

    summary = pd.DataFrame([{key: value for key, value in record.items() if not isinstance(value, dict)}
                            for record in records])
    summary = summary.sort_values(
        ["hybrid_selected_utility", "hybrid_mean_regret", "seed"],
        ascending=[False, True, True],
        kind="stable",
    ).reset_index(drop=True)
    selected_seed = int(summary.iloc[0]["seed"])
    summary["selected_deployment_seed"] = summary["seed"].eq(selected_seed)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_ROOT / "final_seed_validation_summary.csv", index=False)
    selected = next(record for record in records if record["seed"] == selected_seed)
    manifest = {
        "schema": "phase6f-final-checkpoint-manifest-v1",
        "status": "COMPLETE",
        "selection_split": "TRAIN_VALIDATION",
        "selection_rule": config["final_training"]["deployment_seed_rule"],
        "selected_seed": selected_seed,
        "checkpoints": [
            {
                "seed": record["seed"],
                "checkpoint_path": record["checkpoint_path"],
                "checkpoint_sha256": record["checkpoint_sha256"],
                "selected": record["seed"] == selected_seed,
            }
            for record in records
        ],
        "revision_holdout_accessed": False,
    }
    (OUTPUT_ROOT / "checkpoint_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    freeze = {
        "schema": "phase6f-experiment-freeze-v1",
        "status": "FROZEN_BEFORE_R06_LABEL_OPEN",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_architecture": model,
        "selected_objective": objective,
        "selected_distillation_weight": distillation_weight,
        "parameter_count": selected["parameter_count"],
        "selected_seed": selected_seed,
        "selected_checkpoint_path": selected["checkpoint_path"],
        "selected_checkpoint_sha256": selected["checkpoint_sha256"],
        "probability_calibration_method": calibration_method,
        "probability_calibrator": selected["probability_calibrator"],
        "utility_calibrator": selected["utility_calibrator"],
        "selective_intervention_thresholds": selected["thresholds"],
        "selected_validation_policy_metrics": {
            key: selected[key] for key in (
                "hybrid_selected_utility", "hybrid_mean_regret",
                "hybrid_selected_positive_fraction", "intervention_coverage",
                "incremental_utility_vs_fallback",
            )
        },
        "deployment_seed_rule": config["final_training"]["deployment_seed_rule"],
        "latency_protocol": {
            "hardware": "NVIDIA GeForce RTX 4060 Ti",
            "profile_split": "R06_AFTER_LABEL_OPEN",
            "states_per_scale": 12,
            "warmup_repetitions": 5,
            "timed_repetitions": 30,
            "one_state_one_encoding_all_action_scoring": True,
            "model_decision_components": [
                "host_to_device", "csg_encoding", "target_set_pooling",
                "target_scoring", "calibration_gating",
            ],
            "end_to_end_additional_components": [
                "state_reconstruction", "csg_build", "tensorization",
                "action_projection",
            ],
            "p90_hard_ms_each_scale": config["success_gates"][
                "model_decision_p90_ms_hard"
            ],
        },
        "success_gates": config["success_gates"],
        "phase6f_config_sha256": sha256_file(CONFIG),
        "development_summary_sha256": sha256_file(DEVELOPMENT),
        "development_calibration_sha256": sha256_file(DEVELOPMENT_CALIBRATION),
        "mandatory_sanity_sha256": sha256_file(
            ROOT / "outputs/phase6f/audit/mandatory_sanity.json"
        ),
        "latency_bottleneck_audit_sha256": sha256_file(
            ROOT / "outputs/phase6f/profiling/phase6e_latency_bottleneck_audit.json"
        ),
        "checkpoint_manifest_sha256": sha256_file(
            OUTPUT_ROOT / "checkpoint_manifest.json"
        ),
        "sealed_label_freeze_hash": sealed["freeze_hash"],
        "selection_split": "TRAIN_VALIDATION",
        "train_internal_holdout_accessed": False,
        "revision_holdout_labels_opened": False,
    }
    freeze["freeze_hash"] = canonical_hash(freeze)
    EXPERIMENT_FREEZE.parent.mkdir(parents=True, exist_ok=True)
    EXPERIMENT_FREEZE.write_text(
        json.dumps(freeze, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "event": "phase6f_experiment_frozen",
        "selected_seed": selected_seed,
        "freeze_hash": freeze["freeze_hash"],
        "r06_labels_opened": False,
    }), flush=True)


if __name__ == "__main__":
    main()
