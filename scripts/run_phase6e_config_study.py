#!/usr/bin/env python3
"""Run the pre-registered TRAIN/TRAIN_VALIDATION Phase 6E config study."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
        raise RuntimeError(f"training command failed with return code {return_code}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/phase6e_training.json"
    )
    parser.add_argument(
        "--cache-manifest",
        type=Path,
        default=ROOT / "outputs/phase6e/tensorization/cache/cache_manifest.csv",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs/phase6e/training/config_study",
    )
    parser.add_argument(
        "--freeze-output",
        type=Path,
        default=ROOT / "outputs/phase6e/audit/experiment_freeze.json",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    raw = json.loads(args.config.read_text(encoding="utf-8"))
    protocol = raw["development_protocol"]
    seed = int(protocol["seed"])
    epochs = int(protocol["epochs"])
    candidates = list(raw["development_candidates"])
    args.output_root.mkdir(parents=True, exist_ok=True)
    launch = {
        "schema": "phase6e-config-study-launch-v1",
        "status": "RUNNING",
        "candidates": candidates,
        "seed": seed,
        "epochs": epochs,
        "gradient_split": "TRAIN",
        "selection_split": "TRAIN_VALIDATION",
        "internal_holdout_accessed": False,
        "config_sha256": file_sha256(args.config),
    }
    (args.output_root / "study_launch.json").write_text(
        json.dumps(launch, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"event": "config_study_launch", **launch}), flush=True)
    histories = []
    summaries = []
    for candidate in candidates:
        run_directory = args.output_root / candidate
        summary_path = run_directory / "training_summary.json"
        if not summary_path.exists():
            command = [
                sys.executable,
                str(ROOT / "scripts/train_phase6e_supervised_ni.py"),
                "--candidate", candidate,
                "--seed", str(seed),
                "--run-name", candidate,
                "--epochs", str(epochs),
                "--device", args.device,
                "--config", str(args.config),
                "--cache-manifest", str(args.cache_manifest),
                "--output-root", str(args.output_root),
            ]
            checkpoint = run_directory / "checkpoint_last.pt"
            if checkpoint.exists():
                command.extend(["--resume", str(checkpoint)])
            stream_command(command, args.output_root / f"{candidate}.log")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") != "COMPLETE":
            raise RuntimeError(f"candidate {candidate} is not complete")
        history = pd.read_csv(run_directory / "training_history.csv")
        history.insert(0, "candidate", candidate)
        histories.append(history)
        summaries.append({"candidate": candidate, **summary})
        print(json.dumps({
            "event": "candidate_complete",
            "candidate": candidate,
            "best_validation_objective": summary["best_validation_objective"],
            "best_epoch": summary["best_epoch"],
        }), flush=True)

    validation = pd.concat(histories, ignore_index=True)
    validation.to_csv(args.output_root / "validation_metrics.csv", index=False)
    summary_frame = pd.DataFrame(summaries).sort_values(
        ["best_validation_objective", "parameter_count", "candidate"],
        ascending=[False, True, True],
        kind="stable",
    )
    summary_frame.to_csv(args.output_root / "candidate_summary.csv", index=False)
    selected = str(summary_frame.iloc[0]["candidate"])
    selected_raw = raw["development_candidates"][selected]
    cache_audit = ROOT / "outputs/phase6e/audit/tensor_cache_audit.json"
    boundary = ROOT / "outputs/phase6e/environment/freeze_record.json"
    freeze = {
        "schema": "phase6e-experiment-freeze-v1",
        "status": "FROZEN_BEFORE_INTERNAL_HOLDOUT",
        "selection_rule": raw["validation_objective"],
        "selected_candidate": selected,
        "selected_model": selected_raw["model"],
        "selected_training": {
            **raw["training_defaults"],
            **selected_raw.get("training", {}),
        },
        "selected_loss": raw["loss"],
        "final_seeds": raw["final_seeds"],
        "development_seed": seed,
        "development_epochs": epochs,
        "candidate_summary": summary_frame.to_dict("records"),
        "tensor_cache_audit_sha256": file_sha256(cache_audit),
        "phase6e_boundary_sha256": file_sha256(boundary),
        "training_config_sha256": file_sha256(args.config),
        "tensor_schema_hash": json.loads(cache_audit.read_text())["tensor_schema_hash"],
        "internal_holdout_metrics_observed": False,
        "internal_holdout_checkpoint_selection": False,
    }
    args.freeze_output.parent.mkdir(parents=True, exist_ok=True)
    args.freeze_output.write_text(
        json.dumps(freeze, indent=2) + "\n", encoding="utf-8"
    )
    completion = {
        "schema": "phase6e-config-study-summary-v1",
        "status": "COMPLETE",
        "selected_candidate": selected,
        "candidate_count": len(candidates),
        "validation_rows": len(validation),
        "experiment_freeze": str(args.freeze_output),
        "experiment_freeze_sha256": file_sha256(args.freeze_output),
    }
    (args.output_root / "study_summary.json").write_text(
        json.dumps(completion, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"event": "config_study_complete", **completion}), flush=True)


if __name__ == "__main__":
    main()
