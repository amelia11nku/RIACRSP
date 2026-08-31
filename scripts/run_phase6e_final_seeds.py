#!/usr/bin/env python3
"""Train/resume the three frozen Phase 6E final seeds sequentially."""

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
        raise RuntimeError(f"final-seed command failed with return code {return_code}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/phase6e_training.json"
    )
    parser.add_argument(
        "--freeze",
        type=Path,
        default=ROOT / "outputs/phase6e/audit/experiment_freeze.json",
    )
    parser.add_argument(
        "--config-audit",
        type=Path,
        default=ROOT / "outputs/phase6e/audit/config_study_audit.json",
    )
    parser.add_argument(
        "--cache-manifest",
        type=Path,
        default=ROOT / "outputs/phase6e/tensorization/cache/cache_manifest.csv",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs/phase6e/training/final_seeds",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    raw = json.loads(args.config.read_text(encoding="utf-8"))
    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    audit = json.loads(args.config_audit.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS":
        raise ValueError("configuration study audit has not passed")
    if freeze.get("status") != "FROZEN_BEFORE_INTERNAL_HOLDOUT":
        raise ValueError("experiment is not frozen before holdout")
    if freeze.get("internal_holdout_metrics_observed") is not False:
        raise ValueError("holdout metrics were observed before final training")
    candidate = str(freeze["selected_candidate"])
    candidate_raw = raw["development_candidates"][candidate]
    merged_training = {**raw["training_defaults"], **candidate_raw.get("training", {})}
    if candidate_raw["model"] != freeze["selected_model"]:
        raise ValueError("frozen model differs from current config")
    if merged_training != freeze["selected_training"]:
        raise ValueError("frozen training protocol differs from current config")
    if raw["loss"] != freeze["selected_loss"]:
        raise ValueError("frozen loss differs from current config")
    seeds = [int(seed) for seed in freeze["final_seeds"]]
    args.output_root.mkdir(parents=True, exist_ok=True)
    launch = {
        "schema": "phase6e-final-seeds-launch-v1",
        "status": "RUNNING",
        "selected_candidate": candidate,
        "seeds": seeds,
        "epochs": int(merged_training["epochs"]),
        "experiment_freeze_sha256": file_sha256(args.freeze),
        "config_study_audit_sha256": file_sha256(args.config_audit),
        "internal_holdout_accessed": False,
    }
    (args.output_root / "launch_record.json").write_text(
        json.dumps(launch, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"event": "final_seeds_launch", **launch}), flush=True)
    seed_rows = []
    validation_rows = []
    checkpoint_rows = []
    for seed in seeds:
        run_name = f"seed_{seed}"
        run_directory = args.output_root / run_name
        summary_path = run_directory / "training_summary.json"
        if not summary_path.exists():
            command = [
                sys.executable,
                str(ROOT / "scripts/train_phase6e_supervised_ni.py"),
                "--candidate", candidate,
                "--seed", str(seed),
                "--run-name", run_name,
                "--device", args.device,
                "--config", str(args.config),
                "--cache-manifest", str(args.cache_manifest),
                "--output-root", str(args.output_root),
            ]
            checkpoint = run_directory / "checkpoint_last.pt"
            if checkpoint.exists():
                command.extend(["--resume", str(checkpoint)])
            stream_command(command, args.output_root / f"{run_name}.log")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") != "COMPLETE":
            raise RuntimeError(f"final seed {seed} did not complete")
        history = pd.read_csv(run_directory / "training_history.csv")
        history.insert(0, "seed", seed)
        validation_rows.append(history)
        best_epoch = int(summary["best_epoch"])
        best_metrics = history[history["epoch"] == best_epoch].iloc[0].to_dict()
        seed_rows.append({
            "seed": seed,
            "candidate": candidate,
            **summary,
            **{key: value for key, value in best_metrics.items() if key.startswith("validation_")},
        })
        best_checkpoint = run_directory / "checkpoint_best.pt"
        checkpoint_rows.append({
            "seed": seed,
            "candidate": candidate,
            "best_epoch": best_epoch,
            "config_fingerprint": summary["config_fingerprint"],
            "checkpoint_path": str(best_checkpoint),
            "checkpoint_bytes": best_checkpoint.stat().st_size,
            "checkpoint_sha256": file_sha256(best_checkpoint),
            "training_summary_sha256": file_sha256(summary_path),
        })
        print(json.dumps({
            "event": "final_seed_complete",
            "seed": seed,
            "best_epoch": best_epoch,
            "best_validation_objective": summary["best_validation_objective"],
        }), flush=True)

    seed_summary = pd.DataFrame(seed_rows)
    seed_summary.to_csv(args.output_root / "model_seed_summary.csv", index=False)
    pd.concat(validation_rows, ignore_index=True).to_csv(
        args.output_root / "validation_metrics.csv", index=False
    )
    checkpoint_manifest = {
        "schema": "phase6e-final-checkpoint-manifest-v1",
        "status": "COMPLETE",
        "selected_candidate": candidate,
        "experiment_freeze_sha256": file_sha256(args.freeze),
        "checkpoints": checkpoint_rows,
    }
    (args.output_root / "checkpoint_manifest.json").write_text(
        json.dumps(checkpoint_manifest, indent=2) + "\n", encoding="utf-8"
    )
    completion = {
        "schema": "phase6e-final-seeds-summary-v1",
        "status": "COMPLETE",
        "selected_candidate": candidate,
        "seed_count": len(seeds),
        "seeds": seeds,
        "checkpoint_manifest_sha256": file_sha256(
            args.output_root / "checkpoint_manifest.json"
        ),
        "internal_holdout_metrics_observed": False,
    }
    (args.output_root / "final_seeds_summary.json").write_text(
        json.dumps(completion, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"event": "final_seeds_complete", **completion}), flush=True)


if __name__ == "__main__":
    main()
