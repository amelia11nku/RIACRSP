#!/usr/bin/env python3
"""Train/resume frozen flat/static/no-edge Phase 6E controls."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("FLAT_SET", "STATIC_CSG", "NO_EDGE_FEATURES")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stream_command(command: list[str], log_path: Path) -> None:
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
        raise RuntimeError(f"ablation command failed with return code {return_code}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/phase6e_training.json"
    )
    parser.add_argument(
        "--evaluation-config",
        type=Path,
        default=ROOT / "configs/phase6e_evaluation.json",
    )
    parser.add_argument(
        "--experiment-freeze",
        type=Path,
        default=ROOT / "outputs/phase6e/audit/experiment_freeze.json",
    )
    parser.add_argument(
        "--final-audit",
        type=Path,
        default=ROOT / "outputs/phase6e/audit/final_training_audit.json",
    )
    parser.add_argument(
        "--cache-manifest",
        type=Path,
        default=ROOT / "outputs/phase6e/tensorization/cache/cache_manifest.csv",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs/phase6e/ablations/training",
    )
    args = parser.parse_args()
    freeze = json.loads(args.experiment_freeze.read_text(encoding="utf-8"))
    audit = json.loads(args.final_audit.read_text(encoding="utf-8"))
    evaluation = json.loads(args.evaluation_config.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS" or audit.get("internal_holdout_metrics_observed") is not False:
        raise ValueError("final training audit has not passed before holdout")
    candidate = str(freeze["selected_candidate"])
    seed = int(evaluation["controlled_neural_seed"])
    args.output_root.mkdir(parents=True, exist_ok=True)
    launch = {
        "schema": "phase6e-ablation-training-launch-v1",
        "status": "RUNNING",
        "variants": list(VARIANTS),
        "seed": seed,
        "selected_candidate": candidate,
        "evaluation_plan_sha256": file_sha256(args.evaluation_config),
        "experiment_freeze_sha256": file_sha256(args.experiment_freeze),
        "final_training_audit_sha256": file_sha256(args.final_audit),
        "internal_holdout_accessed": False,
    }
    (args.output_root / "launch_record.json").write_text(
        json.dumps(launch, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"event": "ablation_training_launch", **launch}), flush=True)
    summary_rows = []
    checkpoints = []
    for variant in VARIANTS:
        run_directory = args.output_root / variant
        summary_path = run_directory / "training_summary.json"
        if not summary_path.exists():
            command = [
                sys.executable,
                str(ROOT / "scripts/train_phase6e_supervised_ni.py"),
                "--candidate", candidate,
                "--variant", variant,
                "--seed", str(seed),
                "--run-name", variant,
                "--device", args.device,
                "--config", str(args.config),
                "--cache-manifest", str(args.cache_manifest),
                "--output-root", str(args.output_root),
            ]
            checkpoint = run_directory / "checkpoint_last.pt"
            if checkpoint.exists():
                command.extend(["--resume", str(checkpoint)])
            stream_command(command, args.output_root / f"{variant}.log")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") != "COMPLETE":
            raise RuntimeError(f"ablation {variant} did not complete")
        history = pd.read_csv(run_directory / "training_history.csv")
        best = history[history["epoch"] == int(summary["best_epoch"])].iloc[0]
        summary_rows.append({
            "variant": variant,
            **summary,
            **{key: value for key, value in best.items() if key.startswith("validation_")},
        })
        checkpoint = run_directory / "checkpoint_best.pt"
        checkpoints.append({
            "variant": variant,
            "seed": seed,
            "best_epoch": int(summary["best_epoch"]),
            "checkpoint_path": str(checkpoint),
            "checkpoint_bytes": checkpoint.stat().st_size,
            "checkpoint_sha256": file_sha256(checkpoint),
            "config_fingerprint": summary["config_fingerprint"],
        })
        print(json.dumps({
            "event": "ablation_complete",
            "variant": variant,
            "best_epoch": summary["best_epoch"],
            "best_validation_objective": summary["best_validation_objective"],
        }), flush=True)
    pd.DataFrame(summary_rows).to_csv(
        args.output_root / "ablation_training_summary.csv", index=False
    )
    manifest = {
        "schema": "phase6e-ablation-checkpoint-manifest-v1",
        "status": "COMPLETE",
        "evaluation_plan_sha256": file_sha256(args.evaluation_config),
        "checkpoints": checkpoints,
        "internal_holdout_metrics_observed": False,
    }
    (args.output_root / "checkpoint_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    completion = {
        "schema": "phase6e-ablation-training-summary-v1",
        "status": "COMPLETE",
        "variant_count": len(VARIANTS),
        "checkpoint_manifest_sha256": file_sha256(
            args.output_root / "checkpoint_manifest.json"
        ),
        "internal_holdout_metrics_observed": False,
    }
    (args.output_root / "ablation_training_completion.json").write_text(
        json.dumps(completion, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"event": "ablation_training_complete", **completion}), flush=True)


if __name__ == "__main__":
    main()
