#!/usr/bin/env python3
"""Train the single preregistered Phase 6L score-free OOF ensemble."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_clgri.analysis.phase6l_legacy_score import (  # noqa: E402
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
)
from rcias_clgri.ni.phase6l_score_free_model import ScoreFreeCAURModel  # noqa: E402
from scripts import train_phase6j_caur as base  # noqa: E402


FAMILY = "L1_SCORE_FREE_CONT_FROZEN"
CONFIG = ROOT / "configs/phase6l_legacy_score_decoupling_v1.json"
PROTOCOL = ROOT / "outputs/phase6l_legacy_score_decoupling_v1/training/training_protocol.json"
SOURCE = ROOT / "outputs/phase6l_legacy_score_decoupling_v1/data/r12_score_free_grouped_labels.parquet"
OUT = ROOT / "outputs/phase6l_legacy_score_decoupling_v1/training"


def configure_base_module() -> None:
    """Bind validated Phase 6J training utilities to the frozen Phase 6L schema."""
    base.CATEGORICAL_COLUMNS = CATEGORICAL_COLUMNS
    base.NUMERIC_COLUMNS = NUMERIC_COLUMNS
    base.FAMILIES = (FAMILY,)
    base.CAURModel = ScoreFreeCAURModel
    base.CONFIG_PATH = CONFIG
    base.PROTOCOL_PATH = PROTOCOL
    base.SOURCE_PATH = SOURCE
    base.CACHE = ROOT / "outputs/phase6j_caur/tensor_cache"
    base.OUT = OUT


configure_base_module()


def validate_protocol() -> dict:
    protocol = base.load_json(PROTOCOL)
    checks = (
        protocol.get("schema") == "phase6l-score-free-training-protocol-v1",
        protocol.get("status") == "FROZEN_BEFORE_FIRST_OPTIMIZER_STEP",
        protocol.get("r13_accessed") is False,
        protocol.get("r14_accessed") is False,
        base.digest(SOURCE) == protocol["input_hashes"].get("score_free_grouped_labels"),
        not (ROOT / "outputs/phase6j_caur/r13_selection/access_ledger.json").exists(),
        not (ROOT / "outputs/phase6j_caur/r14_holdout/access_ledger.json").exists(),
    )
    if not all(checks):
        raise RuntimeError("Phase 6L training protocol or access boundary failed")
    for relative, expected in protocol["code_hashes"].items():
        if base.digest(ROOT / relative) != expected:
            raise RuntimeError(f"Phase 6L training code changed after freeze: {relative}")
    return protocol


def summarize(predictions: pd.DataFrame, protocol: dict) -> dict:
    result = base.ensemble_family(predictions, FAMILY, protocol)
    base.atomic_parquet(result["ensemble"], OUT / "ensemble_oof.parquet")
    base.atomic_parquet(result["states"], OUT / "state_metrics.parquet")
    base.atomic_parquet(result["selected_winners"], OUT / "selected_winners.parquet")
    base.atomic_csv(result["calibration_table"], OUT / "calibration_metrics.csv")
    base.atomic_csv(result["gate_table"], OUT / "gate_grid.csv")
    seed_rows = []
    for seed, frame in predictions.groupby("training_seed", sort=True):
        states = base.ranking_state_metrics(frame, "predicted_continuation_advantage")
        seed_rows.append({
            "training_seed": int(seed),
            "spearman": float(states.spearman.mean()),
            "pairwise_accuracy": float(states.pairwise_accuracy.mean()),
            "ndcg_at_1": float(states.ndcg_at_1.mean()),
            "selected_lift": float(states.selected_lift.mean()),
            "selection_regret": float(states.selection_regret.mean()),
        })
    base.atomic_csv(pd.DataFrame(seed_rows), OUT / "three_seed_stability.csv")
    summary = {
        "schema": "phase6l-score-free-oof-summary-v1",
        "status": "COMPLETE",
        "model_family": FAMILY,
        "metrics": result["metrics"],
        "calibration": result["calibration"],
        "selected_gate": result["selected_gate"],
        "eligible_gate_count": int(result["gate_table"].retained.sum()),
        "r13_accessed": False,
        "r14_accessed": False,
    }
    base.atomic_json(summary, OUT / "oof_summary.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--max-new-runs", type=int)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    protocol = validate_protocol()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Phase 6L CUDA training requested but CUDA is unavailable")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(4)
    frame = pd.read_parquet(SOURCE)
    samples = base.load_samples()
    frames = base.state_frames(frame)
    root = OUT / "smoke" if args.smoke else OUT
    protocol_sha256 = base.digest(PROTOCOL)
    predictions = []
    completed = 0
    new_runs = 0
    started = time.perf_counter()
    expected_runs = len(protocol["training"]["seeds"]) * 3
    stop = False
    for seed in protocol["training"]["seeds"]:
        for held_fold in range(3):
            paths = base.run_paths(FAMILY, int(seed), held_fold, root=root)
            if not args.smoke and base.valid_run(paths, protocol_sha256):
                predictions.append(pd.read_parquet(paths[1]))
                completed += 1
                print(json.dumps({
                    "event": "phase6l_run_skip", "seed": seed,
                    "held_fold": held_fold,
                }), flush=True)
                continue
            if args.max_new_runs is not None and new_runs >= args.max_new_runs:
                stop = True
                break
            run_started = time.perf_counter()
            model, transform, history, best_epoch, held = base.train_run(
                FAMILY, int(seed), held_fold, samples,
                frames, frame, protocol, device,
                epoch_cap=2 if args.smoke else None,
            )
            held["model_family"] = FAMILY
            held["training_seed"] = int(seed)
            held["held_fold"] = held_fold
            held["best_epoch"] = best_epoch
            checkpoint_path, prediction_path, record_path = paths
            base.save_checkpoint({
                "schema": "phase6l-score-free-oof-checkpoint-v1",
                "model_family": FAMILY,
                "training_seed": int(seed),
                "held_fold": held_fold,
                "training_protocol_sha256": protocol_sha256,
                "base_checkpoint_sha256": protocol["base_checkpoint"]["sha256"],
                "feature_transform": transform.to_dict(),
                "trainable_model_state": base.trainable_state(model),
            }, checkpoint_path)
            base.atomic_parquet(held, prediction_path)
            record = {
                "schema": "phase6l-score-free-oof-run-v1",
                "status": "COMPLETE",
                "model_family": FAMILY,
                "training_seed": int(seed),
                "held_fold": held_fold,
                "best_epoch": best_epoch,
                "inner_epochs_run": len(history["inner_epoch_selection"]),
                "outer_final_epochs_run": len(history["outer_final_fit"]),
                "best_selected_lift_lcb": max(
                    row["validation_selected_lift_lcb"]
                    for row in history["inner_epoch_selection"]
                ),
                "history": history,
                "runtime_seconds": time.perf_counter() - run_started,
                "training_protocol_sha256": protocol_sha256,
                "checkpoint_sha256": base.digest(checkpoint_path),
                "predictions_sha256": base.digest(prediction_path),
                "r13_accessed": False,
                "r14_accessed": False,
            }
            base.atomic_json(record, record_path)
            predictions.append(held)
            completed += 1
            new_runs += 1
            print(json.dumps({
                "event": "phase6l_run_complete", "seed": seed,
                "held_fold": held_fold, "best_epoch": best_epoch,
                "runtime_seconds": record["runtime_seconds"],
            }), flush=True)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if args.smoke:
                stop = True
                break
        if stop:
            break
    status = "RUNNING"
    summary = None
    if not args.smoke and completed == expected_runs:
        combined = pd.concat(predictions, ignore_index=True)
        base.atomic_parquet(combined, OUT / "oof_predictions.parquet")
        summary = summarize(combined, protocol)
        status = "COMPLETE"
    progress = {
        "schema": "phase6l-score-free-training-progress-v1",
        "status": "SMOKE_COMPLETE" if args.smoke else status,
        "completed_runs": completed,
        "expected_runs": expected_runs,
        "new_runs": new_runs,
        "elapsed_seconds": time.perf_counter() - started,
        "training_protocol_sha256": protocol_sha256,
        "summary": summary,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "r13_accessed": False,
        "r14_accessed": False,
    }
    base.atomic_json(progress, root / "progress.json")
    print(json.dumps(progress, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
