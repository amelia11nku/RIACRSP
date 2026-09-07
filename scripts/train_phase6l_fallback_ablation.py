#!/usr/bin/env python3
"""Run the preregistered nonselectable Phase 6L fallback-context ablation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_clgri.analysis.phase6l_legacy_score import (  # noqa: E402
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
)
from scripts import train_phase6l_score_free as primary  # noqa: E402


base = primary.base
FAMILY = primary.FAMILY
SEED = 706101
REMOVED = ("fallback_overlap_fraction", "fallback_jaccard", "is_fallback")
OUT = ROOT / "outputs/phase6l_legacy_score_decoupling_v1/ablation/no_fallback_context"
PROTOCOL = OUT / "ablation_protocol.json"
PRIMARY_TRANSFORM = base.transform_features


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def ablation_transform(frame, transform):
    categorical, numeric, _ = PRIMARY_TRANSFORM(frame, transform)
    supported = np.ones(len(frame), dtype=bool)
    for column_index, column in enumerate(CATEGORICAL_COLUMNS):
        mapping = {
            value: index + 1
            for index, value in enumerate(transform.vocabularies[column])
        }
        supported &= frame[column].astype(str).map(mapping).fillna(0).to_numpy() > 0
    for column_index, column in enumerate(NUMERIC_COLUMNS):
        if column in REMOVED:
            numeric[:, column_index] = 0.0
            continue
        values = frame[column].to_numpy(dtype=float)
        robust = (values - transform.medians[column]) / transform.iqrs[column]
        supported &= np.isfinite(robust) & (robust >= -8.0) & (robust <= 8.0)
    return categorical, numeric, supported


def freeze() -> dict:
    config = json.loads(primary.CONFIG.read_text())
    declared = config["nonselectable_ablation"]
    if declared["model_id"] != "L1_NO_FALLBACK_CONTEXT_ABLATION":
        raise RuntimeError("ablation was not preregistered")
    record = {
        "schema": "phase6l-fallback-context-ablation-protocol-v1",
        "status": "FROZEN_BEFORE_ABLATION_OPTIMIZER_STEP",
        "model_id": declared["model_id"],
        "seed": SEED,
        "removed_online_inputs": list(REMOVED),
        "implementation": "retain semantic fallback index; replace the three model inputs by constant zero and exclude them from support",
        "eligible_for_selection_or_promotion": False,
        "source_hashes": {
            str(path.relative_to(ROOT)): digest(path)
            for path in (
                primary.CONFIG,
                primary.PROTOCOL,
                primary.SOURCE,
                ROOT / "scripts/train_phase6l_fallback_ablation.py",
            )
        },
        "r13_accessed": False,
        "r14_accessed": False,
    }
    text = json.dumps(record, indent=2, sort_keys=True) + "\n"
    if PROTOCOL.exists() and PROTOCOL.read_text() != text:
        raise RuntimeError("refusing to replace frozen ablation protocol")
    PROTOCOL.parent.mkdir(parents=True, exist_ok=True)
    if not PROTOCOL.exists():
        PROTOCOL.write_text(text)
    return record


def main() -> None:
    freeze()
    protocol = primary.validate_protocol()
    base.transform_features = ablation_transform
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 6L ablation requires CUDA")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(4)
    frame = pd.read_parquet(primary.SOURCE)
    samples = base.load_samples()
    frames = base.state_frames(frame)
    predictions = []
    runs = []
    started = time.perf_counter()
    for held_fold in range(3):
        stem = OUT / "oof" / f"fold_{held_fold}"
        checkpoint, prediction, record_path = (
            stem.with_suffix(".pt"), stem.with_suffix(".parquet"), stem.with_suffix(".json")
        )
        if all(path.is_file() for path in (checkpoint, prediction, record_path)):
            record = json.loads(record_path.read_text())
            if (record["checkpoint_sha256"] != digest(checkpoint)
                    or record["predictions_sha256"] != digest(prediction)):
                raise RuntimeError(f"invalid ablation resume artifacts: fold {held_fold}")
            predictions.append(pd.read_parquet(prediction))
            runs.append(record)
            continue
        run_started = time.perf_counter()
        model, transform, history, best_epoch, held = base.train_run(
            FAMILY, SEED, held_fold, samples, frames, frame, protocol, device
        )
        held["model_family"] = "L1_NO_FALLBACK_CONTEXT_ABLATION"
        held["training_seed"] = SEED
        held["held_fold"] = held_fold
        held["best_epoch"] = best_epoch
        base.save_checkpoint({
            "schema": "phase6l-fallback-context-ablation-checkpoint-v1",
            "training_seed": SEED,
            "held_fold": held_fold,
            "ablation_protocol_sha256": digest(PROTOCOL),
            "feature_transform": transform.to_dict(),
            "trainable_model_state": base.trainable_state(model),
        }, checkpoint)
        base.atomic_parquet(held, prediction)
        record = {
            "schema": "phase6l-fallback-context-ablation-run-v1",
            "status": "COMPLETE",
            "training_seed": SEED,
            "held_fold": held_fold,
            "best_epoch": best_epoch,
            "inner_epochs_run": len(history["inner_epoch_selection"]),
            "runtime_seconds": time.perf_counter() - run_started,
            "checkpoint_sha256": digest(checkpoint),
            "predictions_sha256": digest(prediction),
            "r13_accessed": False,
            "r14_accessed": False,
        }
        base.atomic_json(record, record_path)
        predictions.append(held)
        runs.append(record)
        del model
        torch.cuda.empty_cache()
    combined = pd.concat(predictions, ignore_index=True)
    states = base.ranking_state_metrics(combined, "predicted_continuation_advantage")
    lower, upper = base.grouped_bootstrap_interval(
        states, "selected_lift", seed=707001, resamples=2000
    )
    metrics = {
        "state_count": len(states),
        "action_count": len(combined),
        "overall_spearman": float(states.spearman.mean()),
        "pairwise_accuracy": float(states.pairwise_accuracy.mean()),
        "ndcg_at_1": float(states.ndcg_at_1.mean()),
        "selected_lift": float(states.selected_lift.mean()),
        "selected_lift_lcb": lower,
        "selected_lift_ucb": upper,
        "selection_regret": float(states.selection_regret.mean()),
        "mean_spearman_by_scale": {
            scale: float(states.loc[states.scale.eq(scale), "spearman"].mean())
            for scale in ("S", "M", "L")
        },
    }
    result = {
        "schema": "phase6l-fallback-context-ablation-result-v1",
        "status": "COMPLETE_NONSELECTABLE",
        "eligible_for_selection_or_promotion": False,
        "seed": SEED,
        "removed_online_inputs": list(REMOVED),
        "metrics": metrics,
        "runs": runs,
        "elapsed_seconds": time.perf_counter() - started,
        "r13_accessed": False,
        "r14_accessed": False,
    }
    base.atomic_parquet(combined, OUT / "oof_predictions.parquet")
    base.atomic_parquet(states, OUT / "state_metrics.parquet")
    base.atomic_json(result, OUT / "result.json")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
