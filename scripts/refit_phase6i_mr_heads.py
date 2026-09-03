#!/usr/bin/env python3
"""Run the frozen full-data and one-round hard-state Phase 6I-MR refits."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_clgri.ni.phase6i_heads import parameter_count  # noqa: E402
from scripts.prepare_phase6i_mr_training_data import (  # noqa: E402
    atomic_json,
    atomic_parquet,
    digest,
    load_json,
)
from scripts.train_phase6i_mr_heads import (  # noqa: E402
    CACHE_INTEGRITY,
    PROTOCOL_PATH,
    StateTensorData,
    batch_loss,
    cyclic_draw,
    initialize,
    load_source,
    objective_from,
    predict_holdout,
    state_metrics,
    state_tensor_data,
)


OUT = ROOT / "outputs/phase6i_mr/model_training"
HARD_ROOT = ROOT / "outputs/phase6i_mr/hard_state_round1"
HARD_ACTIONS = HARD_ROOT / "hard_state_actions.parquet"
HARD_INTEGRITY = HARD_ROOT / "hard_state_integrity.json"


def save_checkpoint(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def artifact_paths(
    stage: str, family: str, variant: str, seed: int, fold: str | None
) -> tuple[Path, Path, Path | None]:
    base = OUT / stage / family / variant / f"seed_{seed}"
    if fold is not None:
        base = base / fold
        return base.with_suffix(".pt"), base.with_suffix(".json"), base.with_suffix(".parquet")
    return base.with_suffix(".pt"), base.with_suffix(".json"), None


def valid_artifact(
    paths: tuple[Path, Path, Path | None], protocol_hash: str
) -> bool:
    checkpoint, record_path, prediction_path = paths
    required = [checkpoint, record_path]
    if prediction_path is not None:
        required.append(prediction_path)
    if not all(path.is_file() for path in required):
        return False
    try:
        record = load_json(record_path)
    except (OSError, json.JSONDecodeError):
        return False
    return all([
        record.get("status") == "COMPLETE",
        record.get("training_protocol_sha256") == protocol_hash,
        record.get("checkpoint_sha256") == digest(checkpoint),
        prediction_path is None
        or record.get("predictions_sha256") == digest(prediction_path),
        record.get("r10_accessed") is False,
        record.get("r11_accessed") is False,
    ])


def train_refit(
    family: str,
    variant: str,
    seed: int,
    live: StateTensorData,
    old: StateTensorData | None,
    hard: StateTensorData | None,
    protocol: dict,
    *,
    held_fold: str | None,
    target: str,
) -> tuple[torch.nn.Module, list[dict[str, float]]]:
    base_budget = protocol["training_budgets"][target]
    hard_budget = protocol["training_budgets"]["one_round_hard_refit"]
    is_hard = variant == "HARD_AGG_20_60_20"
    epochs = hard_budget["epochs"] if is_hard else base_budget["epochs"]
    model = initialize(seed, family, base_budget["dropout"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base_budget["learning_rate"],
        weight_decay=base_budget["weight_decay"],
    )
    objective = objective_from(protocol, target)
    live_indices = np.flatnonzero(live.folds != held_fold) if held_fold else np.arange(live.state_count)
    old_indices = np.arange(old.state_count) if old is not None else np.empty(0, dtype=np.int64)
    hard_indices = (
        np.flatnonzero(hard.folds != held_fold) if held_fold and hard is not None
        else np.arange(hard.state_count) if hard is not None
        else np.empty(0, dtype=np.int64)
    )
    history = []
    model.train()
    for epoch in range(epochs):
        rng = np.random.default_rng(seed + epoch * 1_000_003)
        if is_hard:
            if old is None or hard is None:
                raise RuntimeError("hard refit requires old, live, and hard R09 states")
            counts = hard_budget["state_batch_counts"]
            steps = max(
                math.ceil(len(old_indices) / counts["earlier_train"]),
                math.ceil(len(live_indices) / counts["r09_live"]),
                math.ceil(len(hard_indices) / counts["r09_hard_state"]),
            )
            draws = {
                "old": cyclic_draw(old_indices, steps * counts["earlier_train"], rng),
                "live": cyclic_draw(live_indices, steps * counts["r09_live"], rng),
                "hard": cyclic_draw(hard_indices, steps * counts["r09_hard_state"], rng),
            }
            jobs = [
                (
                    draws["live"][step * counts["r09_live"]:(step + 1) * counts["r09_live"]],
                    draws["old"][step * counts["earlier_train"]:(step + 1) * counts["earlier_train"]],
                    draws["hard"][step * counts["r09_hard_state"]:(step + 1) * counts["r09_hard_state"]],
                )
                for step in range(steps)
            ]
        elif variant == "MIXED_OLD_NEW":
            if old is None:
                raise RuntimeError("mixed base refit requires old TRAIN states")
            old_size, live_size = base_budget["mixed_state_batch_counts"]
            steps = max(
                math.ceil(len(old_indices) / old_size),
                math.ceil(len(live_indices) / live_size),
            )
            live_draw = cyclic_draw(live_indices, steps * live_size, rng)
            old_draw = cyclic_draw(old_indices, steps * old_size, rng)
            jobs = [
                (
                    live_draw[step * live_size:(step + 1) * live_size],
                    old_draw[step * old_size:(step + 1) * old_size],
                    None,
                )
                for step in range(steps)
            ]
        else:
            order = rng.permutation(live_indices)
            size = min(base_budget["state_batch_size"], len(order))
            jobs = [
                (order[start:start + size], None, None)
                for start in range(0, len(order), size)
            ]
        values: dict[str, list[float]] = {}
        for live_batch, old_batch, hard_batch in jobs:
            optimizer.zero_grad(set_to_none=True)
            live_loss = batch_loss(model, live, live_batch, objective)
            if hard_batch is not None:
                assert old is not None and hard is not None and old_batch is not None
                old_loss = batch_loss(model, old, old_batch, objective)
                hard_loss = batch_loss(model, hard, hard_batch, objective)
                weights = hard_budget["source_loss_weights"]
                loss = (
                    weights["r09_live"] * live_loss["loss"]
                    + weights["earlier_train"] * old_loss["loss"]
                    + weights["r09_hard_state"] * hard_loss["loss"]
                )
                components = {
                    name: (
                        weights["r09_live"] * live_loss[name]
                        + weights["earlier_train"] * old_loss[name]
                        + weights["r09_hard_state"] * hard_loss[name]
                    )
                    for name in live_loss if name != "pair_count"
                }
            elif old_batch is not None:
                assert old is not None
                old_loss = batch_loss(model, old, old_batch, objective)
                weights = protocol["source_loss_weights"]
                loss = weights["r09_live"] * live_loss["loss"] + weights["earlier_train"] * old_loss["loss"]
                components = {
                    name: weights["r09_live"] * live_loss[name] + weights["earlier_train"] * old_loss[name]
                    for name in live_loss if name != "pair_count"
                }
            else:
                loss = live_loss["loss"]
                components = {name: value for name, value in live_loss.items() if name != "pair_count"}
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), base_budget["gradient_norm_clip"])
            optimizer.step()
            for name, value in components.items():
                values.setdefault(name, []).append(float(value.detach()))
        row = {"epoch": epoch + 1}
        row.update({name: float(np.mean(items)) for name, items in values.items()})
        history.append(row)
    return model, history


def hard_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, group in predictions.groupby(
        ["model_family", "data_variant", "training_seed"], sort=True
    ):
        state = state_metrics(group, "decoded_immediate_utility")
        instance = state.groupby(["instance_id", "scale"], as_index=False).mean(numeric_only=True)
        scales = instance.groupby("scale").spearman.mean().to_dict()
        rows.append({
            "model_family": key[0],
            "data_variant": key[1],
            "training_seed": int(key[2]),
            "overall_per_instance_spearman": float(instance.groupby("instance_id").spearman.mean().mean()),
            "scale_S_per_instance_spearman": float(scales.get("S", math.nan)),
            "scale_M_per_instance_spearman": float(scales.get("M", math.nan)),
            "scale_L_per_instance_spearman": float(scales.get("L", math.nan)),
            "pairwise_accuracy": float(state.pairwise_accuracy.mean()),
            "ndcg_at_1": float(state.ndcg_at_1.mean()),
            "top1_agreement": float(state.top1_agreement.mean()),
            "selected_value": float(state.selected_value.mean()),
            "selected_lift_over_fallback": float(state.selected_lift_over_fallback.mean()),
            "selected_sign_error": float(state.selected_sign_error.mean()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=["hard_oof", "full", "continuation_full"], required=True
    )
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    protocol = load_json(PROTOCOL_PATH)
    cache = load_json(CACHE_INTEGRITY)
    hard_integrity = load_json(HARD_INTEGRITY)
    if not all([
        protocol.get("r10_accessed") is False,
        protocol.get("r11_accessed") is False,
        cache.get("status") == "PASS",
        hard_integrity.get("status") == "PASS",
        hard_integrity.get("round") == 1,
    ]):
        raise RuntimeError("refit inputs are not complete and leakage-safe")
    protocol_hash = digest(PROTOCOL_PATH)
    fold_map = {
        cell: fold for fold, cells in protocol["grouped_oof_folds"].items() for cell in cells
    }
    contexts = protocol["context_columns"]
    if args.stage == "continuation_full":
        scale = protocol["targets"]["continuation"]["absolute_p99_scale"]
        live = state_tensor_data(
            load_source("continuation"), raw_target="continuation_value",
            target_scale=scale, fold_map=fold_map, context_columns=contexts,
        )
        old = None
        hard = None
        candidates = [("U2_H_CONTINUATION", "CONTINUATION_ONLY")]
        folds = [None]
        target = "continuation"
    else:
        scale = protocol["targets"]["immediate"]["absolute_p99_scale"]
        live = state_tensor_data(
            load_source("r09"), raw_target="decoded_immediate_utility",
            target_scale=scale, fold_map=fold_map, context_columns=contexts,
        )
        old = state_tensor_data(
            load_source("old_train"), raw_target="decoded_immediate_utility",
            target_scale=scale, fold_map=fold_map, context_columns=contexts,
        )
        hard = state_tensor_data(
            pd.read_parquet(HARD_ACTIONS), raw_target="decoded_immediate_utility",
            target_scale=scale, fold_map=fold_map, context_columns=contexts,
        )
        target = "immediate"
    if args.stage == "hard_oof":
        candidates = [(family, "HARD_AGG_20_60_20") for family in ["U1", "U2"]]
        folds: list[str | None] = list(protocol["grouped_oof_folds"])
    elif args.stage == "full":
        candidates = [
            (family, variant)
            for family in ["U1", "U2"]
            for variant in ["R09_ONLY", "MIXED_OLD_NEW", "HARD_AGG_20_60_20"]
        ]
        folds = [None]
    expected = len(candidates) * len(protocol["training_seeds"]) * len(folds)
    completed = 0
    prediction_frames = []
    started = time.perf_counter()
    for family, variant in candidates:
        for seed in protocol["training_seeds"]:
            for fold in folds:
                stage_name = (
                    "hard_oof" if fold else
                    "full_artifacts_continuation" if target == "continuation" else
                    "full_artifacts"
                )
                paths = artifact_paths(stage_name, family, variant, seed, fold)
                if valid_artifact(paths, protocol_hash):
                    completed += 1
                    if paths[2] is not None:
                        prediction_frames.append(pd.read_parquet(paths[2]))
                    print(json.dumps({"event": "refit_skip", "family": family, "variant": variant, "seed": seed, "fold": fold}), flush=True)
                    continue
                run_started = time.perf_counter()
                model, history = train_refit(
                    family, variant, seed, live, old, hard, protocol,
                    held_fold=fold, target=target,
                )
                checkpoint, record_path, prediction_path = paths
                save_checkpoint({
                    "schema": "phase6i-mr-refit-checkpoint-v1.2",
                    "family": family,
                    "variant": variant,
                    "training_seed": seed,
                    "held_fold": fold,
                    "target": target,
                    "model_state_dict": model.state_dict(),
                    "training_protocol_sha256": protocol_hash,
                    "hard_state_integrity_sha256": digest(HARD_INTEGRITY) if variant.startswith("HARD") else None,
                }, checkpoint)
                record = {
                    "schema": "phase6i-mr-head-refit-run-v1.2",
                    "status": "COMPLETE",
                    "family": family,
                    "variant": variant,
                    "training_seed": seed,
                    "held_fold": fold,
                    "target": target,
                    "parameter_count": parameter_count(model),
                    "history": history,
                    "runtime_seconds": time.perf_counter() - run_started,
                    "training_protocol_sha256": protocol_hash,
                    "checkpoint_sha256": digest(checkpoint),
                    "r10_accessed": False,
                    "r11_accessed": False,
                }
                if fold is not None and prediction_path is not None:
                    predictions = predict_holdout(
                        model, live, fold, family=family, data_variant=variant,
                        seed=seed, target="immediate",
                    )
                    atomic_parquet(predictions, prediction_path)
                    record["predictions_sha256"] = digest(prediction_path)
                    prediction_frames.append(predictions)
                atomic_json(record, record_path)
                completed += 1
                print(json.dumps({"event": "refit_complete", "stage": args.stage, "family": family, "variant": variant, "seed": seed, "fold": fold, "runtime_seconds": record["runtime_seconds"]}), flush=True)
    if args.stage == "hard_oof":
        predictions = pd.concat(prediction_frames, ignore_index=True)
        atomic_parquet(predictions, OUT / "hard_agg_oof_predictions.parquet")
        atomic_parquet(hard_metrics(predictions), OUT / "hard_agg_oof_metrics.parquet")
    progress = {
        "schema": "phase6i-mr-refit-progress-v1.2",
        "status": "COMPLETE",
        "stage": args.stage,
        "completed_runs": completed,
        "expected_runs": expected,
        "elapsed_seconds": time.perf_counter() - started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_protocol_sha256": protocol_hash,
        "hard_state_integrity_sha256": digest(HARD_INTEGRITY),
        "r10_accessed": False,
        "r11_accessed": False,
    }
    atomic_json(progress, OUT / f"{args.stage}_progress.json")
    print(json.dumps(progress, indent=2), flush=True)


if __name__ == "__main__":
    main()
