#!/usr/bin/env python3
"""Train the preregistered Phase 6I-MR frozen-embedding heads."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_clgri.ni.phase6i_heads import (  # noqa: E402
    Phase6IObjective,
    build_phase6i_head,
    parameter_count,
    phase6i_state_loss,
)
from scripts.prepare_phase6i_mr_training_data import (  # noqa: E402
    atomic_json,
    atomic_parquet,
    digest,
    load_json,
)


PROTOCOL_PATH = ROOT / "outputs/phase6i_mr/frozen/training_protocol.json"
CACHE_ROOT = ROOT / "outputs/phase6i_mr/embedding_cache"
CACHE_INTEGRITY = CACHE_ROOT / "embedding_cache_integrity.json"
OUT = ROOT / "outputs/phase6i_mr/model_training"


@dataclass
class StateTensorData:
    frame: pd.DataFrame
    embeddings: torch.Tensor
    context: torch.Tensor
    raw_targets: torch.Tensor
    normalized_targets: torch.Tensor
    positive: torch.Tensor
    mask: torch.Tensor
    row_indices: np.ndarray
    state_ids: np.ndarray
    folds: np.ndarray

    @property
    def state_count(self) -> int:
        return len(self.state_ids)


def load_source(source: str) -> pd.DataFrame:
    paths = sorted((CACHE_ROOT / source).glob("*.parquet"))
    if not paths:
        raise RuntimeError(f"missing embedding cache source: {source}")
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def state_tensor_data(
    frame: pd.DataFrame,
    *,
    raw_target: str,
    target_scale: float,
    fold_map: dict[str, str],
    context_columns: list[str],
) -> StateTensorData:
    embedding_columns = [f"embedding_{index:03d}" for index in range(128)]
    required = {
        "state_id", "instance_id", "scale", "CF_level", raw_target,
        *embedding_columns, *context_columns,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RuntimeError(f"training cache columns missing: {missing}")
    ordered = frame.sort_values(
        ["state_id", "target_set_id"], kind="stable"
    ).reset_index(drop=True)
    ordered["training_row_index"] = np.arange(len(ordered), dtype=np.int64)
    groups = list(ordered.groupby("state_id", sort=True))
    width = max(len(group) for _, group in groups)
    state_count = len(groups)
    embeddings = np.zeros((state_count, width, 128), dtype=np.float32)
    context = np.zeros((state_count, width, len(context_columns)), dtype=np.float32)
    targets = np.zeros((state_count, width), dtype=np.float32)
    mask = np.zeros((state_count, width), dtype=bool)
    row_indices = np.full((state_count, width), -1, dtype=np.int64)
    state_ids = []
    folds = []
    for state_index, (state_id, group) in enumerate(groups):
        count = len(group)
        embeddings[state_index, :count] = group[embedding_columns].to_numpy(
            dtype=np.float32
        )
        context[state_index, :count] = group[context_columns].to_numpy(
            dtype=np.float32
        )
        targets[state_index, :count] = group[raw_target].to_numpy(dtype=np.float32)
        mask[state_index, :count] = True
        row_indices[state_index, :count] = group.training_row_index.to_numpy(
            dtype=np.int64
        )
        cell = f"{group.scale.iloc[0]}_{group.CF_level.iloc[0]}"
        if "oof_fold" in group and group.oof_fold.notna().all():
            fold = str(group.oof_fold.iloc[0])
        else:
            fold = fold_map[cell]
        state_ids.append(state_id)
        folds.append(fold)
    normalized = np.clip(targets / target_scale, -1.0, 1.0)
    positive = targets > 0
    return StateTensorData(
        ordered,
        torch.from_numpy(embeddings),
        torch.from_numpy(context),
        torch.from_numpy(targets),
        torch.from_numpy(normalized),
        torch.from_numpy(positive),
        torch.from_numpy(mask),
        row_indices,
        np.asarray(state_ids),
        np.asarray(folds),
    )


def take(data: StateTensorData, indices: np.ndarray) -> tuple[torch.Tensor, ...]:
    index = torch.from_numpy(np.asarray(indices, dtype=np.int64))
    return (
        data.embeddings[index],
        data.context[index],
        data.raw_targets[index],
        data.normalized_targets[index],
        data.positive[index],
        data.mask[index],
    )


def cyclic_draw(
    indices: np.ndarray, count: int, rng: np.random.Generator
) -> np.ndarray:
    chunks = []
    remaining = count
    while remaining:
        shuffled = rng.permutation(indices)
        take_count = min(remaining, len(shuffled))
        chunks.append(shuffled[:take_count])
        remaining -= take_count
    return np.concatenate(chunks)


def objective_from(protocol: dict, target: str) -> Phase6IObjective:
    spec = protocol["targets"][target]
    loss = protocol["objective"]
    return Phase6IObjective(
        pairwise_weight=loss["pairwise_weight"],
        listwise_weight=loss["listwise_weight"],
        huber_weight=loss["huber_weight"],
        positive_weight=loss["positive_consistency_weight"],
        pair_gap_scale=spec["pair_gap_scale"],
        pair_weight_min=loss["pair_gap_weight_clip"][0],
        pair_weight_max=loss["pair_gap_weight_clip"][1],
        pair_margin_min=loss["pair_margin_clip"][0],
        pair_margin_max=loss["pair_margin_clip"][1],
        listwise_temperature=spec["listwise_temperature"],
        huber_delta=loss["huber_delta"],
        positive_class_weight=spec["positive_class_weight"],
    )


def batch_loss(
    model: torch.nn.Module,
    data: StateTensorData,
    indices: np.ndarray,
    objective: Phase6IObjective,
) -> dict[str, torch.Tensor]:
    embeddings, context, raw, normalized, positive, mask = take(data, indices)
    predictions = model(embeddings, context)
    return phase6i_state_loss(
        predictions, raw, normalized, positive, mask, objective
    )


def initialize(seed: int, family: str, dropout: float) -> torch.nn.Module:
    torch.manual_seed(seed)
    model = build_phase6i_head(family, dropout=dropout)
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            torch.nn.init.zeros_(module.bias)
    return model


def fit_model(
    family: str,
    data_variant: str,
    seed: int,
    held_fold: str,
    live: StateTensorData,
    old: StateTensorData | None,
    protocol: dict,
    *,
    target: str,
) -> tuple[torch.nn.Module, list[dict[str, float]]]:
    budget = protocol["training_budgets"][target]
    model = initialize(seed, family, budget["dropout"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=budget["learning_rate"],
        weight_decay=budget["weight_decay"],
    )
    objective = objective_from(protocol, target)
    train_live = np.flatnonzero(live.folds != held_fold)
    if not len(train_live):
        raise RuntimeError(f"empty live training fold for {held_fold}")
    if data_variant == "MIXED_OLD_NEW" and old is None:
        raise RuntimeError("mixed training requires the frozen old TRAIN source")
    history = []
    model.train()
    for epoch in range(budget["epochs"]):
        rng = np.random.default_rng(seed + epoch * 1_000_003)
        epoch_values: dict[str, list[float]] = {}
        if data_variant == "R09_ONLY" or data_variant == "CONTINUATION_ONLY":
            batch_size = min(budget["state_batch_size"], len(train_live))
            order = rng.permutation(train_live)
            batches = [order[start:start + batch_size] for start in range(0, len(order), batch_size)]
            jobs = [(indices, None) for indices in batches]
        else:
            assert old is not None
            old_indices = np.arange(old.state_count, dtype=np.int64)
            old_size, live_size = budget["mixed_state_batch_counts"]
            steps = max(
                math.ceil(len(train_live) / live_size),
                math.ceil(len(old_indices) / old_size),
            )
            live_draw = cyclic_draw(train_live, steps * live_size, rng)
            old_draw = cyclic_draw(old_indices, steps * old_size, rng)
            jobs = [
                (
                    live_draw[step * live_size:(step + 1) * live_size],
                    old_draw[step * old_size:(step + 1) * old_size],
                )
                for step in range(steps)
            ]
        for live_indices, old_indices in jobs:
            optimizer.zero_grad(set_to_none=True)
            live_loss = batch_loss(model, live, live_indices, objective)
            if old_indices is None:
                combined = live_loss["loss"]
                values = live_loss
            else:
                assert old is not None
                old_loss = batch_loss(model, old, old_indices, objective)
                weights = protocol["source_loss_weights"]
                combined = (
                    weights["r09_live"] * live_loss["loss"]
                    + weights["earlier_train"] * old_loss["loss"]
                )
                values = {
                    name: (
                        weights["r09_live"] * live_loss[name]
                        + weights["earlier_train"] * old_loss[name]
                    )
                    for name in live_loss
                    if name != "pair_count"
                }
                values["loss"] = combined
            combined.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), budget["gradient_norm_clip"]
            )
            optimizer.step()
            for name, value in values.items():
                if name == "pair_count":
                    continue
                epoch_values.setdefault(name, []).append(float(value.detach()))
        row = {"epoch": epoch + 1}
        row.update({name: float(np.mean(values)) for name, values in epoch_values.items()})
        history.append(row)
    return model, history


def predict_holdout(
    model: torch.nn.Module,
    data: StateTensorData,
    held_fold: str,
    *,
    family: str,
    data_variant: str,
    seed: int,
    target: str,
) -> pd.DataFrame:
    state_indices = np.flatnonzero(data.folds == held_fold)
    model.eval()
    with torch.inference_mode():
        embeddings, context, _, _, _, mask = take(data, state_indices)
        predictions = model(embeddings, context).numpy()
    rows = data.row_indices[state_indices]
    valid = rows >= 0
    result = data.frame.iloc[rows[valid]].copy()
    result["predicted_normalized_value"] = predictions[valid]
    result["model_family"] = family
    result["data_variant"] = data_variant
    result["training_seed"] = seed
    result["held_fold"] = held_fold
    result["target"] = target
    return result


def run_paths(
    family: str, data_variant: str, seed: int, held_fold: str
) -> tuple[Path, Path, Path]:
    base = OUT / "oof" / family / data_variant / f"seed_{seed}" / held_fold
    return base.with_suffix(".pt"), base.with_suffix(".parquet"), base.with_suffix(".json")


def valid_run(paths: tuple[Path, Path, Path], protocol_hash: str) -> bool:
    checkpoint, predictions, record_path = paths
    if not all(path.is_file() for path in paths):
        return False
    try:
        record = load_json(record_path)
    except (OSError, json.JSONDecodeError):
        return False
    return all([
        record.get("status") == "COMPLETE",
        record.get("training_protocol_sha256") == protocol_hash,
        record.get("checkpoint_sha256") == digest(checkpoint),
        record.get("predictions_sha256") == digest(predictions),
        record.get("r10_accessed") is False,
        record.get("r11_accessed") is False,
    ])


def save_checkpoint(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def state_metrics(frame: pd.DataFrame, target_column: str) -> pd.DataFrame:
    rows = []
    for state_id, group in frame.groupby("state_id", sort=True):
        truth = group[target_column].to_numpy(dtype=float)
        score = group.predicted_normalized_value.to_numpy(dtype=float)
        comparable = 0
        concordant = 0.0
        for first in range(len(group)):
            for second in range(first + 1, len(group)):
                truth_delta = truth[first] - truth[second]
                if truth_delta == 0:
                    continue
                comparable += 1
                score_delta = score[first] - score[second]
                concordant += 1.0 if truth_delta * score_delta > 0 else 0.5 if score_delta == 0 else 0.0
        chosen_index = int(np.argmax(score))
        fallback = group[group.candidate_role.eq("ALNS_RELATED_FALLBACK")]
        fallback_value = float(fallback[target_column].iloc[0]) if len(fallback) else math.nan
        rank_order = np.argsort(np.argsort(-truth, kind="stable"), kind="stable")
        relevance = len(group) - rank_order
        predicted_order = np.argsort(-score, kind="stable")

        def ndcg_at(k: int) -> float:
            ideal = np.sort(relevance)[::-1][:k]
            selected = relevance[predicted_order[:k]]
            discount = 1.0 / np.log2(np.arange(2, k + 2))
            return float(np.sum(selected * discount) / np.sum(ideal * discount))

        correlation = spearmanr(truth, score).statistic
        kendall = kendalltau(truth, score).statistic
        rows.append({
            "state_id": state_id,
            "instance_id": group.instance_id.iloc[0],
            "scale": group.scale.iloc[0],
            "CF_level": group.CF_level.iloc[0],
            "search_stage": group.search_stage.iloc[0],
            "spearman": float(correlation) if np.isfinite(correlation) else math.nan,
            "kendall": float(kendall) if np.isfinite(kendall) else math.nan,
            "pairwise_accuracy": concordant / comparable if comparable else math.nan,
            "ndcg_at_1": ndcg_at(1),
            "ndcg_at_2": ndcg_at(min(2, len(group))),
            "top1_agreement": float(chosen_index == int(np.argmax(truth))),
            "selected_value": float(truth[chosen_index]),
            "selected_regret": float(np.max(truth) - truth[chosen_index]),
            "fallback_value": fallback_value,
            "selected_lift_over_fallback": float(truth[chosen_index] - fallback_value),
            "selected_positive": float(truth[chosen_index] > 0),
            "selected_sign_error": float((score[chosen_index] >= 0) != (truth[chosen_index] >= 0)),
        })
    return pd.DataFrame(rows)


def summarize_oof(predictions: pd.DataFrame, target_column: str, target: str) -> None:
    metric_rows = []
    state_outputs = []
    keys = ["model_family", "data_variant", "training_seed"]
    for key, group in predictions.groupby(keys, sort=True):
        state = state_metrics(group, target_column)
        for name, value in zip(keys, key):
            state[name] = value
        state_outputs.append(state)
        instance = state.groupby(["instance_id", "scale"], as_index=False).mean(numeric_only=True)
        scale_spearman = instance.groupby("scale").spearman.mean().to_dict()
        metric_rows.append({
            "target": target,
            "model_family": key[0],
            "data_variant": key[1],
            "training_seed": int(key[2]),
            "action_count": len(group),
            "state_count": state.state_id.nunique(),
            "instance_count": state.instance_id.nunique(),
            "overall_per_instance_spearman": float(instance.groupby("instance_id").spearman.mean().mean()),
            "scale_S_per_instance_spearman": float(scale_spearman.get("S", math.nan)),
            "scale_M_per_instance_spearman": float(scale_spearman.get("M", math.nan)),
            "scale_L_per_instance_spearman": float(scale_spearman.get("L", math.nan)),
            **{
                name: float(state[name].mean())
                for name in [
                    "kendall", "pairwise_accuracy", "ndcg_at_1", "ndcg_at_2",
                    "top1_agreement", "selected_value", "selected_regret",
                    "selected_lift_over_fallback", "selected_positive", "selected_sign_error",
                ]
            },
        })
    metrics = pd.DataFrame(metric_rows).sort_values(keys)
    states = pd.concat(state_outputs, ignore_index=True)
    atomic_parquet(states, OUT / f"{target}_oof_state_metrics.parquet")
    atomic_parquet(metrics, OUT / f"{target}_oof_metrics.parquet")
    if target == "immediate":
        family = metrics.groupby(["model_family", "data_variant"], as_index=False).agg(
            mean_overall_spearman=("overall_per_instance_spearman", "mean"),
            worst_seed_spearman=("overall_per_instance_spearman", "min"),
            seed_std_spearman=("overall_per_instance_spearman", "std"),
            mean_scale_S_spearman=("scale_S_per_instance_spearman", "mean"),
            mean_scale_M_spearman=("scale_M_per_instance_spearman", "mean"),
            mean_scale_L_spearman=("scale_L_per_instance_spearman", "mean"),
        )
        best = family.sort_values(
            ["mean_overall_spearman", "model_family", "data_variant"],
            ascending=[False, True, True],
        ).iloc[0]
        scale_values = [best[f"mean_scale_{scale}_spearman"] for scale in "SML"]
        activate = bool(best.mean_overall_spearman < 0.20 or min(scale_values) < 0)
        atomic_json({
            "schema": "phase6i-mr-u3-activation-decision-v1.2",
            "status": "U3_ACTIVATED" if activate else "U3_NOT_ACTIVATED",
            "rule": "best U1/U2 grouped-OOF R09 overall per-instance Spearman < 0.20 or any scale mean < 0",
            "best_candidate": best.to_dict(),
            "u3_activated": activate,
            "r10_accessed": False,
            "r11_accessed": False,
        }, OUT / "u3_activation_decision.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target", choices=["immediate", "continuation"], default="immediate"
    )
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    protocol = load_json(PROTOCOL_PATH)
    cache_integrity = load_json(CACHE_INTEGRITY)
    if not all([
        protocol.get("status") == "FROZEN_BEFORE_FIRST_MODEL_OPTIMIZER_STEP_AND_R10_R11_ACCESS",
        protocol.get("r10_accessed") is False,
        protocol.get("r11_accessed") is False,
        cache_integrity.get("status") == "PASS",
        cache_integrity.get("r10_accessed") is False,
        cache_integrity.get("r11_accessed") is False,
    ]):
        raise RuntimeError("Phase 6I training inputs are not frozen and leakage-safe")
    for path, expected in protocol["code_hashes"].items():
        if digest(ROOT / path) != expected:
            raise RuntimeError(f"training code changed after protocol freeze: {path}")
    protocol_hash = digest(PROTOCOL_PATH)
    fold_map = {
        cell: fold
        for fold, cells in protocol["grouped_oof_folds"].items()
        for cell in cells
    }
    contexts = protocol["context_columns"]
    if args.target == "immediate":
        live_frame = load_source("r09")
        old_frame = load_source("old_train")
        live = state_tensor_data(
            live_frame,
            raw_target="decoded_immediate_utility",
            target_scale=protocol["targets"]["immediate"]["absolute_p99_scale"],
            fold_map=fold_map,
            context_columns=contexts,
        )
        old = state_tensor_data(
            old_frame,
            raw_target="decoded_immediate_utility",
            target_scale=protocol["targets"]["immediate"]["absolute_p99_scale"],
            fold_map=fold_map,
            context_columns=contexts,
        )
        candidates = [(family, variant) for family in ["U1", "U2"] for variant in ["R09_ONLY", "MIXED_OLD_NEW"]]
        target_column = "decoded_immediate_utility"
    else:
        live_frame = load_source("continuation")
        live = state_tensor_data(
            live_frame,
            raw_target="continuation_value",
            target_scale=protocol["targets"]["continuation"]["absolute_p99_scale"],
            fold_map=fold_map,
            context_columns=contexts,
        )
        old = None
        candidates = [("U2_H_CONTINUATION", "CONTINUATION_ONLY")]
        target_column = "continuation_value"
    expected_runs = len(candidates) * len(protocol["training_seeds"]) * len(protocol["grouped_oof_folds"])
    complete_predictions = []
    completed = 0
    started = time.perf_counter()
    for family, variant in candidates:
        for seed in protocol["training_seeds"]:
            for held_fold in protocol["grouped_oof_folds"]:
                paths = run_paths(family, variant, seed, held_fold)
                if valid_run(paths, protocol_hash):
                    predictions = pd.read_parquet(paths[1])
                    completed += 1
                    complete_predictions.append(predictions)
                    print(json.dumps({"event": "training_skip", "family": family, "variant": variant, "seed": seed, "fold": held_fold}), flush=True)
                    continue
                run_started = time.perf_counter()
                model, history = fit_model(
                    family, variant, seed, held_fold, live, old, protocol,
                    target=args.target,
                )
                predictions = predict_holdout(
                    model, live, held_fold, family=family,
                    data_variant=variant, seed=seed, target=args.target,
                )
                checkpoint, prediction_path, record_path = paths
                save_checkpoint({
                    "schema": "phase6i-mr-head-checkpoint-v1.2",
                    "family": family,
                    "data_variant": variant,
                    "training_seed": seed,
                    "held_fold": held_fold,
                    "target": args.target,
                    "model_state_dict": model.state_dict(),
                    "training_protocol_sha256": protocol_hash,
                }, checkpoint)
                atomic_parquet(predictions, prediction_path)
                record = {
                    "schema": "phase6i-mr-oof-training-run-v1.2",
                    "status": "COMPLETE",
                    "family": family,
                    "data_variant": variant,
                    "training_seed": seed,
                    "held_fold": held_fold,
                    "target": args.target,
                    "parameter_count": parameter_count(model),
                    "train_state_count": int(np.sum(live.folds != held_fold)),
                    "validation_state_count": int(np.sum(live.folds == held_fold)),
                    "history": history,
                    "runtime_seconds": time.perf_counter() - run_started,
                    "training_protocol_sha256": protocol_hash,
                    "checkpoint_sha256": digest(checkpoint),
                    "predictions_sha256": digest(prediction_path),
                    "r10_accessed": False,
                    "r11_accessed": False,
                }
                atomic_json(record, record_path)
                completed += 1
                complete_predictions.append(predictions)
                atomic_json({
                    "schema": "phase6i-mr-head-training-progress-v1.2",
                    "status": "RUNNING",
                    "target": args.target,
                    "completed_runs": completed,
                    "expected_runs": expected_runs,
                    "elapsed_seconds": time.perf_counter() - started,
                    "r10_accessed": False,
                    "r11_accessed": False,
                }, OUT / f"{args.target}_progress.json")
                print(json.dumps({"event": "training_complete", **{key: record[key] for key in ["family", "data_variant", "training_seed", "held_fold", "parameter_count", "runtime_seconds"]}}), flush=True)
    all_predictions = pd.concat(complete_predictions, ignore_index=True)
    atomic_parquet(all_predictions, OUT / f"{args.target}_oof_predictions.parquet")
    summarize_oof(all_predictions, target_column, args.target)
    final = {
        "schema": "phase6i-mr-head-training-progress-v1.2",
        "status": "COMPLETE",
        "target": args.target,
        "completed_runs": completed,
        "expected_runs": expected_runs,
        "prediction_rows": len(all_predictions),
        "elapsed_seconds": time.perf_counter() - started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_protocol_sha256": protocol_hash,
        "r10_accessed": False,
        "r11_accessed": False,
    }
    atomic_json(final, OUT / f"{args.target}_progress.json")
    print(json.dumps(final, indent=2), flush=True)


if __name__ == "__main__":
    main()
