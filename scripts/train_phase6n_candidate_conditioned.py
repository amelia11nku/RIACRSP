#!/usr/bin/env python3
"""Train the preregistered Phase 6N nested whole-instance OOF ensemble."""

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
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_clgri.analysis.phase6l_legacy_score import (  # noqa: E402
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
)
from rcias_clgri.ni.batching import batch_state_samples  # noqa: E402
from rcias_clgri.ni.cache import load_shard_cache  # noqa: E402
from rcias_clgri.ni.encoder import NIModelConfig  # noqa: E402
from rcias_clgri.ni.phase6n_candidate_conditioned import (  # noqa: E402
    FAMILY,
    CandidateConditionedCSGModel,
    candidate_conditioned_loss,
)
from rcias_clgri.ni.scorer import CSGTargetSetScorer  # noqa: E402
from rcias_clgri.ni.tensorize import CSGTensorizer  # noqa: E402
from scripts import train_phase6j_caur as metrics  # noqa: E402


CONFIG = ROOT / "configs/phase6n_candidate_conditioned_csg_v1.json"
PROTOCOL = ROOT / "outputs/phase6n_candidate_conditioned_csg_v1/training/training_protocol.json"
SOURCE = ROOT / "outputs/phase6n_candidate_conditioned_csg_v1/data/combined/r12_expanded_grouped_labels.parquet"
CACHE = ROOT / "outputs/phase6n_candidate_conditioned_csg_v1/tensor_cache"
OUT = ROOT / "outputs/phase6n_candidate_conditioned_csg_v1/training"


@dataclass(frozen=True)
class FeatureTransform:
    vocabularies: dict[str, tuple[str, ...]]
    medians: dict[str, float]
    iqrs: dict[str, float]

    def to_dict(self) -> dict:
        return {
            "categorical_columns": list(CATEGORICAL_COLUMNS),
            "numeric_columns": list(NUMERIC_COLUMNS),
            "vocabularies": {
                key: list(values) for key, values in self.vocabularies.items()
            },
            "medians": self.medians,
            "iqrs": self.iqrs,
            "unknown_category_index": 0,
            "numeric_clip": [-8.0, 8.0],
        }


def fit_feature_transform(frame: pd.DataFrame) -> FeatureTransform:
    missing = set((*CATEGORICAL_COLUMNS, *NUMERIC_COLUMNS)) - set(frame)
    if missing or frame.empty:
        raise ValueError(f"cannot fit Phase 6N transform: missing={sorted(missing)}")
    vocabularies = {
        column: tuple(sorted(frame[column].astype(str).unique()))
        for column in CATEGORICAL_COLUMNS
    }
    medians = {}
    iqrs = {}
    for column in NUMERIC_COLUMNS:
        values = frame[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite Phase 6N feature: {column}")
        medians[column] = float(np.median(values))
        iqrs[column] = max(
            float(np.quantile(values, 0.75) - np.quantile(values, 0.25)), 1e-6
        )
    return FeatureTransform(vocabularies, medians, iqrs)


def transform_features(
    frame: pd.DataFrame, transform: FeatureTransform
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    categorical = []
    supported = np.ones(len(frame), dtype=bool)
    for column in CATEGORICAL_COLUMNS:
        mapping = {
            value: index + 1
            for index, value in enumerate(transform.vocabularies[column])
        }
        encoded = frame[column].astype(str).map(mapping).fillna(0).to_numpy(dtype=np.int64)
        categorical.append(encoded)
        supported &= encoded > 0
    numeric = []
    for column in NUMERIC_COLUMNS:
        values = frame[column].to_numpy(dtype=float)
        robust = (values - transform.medians[column]) / transform.iqrs[column]
        supported &= np.isfinite(robust) & (robust >= -8.0) & (robust <= 8.0)
        numeric.append(np.clip(robust, -8.0, 8.0))
    return (
        np.column_stack(categorical),
        np.column_stack(numeric).astype(np.float32),
        supported,
    )


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def digest(path: Path) -> str:
    return metrics.digest(path)


def validate_protocol() -> dict:
    protocol = load_json(PROTOCOL)
    checks = (
        protocol.get("schema") == "phase6n-training-protocol-v1",
        protocol.get("status") == "FROZEN_BEFORE_FIRST_OPTIMIZER_STEP",
        protocol.get("optimizer_steps_started") is False,
        protocol.get("r13_accessed") is False,
        protocol.get("r14_accessed") is False,
        digest(CONFIG) == protocol["input_hashes"].get("config"),
        digest(SOURCE) == protocol["input_hashes"].get("expanded_grouped_labels"),
        digest(CACHE / "tensor_cache_integrity.json")
        == protocol["input_hashes"].get("tensor_cache_integrity"),
        not (ROOT / "outputs/phase6j_caur/r13_selection/access_ledger.json").exists(),
        not (ROOT / "outputs/phase6j_caur/r14_holdout/access_ledger.json").exists(),
        not (ROOT / "outputs/phase6n_candidate_conditioned_csg_v1/r13_selection/access_ledger.json").exists(),
        not (ROOT / "outputs/phase6n_candidate_conditioned_csg_v1/r14_holdout/access_ledger.json").exists(),
    )
    if not all(checks):
        raise RuntimeError("Phase 6N training protocol or access boundary failed")
    for relative, expected in protocol["code_hashes"].items():
        if digest(ROOT / relative) != expected:
            raise RuntimeError(f"Phase 6N training code changed after freeze: {relative}")
    return protocol


def load_samples() -> dict[str, object]:
    manifest = pd.read_csv(CACHE / "tensor_manifest.csv")
    samples = {}
    for row in manifest.sort_values("instance_id").itertuples(index=False):
        shard, _ = load_shard_cache(
            Path(row.cache_path),
            expected_tensor_schema_hash=str(row.tensor_schema_hash),
            expected_source_shard_sha256=str(row.source_shard_sha256),
        )
        for sample in shard:
            state_id = sample.graph.state_id
            if state_id in samples:
                raise RuntimeError(f"duplicate Phase 6N tensor state: {state_id}")
            samples[state_id] = sample
    if len(samples) != 864:
        raise RuntimeError("Phase 6N training requires exactly 864 tensorized states")
    return samples


def state_frames(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    result = {
        str(state_id): group.sort_values("target_set_id", kind="stable").reset_index(drop=True)
        for state_id, group in frame.groupby("state_id", sort=True)
    }
    if len(result) != 864:
        raise RuntimeError("Phase 6N training requires exactly 864 label groups")
    return result


def load_base_model(protocol: dict) -> CSGTargetSetScorer:
    path = ROOT / protocol["base_checkpoint"]["path"]
    if digest(path) != protocol["base_checkpoint"]["sha256"]:
        raise RuntimeError("Phase 6N base checkpoint changed")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = CSGTargetSetScorer(
        CSGTensorizer(), NIModelConfig(**checkpoint["model_config"])
    )
    model.load_state_dict(checkpoint["model_state"])
    return model


def initialize_model(
    seed: int,
    transform: FeatureTransform,
    protocol: dict,
    device: torch.device,
) -> CandidateConditionedCSGModel:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    categorical_sizes = tuple(
        len(transform.vocabularies[column]) + 1 for column in CATEGORICAL_COLUMNS
    )
    model = CandidateConditionedCSGModel(
        load_base_model(protocol), categorical_sizes, family=FAMILY
    )
    new_modules = (
        model.pooler,
        model.fallback_fusion,
        model.continuation_advantage_head,
        model.beats_fallback_head,
    )
    for parent in new_modules:
        for module in parent.modules():
            if isinstance(module, torch.nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                with torch.no_grad():
                    module.weight[0].zero_()
            elif isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
    total, trainable = model.parameter_counts()
    expected = protocol["model"]
    if total != expected["total_parameters"] or trainable != expected["trainable_parameters"]:
        raise RuntimeError(f"Phase 6N parameter boundary changed: {(total, trainable)}")
    return model.to(device)


def _operation_mask(sample, serialized: str) -> torch.Tensor:
    operation_ids = set(json.loads(str(serialized)))
    unknown = operation_ids - set(sample.graph.node_keys["OP"])
    if unknown:
        raise RuntimeError(f"unknown conditioned operation IDs: {sorted(unknown)[:3]}")
    return torch.tensor(
        [operation_id in operation_ids for operation_id in sample.graph.node_keys["OP"]],
        dtype=torch.bool,
    )


def build_batch(
    state_ids: list[str],
    samples: dict[str, object],
    frames: dict[str, pd.DataFrame],
    transform: FeatureTransform,
    device: torch.device,
) -> dict:
    chosen_samples = [samples[state_id] for state_id in state_ids]
    groups = [frames[state_id] for state_id in state_ids]
    critical_masks = []
    bottleneck_masks = []
    for sample, group in zip(chosen_samples, groups):
        if tuple(group.target_set_id.astype(str)) != sample.actions.target_set_ids:
            raise RuntimeError(f"Phase 6N tensor/label action mismatch: {sample.graph.state_id}")
        if group.critical_operation_ids.astype(str).nunique() != 1:
            raise RuntimeError("critical-operation context varies within a state")
        if group.bottleneck_operation_ids.astype(str).nunique() != 1:
            raise RuntimeError("bottleneck-operation context varies within a state")
        critical_masks.append(_operation_mask(sample, group.critical_operation_ids.iloc[0]))
        bottleneck_masks.append(_operation_mask(sample, group.bottleneck_operation_ids.iloc[0]))
    combined = pd.concat(groups, ignore_index=True)
    categorical, numeric, supported = transform_features(combined, transform)
    fallback_indices = []
    offset = 0
    for group in groups:
        local = np.flatnonzero(group.is_fallback.to_numpy(dtype=bool))
        if len(local) != 1:
            raise RuntimeError("Phase 6N state does not contain exactly one fallback")
        fallback_indices.append(offset + int(local[0]))
        offset += len(group)
    return {
        "batch": batch_state_samples(chosen_samples).to(device),
        "categorical": torch.as_tensor(categorical, dtype=torch.long, device=device),
        "numeric": torch.as_tensor(numeric, dtype=torch.float32, device=device),
        "supported": supported,
        "fallback_indices": torch.as_tensor(fallback_indices, dtype=torch.long, device=device),
        "critical_operation_mask": torch.cat(critical_masks).to(device),
        "bottleneck_operation_mask": torch.cat(bottleneck_masks).to(device),
        "advantage": torch.as_tensor(
            combined.continuation_advantage_mean.to_numpy(dtype=np.float32), device=device
        ),
        "beats": torch.as_tensor(
            combined.beats_fallback.to_numpy(dtype=np.float32), device=device
        ),
        "frame": combined,
    }


@dataclass(frozen=True)
class ObjectiveScales:
    pair_gap_scale: float
    huber_delta: float
    positive_pair_gaps: int

    def to_dict(self) -> dict:
        return {
            "pair_gap_scale": self.pair_gap_scale,
            "huber_delta": self.huber_delta,
            "positive_pair_gaps": self.positive_pair_gaps,
        }


def fit_objective_scales(frame: pd.DataFrame) -> ObjectiveScales:
    positive_gaps = []
    for _, group in frame.groupby("state_id", sort=True):
        values = group.continuation_advantage_mean.to_numpy(dtype=float)
        gaps = np.abs(values[:, None] - values[None, :])
        positive_gaps.extend(gaps[gaps > 1e-12].tolist())
    if not positive_gaps:
        raise ValueError("Phase 6N objective requires positive within-state gaps")
    values = frame.continuation_advantage_mean.to_numpy(dtype=float)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return ObjectiveScales(
        pair_gap_scale=float(np.median(positive_gaps)),
        huber_delta=float(np.clip(mad, 0.005, 0.05)),
        positive_pair_gaps=len(positive_gaps),
    )


def forward_model(model: CandidateConditionedCSGModel, packed: dict):
    return model(
        packed["batch"],
        fallback_action_indices=packed["fallback_indices"],
        categorical=packed["categorical"],
        numeric=packed["numeric"],
        critical_operation_mask=packed["critical_operation_mask"],
        bottleneck_operation_mask=packed["bottleneck_operation_mask"],
    )


def loss_for_batch(model, packed: dict, objective: ObjectiveScales, protocol: dict) -> dict:
    output = forward_model(model, packed)
    weights = protocol["training"]["objective_weights"]
    return candidate_conditioned_loss(
        output.advantage,
        output.beats_fallback_logit,
        packed["advantage"],
        packed["beats"],
        packed["batch"].action_ptr,
        pair_gap_scale=objective.pair_gap_scale,
        huber_delta=objective.huber_delta,
        pairwise_weight=float(weights["pairwise_logistic_advantage"]),
        listnet_weight=float(weights["listnet_state_list"]),
        advantage_huber_weight=float(weights["huber_advantage"]),
        beats_bce_weight=float(weights["bce_beats_fallback"]),
    )


def predict(
    model,
    state_ids: list[str],
    samples,
    frames,
    transform,
    protocol,
    device,
) -> pd.DataFrame:
    rows = []
    width = int(protocol["training"]["state_groups_per_batch"])
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(state_ids), width):
            packed = build_batch(
                state_ids[start:start + width], samples, frames, transform, device
            )
            output = forward_model(model, packed)
            frame = packed["frame"].copy()
            frame["predicted_continuation_advantage"] = output.advantage.float().cpu().numpy()
            frame["predicted_beats_fallback_logit"] = (
                output.beats_fallback_logit.float().cpu().numpy()
            )
            frame["predicted_beats_fallback_probability_raw"] = torch.sigmoid(
                output.beats_fallback_logit
            ).float().cpu().numpy()
            frame["supported"] = packed["supported"]
            rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def prediction_loss(frame: pd.DataFrame, objective: ObjectiveScales, protocol: dict) -> float:
    ordered = []
    ptr = [0]
    for _, group in frame.groupby("state_id", sort=True):
        group = group.sort_values("target_set_id", kind="stable")
        ordered.append(group)
        ptr.append(ptr[-1] + len(group))
    joined = pd.concat(ordered, ignore_index=True)
    loss = candidate_conditioned_loss(
        torch.tensor(joined.predicted_continuation_advantage.to_numpy(dtype=np.float32)),
        torch.tensor(joined.predicted_beats_fallback_logit.to_numpy(dtype=np.float32)),
        torch.tensor(joined.continuation_advantage_mean.to_numpy(dtype=np.float32)),
        torch.tensor(joined.beats_fallback.to_numpy(dtype=np.float32)),
        torch.tensor(ptr, dtype=torch.long),
        pair_gap_scale=objective.pair_gap_scale,
        huber_delta=objective.huber_delta,
        **{
            "pairwise_weight": float(protocol["training"]["objective_weights"]["pairwise_logistic_advantage"]),
            "listnet_weight": float(protocol["training"]["objective_weights"]["listnet_state_list"]),
            "advantage_huber_weight": float(protocol["training"]["objective_weights"]["huber_advantage"]),
            "beats_bce_weight": float(protocol["training"]["objective_weights"]["bce_beats_fallback"]),
        },
    )["loss"]
    return float(loss)


def validation_score(predictions: pd.DataFrame, protocol: dict) -> tuple[float, dict]:
    states = metrics.ranking_state_metrics(
        predictions, "predicted_continuation_advantage"
    )
    lower, upper = metrics.grouped_bootstrap_interval(
        states,
        "selected_lift",
        seed=int(protocol["bootstrap"]["seed"]),
        resamples=int(protocol["bootstrap"]["resamples"]),
    )
    return lower, {
        "selected_lift": float(states.selected_lift.mean()),
        "selected_lift_lcb": lower,
        "selected_lift_ucb": upper,
        "spearman": float(states.spearman.mean()),
        "pairwise_accuracy": float(states.pairwise_accuracy.mean()),
        "ndcg_at_1": float(states.ndcg_at_1.mean()),
    }


def optimizer_for(model: CandidateConditionedCSGModel, protocol: dict):
    final_block = list(model.state_encoder.layers[-1].parameters())
    final_ids = {id(parameter) for parameter in final_block}
    new_parameters = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in final_ids
    ]
    return torch.optim.AdamW(
        [
            {
                "params": new_parameters,
                "lr": float(protocol["training"]["learning_rate_new_modules"]),
            },
            {
                "params": final_block,
                "lr": float(protocol["training"]["learning_rate_final_relation_block"]),
            },
        ],
        weight_decay=float(protocol["training"]["weight_decay"]),
    )


def optimize_model(
    seed: int,
    train_ids: list[str],
    validation_ids: list[str] | None,
    samples,
    frames,
    transform,
    protocol,
    device,
    *,
    maximum_epochs: int,
    shuffle_salt: int,
    early_stopping: bool,
):
    model = initialize_model(seed, transform, protocol, device)
    optimizer = optimizer_for(model, protocol)
    objective = fit_objective_scales(pd.concat([frames[x] for x in train_ids]))
    patience = int(protocol["training"]["patience"])
    best_key = None
    best_epoch = 0
    best_state = None
    stale = 0
    history = []
    batch_size = int(protocol["training"]["state_groups_per_batch"])
    first_batch_reported = False
    for epoch in range(1, maximum_epochs + 1):
        model.train()
        rng = np.random.default_rng(seed + epoch * 1_000_003 + shuffle_salt)
        order = [train_ids[index] for index in rng.permutation(len(train_ids))]
        values: dict[str, list[float]] = {}
        for start in range(0, len(order), batch_size):
            packed = build_batch(
                order[start:start + batch_size], samples, frames, transform, device
            )
            optimizer.zero_grad(set_to_none=True)
            losses = loss_for_batch(model, packed, objective, protocol)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                float(protocol["training"]["gradient_norm_clip"]),
            )
            optimizer.step()
            if not first_batch_reported:
                print(json.dumps({
                    "event": "phase6n_first_batch",
                    "seed": seed,
                    "loss": float(losses["loss"].detach()),
                    "finite": bool(torch.isfinite(losses["loss"]).item()),
                    "actions": packed["batch"].action_count,
                    "states": packed["batch"].state_count,
                }), flush=True)
                first_batch_reported = True
            for name, value in losses.items():
                if name != "pair_count":
                    values.setdefault(name, []).append(float(value.detach()))
        record = {"epoch": epoch, **{name: float(np.mean(x)) for name, x in values.items()}}
        if validation_ids is not None:
            predictions = predict(
                model, validation_ids, samples, frames, transform, protocol, device
            )
            score, quality = validation_score(predictions, protocol)
            validation_loss = prediction_loss(predictions, objective, protocol)
            record.update({f"validation_{name}": value for name, value in quality.items()})
            record["validation_joint_loss"] = validation_loss
            candidate_key = (score, -validation_loss, -epoch)
        else:
            candidate_key = (float(epoch), 0.0, 0)
        history.append(record)
        improved = best_key is None or candidate_key > best_key
        if improved:
            best_key = candidate_key
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        print(json.dumps({
            "event": "phase6n_epoch",
            "seed": seed,
            "shuffle_salt": shuffle_salt,
            "mode": "INNER_EPOCH_SELECTION" if early_stopping else "OUTER_FINAL_FIT",
            "best_epoch": best_epoch,
            "stale_epochs": stale,
            **record,
        }), flush=True)
        if early_stopping and stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("Phase 6N training produced no checkpoint")
    model.load_state_dict(best_state)
    return model, history, best_epoch, objective


def nested_fold_roles(held_fold: int) -> tuple[int, int]:
    if held_fold not in (0, 1, 2):
        raise ValueError(f"invalid Phase 6N held fold: {held_fold}")
    return (held_fold + 2) % 3, (held_fold + 1) % 3


def train_run(seed, held_fold, samples, frames, full_frame, protocol, device, *, epoch_cap=None):
    outer_train_ids = sorted(
        state_id for state_id, frame in frames.items()
        if int(frame.oof_fold.iloc[0]) != held_fold
    )
    held_ids = sorted(set(frames) - set(outer_train_ids))
    inner_train_fold, inner_validation_fold = nested_fold_roles(held_fold)
    inner_train_ids = sorted(
        state_id for state_id in outer_train_ids
        if int(frames[state_id].oof_fold.iloc[0]) == inner_train_fold
    )
    inner_validation_ids = sorted(set(outer_train_ids) - set(inner_train_ids))
    maximum_epochs = min(
        int(protocol["training"]["maximum_epochs"]),
        epoch_cap if epoch_cap is not None else int(protocol["training"]["maximum_epochs"]),
    )
    inner_transform = fit_feature_transform(
        full_frame[full_frame.state_id.isin(inner_train_ids)]
    )
    inner_model, inner_history, best_epoch, inner_objective = optimize_model(
        seed,
        inner_train_ids,
        inner_validation_ids,
        samples,
        frames,
        inner_transform,
        protocol,
        device,
        maximum_epochs=maximum_epochs,
        shuffle_salt=held_fold * 10_007 + 101,
        early_stopping=True,
    )
    inner_predictions = predict(
        inner_model,
        inner_validation_ids,
        samples,
        frames,
        inner_transform,
        protocol,
        device,
    )
    del inner_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    outer_transform = fit_feature_transform(
        full_frame[full_frame.state_id.isin(outer_train_ids)]
    )
    model, final_history, final_epoch, outer_objective = optimize_model(
        seed,
        outer_train_ids,
        None,
        samples,
        frames,
        outer_transform,
        protocol,
        device,
        maximum_epochs=best_epoch,
        shuffle_salt=held_fold * 10_007 + 202,
        early_stopping=False,
    )
    if final_epoch != best_epoch:
        raise RuntimeError("Phase 6N outer fit did not reproduce selected epoch count")
    held = predict(model, held_ids, samples, frames, outer_transform, protocol, device)
    history = {
        "inner_training_fold": inner_train_fold,
        "inner_validation_fold": inner_validation_fold,
        "inner_epoch_selection": inner_history,
        "outer_final_fit": final_history,
    }
    return (
        model,
        outer_transform,
        history,
        best_epoch,
        held,
        inner_predictions,
        inner_objective,
        outer_objective,
    )


def run_paths(seed: int, held_fold: int, root: Path = OUT):
    stem = root / "oof" / f"seed_{seed}" / f"fold_{held_fold}"
    return (
        Path(f"{stem}.pt"),
        Path(f"{stem}.parquet"),
        Path(f"{stem}_inner_validation.parquet"),
        Path(f"{stem}.json"),
    )


def trainable_state(model) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if name.startswith("state_encoder.layers.1")
        or not name.startswith("state_encoder.")
    }


def valid_run(paths, protocol_sha256: str) -> bool:
    checkpoint, predictions, inner_predictions, record_path = paths
    if not all(path.is_file() for path in paths):
        return False
    try:
        record = load_json(record_path)
    except (OSError, json.JSONDecodeError):
        return False
    return all((
        record.get("status") == "COMPLETE",
        record.get("training_protocol_sha256") == protocol_sha256,
        record.get("checkpoint_sha256") == digest(checkpoint),
        record.get("predictions_sha256") == digest(predictions),
        record.get("inner_predictions_sha256") == digest(inner_predictions),
        record.get("r13_accessed") is False,
        record.get("r14_accessed") is False,
    ))


def summarize(predictions: pd.DataFrame) -> dict:
    key = ["state_id", "target_set_id"]
    seeds = sorted(predictions.training_seed.unique())
    if len(seeds) != 3 or not predictions.groupby(key).training_seed.nunique().eq(3).all():
        raise RuntimeError("Phase 6N OOF seed ensemble is incomplete")
    first = predictions[predictions.training_seed.eq(seeds[0])].sort_values(
        key, kind="stable"
    ).reset_index(drop=True)
    means = predictions.groupby(key, sort=True).agg(
        ensemble_advantage_mean=("predicted_continuation_advantage", "mean"),
        ensemble_advantage_std=("predicted_continuation_advantage", lambda x: float(np.std(x, ddof=0))),
        ensemble_beats_fallback_logit=("predicted_beats_fallback_logit", "mean"),
        ensemble_beats_fallback_probability_raw=("predicted_beats_fallback_probability_raw", "mean"),
        supported=("supported", "all"),
    ).reset_index()
    drop = [
        "training_seed",
        "held_fold",
        "best_epoch",
        "predicted_continuation_advantage",
        "predicted_beats_fallback_logit",
        "predicted_beats_fallback_probability_raw",
        "supported",
    ]
    ensemble = first.drop(columns=[column for column in drop if column in first]).merge(
        means, on=key, how="inner", validate="one_to_one"
    )
    metrics.atomic_parquet(ensemble, OUT / "ensemble_oof.parquet")
    states = metrics.ranking_state_metrics(ensemble, "ensemble_advantage_mean")
    metrics.atomic_parquet(states, OUT / "state_metrics.parquet")

    def scope_summary(scope: pd.DataFrame) -> dict:
        state = metrics.ranking_state_metrics(scope, "ensemble_advantage_mean")
        lower, upper = metrics.grouped_bootstrap_interval(
            state, "selected_lift", seed=727001, resamples=5000
        )
        return {
            "states": int(state.state_id.nunique()),
            "candidates": len(scope),
            "spearman": float(state.spearman.mean()),
            "pairwise_accuracy": float(state.pairwise_accuracy.mean()),
            "ndcg_at_1": float(state.ndcg_at_1.mean()),
            "top1_agreement": float(state.top1_agreement.mean()),
            "selected_lift": float(state.selected_lift.mean()),
            "selected_lift_lcb": lower,
            "selected_lift_ucb": upper,
            "selection_regret": float(state.selection_regret.mean()),
        }

    return {
        "schema": "phase6n-oof-training-summary-v1",
        "status": "COMPLETE",
        "family": FAMILY,
        "expanded": scope_summary(ensemble),
        "common_original_288": scope_summary(
            ensemble[ensemble.phase6n_data_origin.eq("ORIGINAL_PHASE6J_CAUR")]
        ),
        "r13_accessed": False,
        "r14_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--max-new-runs", type=int)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    protocol = validate_protocol()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Phase 6N CUDA training requested but CUDA is unavailable")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(4)
    frame = pd.read_parquet(SOURCE)
    samples = load_samples()
    frames = state_frames(frame)
    root = OUT / "smoke" if args.smoke else OUT
    protocol_sha256 = digest(PROTOCOL)
    predictions = []
    inner_predictions = []
    completed = new_runs = 0
    expected_runs = 9
    started = time.perf_counter()
    print(json.dumps({
        "event": "phase6n_training_start",
        "device": str(device),
        "states": len(frames),
        "candidates": len(frame),
        "smoke": args.smoke,
        "historical_score_calls": 0,
        "r13_accessed": False,
        "r14_accessed": False,
    }), flush=True)
    stop = False
    for seed in protocol["training"]["seeds"]:
        for held_fold in range(3):
            paths = run_paths(int(seed), held_fold, root=root)
            if not args.smoke and valid_run(paths, protocol_sha256):
                predictions.append(pd.read_parquet(paths[1]))
                inner_predictions.append(pd.read_parquet(paths[2]))
                completed += 1
                print(json.dumps({
                    "event": "phase6n_run_skip", "seed": seed, "held_fold": held_fold
                }), flush=True)
                continue
            if args.max_new_runs is not None and new_runs >= args.max_new_runs:
                stop = True
                break
            run_started = time.perf_counter()
            result = train_run(
                int(seed),
                held_fold,
                samples,
                frames,
                frame,
                protocol,
                device,
                epoch_cap=1 if args.smoke else None,
            )
            model, transform, history, best_epoch, held, inner, inner_obj, outer_obj = result
            held["model_family"] = FAMILY
            held["training_seed"] = int(seed)
            held["held_fold"] = held_fold
            held["best_epoch"] = best_epoch
            inner["model_family"] = FAMILY
            inner["training_seed"] = int(seed)
            inner["calibration_outer_fold"] = held_fold
            inner["inner_training_fold"] = history["inner_training_fold"]
            inner["inner_validation_fold"] = history["inner_validation_fold"]
            checkpoint_path, prediction_path, inner_path, record_path = paths
            metrics.save_checkpoint({
                "schema": "phase6n-oof-checkpoint-v1",
                "model_family": FAMILY,
                "training_seed": int(seed),
                "held_fold": held_fold,
                "training_protocol_sha256": protocol_sha256,
                "base_checkpoint_sha256": protocol["base_checkpoint"]["sha256"],
                "feature_transform": transform.to_dict(),
                "objective_scales": outer_obj.to_dict(),
                "trainable_model_state": trainable_state(model),
            }, checkpoint_path)
            metrics.atomic_parquet(held, prediction_path)
            metrics.atomic_parquet(inner, inner_path)
            record = {
                "schema": "phase6n-oof-run-v1",
                "status": "COMPLETE",
                "model_family": FAMILY,
                "training_seed": int(seed),
                "held_fold": held_fold,
                "best_epoch": best_epoch,
                "inner_epochs_run": len(history["inner_epoch_selection"]),
                "outer_final_epochs_run": len(history["outer_final_fit"]),
                "inner_objective_scales": inner_obj.to_dict(),
                "outer_objective_scales": outer_obj.to_dict(),
                "history": history,
                "runtime_seconds": time.perf_counter() - run_started,
                "training_protocol_sha256": protocol_sha256,
                "checkpoint_sha256": digest(checkpoint_path),
                "predictions_sha256": digest(prediction_path),
                "inner_predictions_sha256": digest(inner_path),
                "historical_score_calls": 0,
                "r13_accessed": False,
                "r14_accessed": False,
            }
            metrics.atomic_json(record, record_path)
            predictions.append(held)
            inner_predictions.append(inner)
            completed += 1
            new_runs += 1
            progress = {
                "schema": "phase6n-training-progress-v1",
                "status": "SMOKE_RUNNING" if args.smoke else "RUNNING",
                "completed_runs": completed,
                "expected_runs": expected_runs,
                "new_runs": new_runs,
                "elapsed_seconds": time.perf_counter() - started,
                "last_completed_seed": int(seed),
                "last_completed_fold": held_fold,
                "training_protocol_sha256": protocol_sha256,
                "r13_accessed": False,
                "r14_accessed": False,
            }
            metrics.atomic_json(progress, root / "progress.json")
            print(json.dumps({
                "event": "phase6n_run_complete",
                "seed": seed,
                "held_fold": held_fold,
                "best_epoch": best_epoch,
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
    summary = None
    status = "RUNNING"
    if not args.smoke and completed == expected_runs:
        combined = pd.concat(predictions, ignore_index=True)
        combined_inner = pd.concat(inner_predictions, ignore_index=True)
        metrics.atomic_parquet(combined, OUT / "oof_predictions.parquet")
        metrics.atomic_parquet(combined_inner, OUT / "inner_validation_predictions.parquet")
        summary = summarize(combined)
        metrics.atomic_json(summary, OUT / "oof_summary.json")
        status = "COMPLETE"
    progress = {
        "schema": "phase6n-training-progress-v1",
        "status": "SMOKE_COMPLETE" if args.smoke else status,
        "completed_runs": completed,
        "expected_runs": expected_runs,
        "new_runs": new_runs,
        "elapsed_seconds": time.perf_counter() - started,
        "training_protocol_sha256": protocol_sha256,
        "summary": summary,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "historical_score_calls": 0,
        "r13_accessed": False,
        "r14_accessed": False,
    }
    metrics.atomic_json(progress, root / "progress.json")
    print(json.dumps(progress, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
