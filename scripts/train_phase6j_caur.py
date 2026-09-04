#!/usr/bin/env python3
"""Train and evaluate Phase 6J J1/J2 with grouped R12 out-of-fold evidence."""

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
from scipy.stats import spearmanr
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_clgri.analysis.phase6j_caur import grouped_oof_fold  # noqa: E402
from rcias_clgri.ni.batching import batch_state_samples  # noqa: E402
from rcias_clgri.ni.cache import load_shard_cache  # noqa: E402
from rcias_clgri.ni.calibration import (  # noqa: E402
    calibration_metrics,
    fit_probability_calibrator,
)
from rcias_clgri.ni.encoder import NIModelConfig  # noqa: E402
from rcias_clgri.ni.phase6j_caur_model import (  # noqa: E402
    CAURModel,
    caur_grouped_state_loss,
)
from rcias_clgri.ni.scorer import CSGTargetSetScorer  # noqa: E402
from rcias_clgri.ni.tensorize import CSGTensorizer  # noqa: E402
from scripts.run_phase6j_caur_pilot import (  # noqa: E402
    atomic_csv,
    atomic_json,
    atomic_parquet,
    digest,
    load_json,
)


CONFIG_PATH = ROOT / "configs/phase6j_caur.json"
PROTOCOL_PATH = ROOT / "outputs/phase6j_caur/frozen/r12_training_protocol.json"
COLLECTION = ROOT / "outputs/phase6j_caur/r12_collection"
SOURCE_PATH = COLLECTION / "r12_grouped_labels.parquet"
CACHE = ROOT / "outputs/phase6j_caur/tensor_cache"
OUT = ROOT / "outputs/phase6j_caur/training"

CATEGORICAL_COLUMNS = (
    "primary_origin_rule",
    "origin_destroy_operator",
    "origin_family",
)
NUMERIC_COLUMNS = (
    "origin_rule_count",
    "origin_family_count",
    "destroy_target_cardinality",
    "destroy_target_fraction",
    "fallback_overlap_fraction",
    "fallback_jaccard",
    "critical_overlap_fraction",
    "bottleneck_overlap_fraction",
    "best_frozen_score_jaccard",
    "normalized_frozen_score_rank",
    "normalized_diversity_rank",
    "is_fallback",
)
FAMILIES = ("J1_CONT_FROZEN", "J2_CONT_LASTBLOCK")


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
        raise ValueError(f"cannot fit CAUR feature transform: missing={sorted(missing)}")
    vocabularies = {
        column: tuple(sorted(frame[column].astype(str).unique()))
        for column in CATEGORICAL_COLUMNS
    }
    medians = {}
    iqrs = {}
    for column in NUMERIC_COLUMNS:
        values = frame[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite CAUR numeric feature: {column}")
        medians[column] = float(np.median(values))
        iqrs[column] = max(
            float(np.quantile(values, 0.75) - np.quantile(values, 0.25)),
            1e-6,
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
        values = frame[column].astype(str)
        encoded = values.map(mapping).fillna(0).to_numpy(dtype=np.int64)
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


def grouped_bootstrap_interval(
    state_frame: pd.DataFrame,
    value_column: str,
    *,
    seed: int,
    resamples: int,
) -> tuple[float, float]:
    grouped = state_frame.groupby("instance_id")[value_column].mean()
    values = grouped.to_numpy(dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("grouped bootstrap requires finite instance values")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(resamples, len(values)))
    means = values[draws].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def validate_protocol() -> dict:
    protocol = load_json(PROTOCOL_PATH)
    checks = (
        protocol.get("schema") == "phase6j-caur-r12-training-protocol-v1",
        protocol.get("status") == "FROZEN_BEFORE_FIRST_OPTIMIZER_STEP",
        protocol.get("r13_accessed") is False,
        protocol.get("r14_accessed") is False,
        digest(CONFIG_PATH) == protocol.get("config_sha256"),
        digest(SOURCE_PATH) == protocol["input_hashes"].get("r12_grouped_labels"),
        digest(CACHE / "tensor_cache_integrity.json")
        == protocol["input_hashes"].get("tensor_cache_integrity"),
        not (ROOT / "outputs/phase6j_caur/r13_selection/access_ledger.json").exists(),
        not (ROOT / "outputs/phase6j_caur/r14_holdout/access_ledger.json").exists(),
    )
    if not all(checks):
        raise RuntimeError("Phase 6J training protocol or access boundary failed")
    for relative, expected in protocol["code_hashes"].items():
        if digest(ROOT / relative) != expected:
            raise RuntimeError(f"training code changed after freeze: {relative}")
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
                raise RuntimeError(f"duplicate CAUR tensor state: {state_id}")
            samples[state_id] = sample
    if len(samples) != 288:
        raise RuntimeError("CAUR training requires exactly 288 tensorized states")
    return samples


def state_frames(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    result = {}
    for state_id, group in frame.groupby("state_id", sort=True):
        ordered = group.sort_values("target_set_id", kind="stable").reset_index(drop=True)
        result[str(state_id)] = ordered
    if len(result) != 288:
        raise RuntimeError("CAUR training requires exactly 288 label groups")
    return result


def load_base_model(protocol: dict) -> CSGTargetSetScorer:
    checkpoint_path = ROOT / protocol["base_checkpoint"]["path"]
    if digest(checkpoint_path) != protocol["base_checkpoint"]["sha256"]:
        raise RuntimeError("Phase 6F base checkpoint changed")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = CSGTargetSetScorer(
        CSGTensorizer(), NIModelConfig(**checkpoint["model_config"])
    )
    model.load_state_dict(checkpoint["model_state"])
    return model


def initialize_model(
    family: str,
    seed: int,
    transform: FeatureTransform,
    protocol: dict,
    device: torch.device,
) -> CAURModel:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    categorical_sizes = tuple(
        len(transform.vocabularies[column]) + 1 for column in CATEGORICAL_COLUMNS
    )
    model = CAURModel(
        load_base_model(protocol), categorical_sizes, family=family
    )
    for module in model.heads.modules():
        if isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            with torch.no_grad():
                module.weight[0].zero_()
        elif isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            torch.nn.init.zeros_(module.bias)
    total, trainable = model.parameter_counts()
    expected = protocol["families"][family]
    if total != expected["total_parameters"] or trainable != expected["trainable_parameters"]:
        raise RuntimeError(
            f"{family} parameter boundary changed: {(total, trainable)}"
        )
    return model.to(device)


def build_batch(
    state_ids: list[str],
    samples: dict[str, object],
    frames: dict[str, pd.DataFrame],
    transform: FeatureTransform,
    device: torch.device,
):
    chosen_samples = [samples[state_id] for state_id in state_ids]
    batch = batch_state_samples(chosen_samples).to(device)
    groups = [frames[state_id] for state_id in state_ids]
    for sample, group in zip(chosen_samples, groups):
        if tuple(group.target_set_id.astype(str)) != sample.actions.target_set_ids:
            raise RuntimeError(f"CAUR tensor/feature action mismatch: {sample.graph.state_id}")
    combined = pd.concat(groups, ignore_index=True)
    categorical, numeric, supported = transform_features(combined, transform)
    fallback_indices = []
    offset = 0
    for group in groups:
        local = np.flatnonzero(group.is_fallback.to_numpy(dtype=bool))
        if len(local) != 1:
            raise RuntimeError("CAUR state does not contain exactly one fallback")
        fallback_indices.append(offset + int(local[0]))
        offset += len(group)
    return {
        "batch": batch,
        "categorical": torch.as_tensor(categorical, dtype=torch.long, device=device),
        "numeric": torch.as_tensor(numeric, dtype=torch.float32, device=device),
        "supported": supported,
        "fallback_indices": torch.as_tensor(
            fallback_indices, dtype=torch.long, device=device
        ),
        "advantage": torch.as_tensor(
            combined.continuation_advantage_mean.to_numpy(dtype=np.float32),
            device=device,
        ),
        "beats": torch.as_tensor(
            combined.beats_fallback.to_numpy(dtype=np.float32), device=device
        ),
        "immediate": torch.as_tensor(
            combined.immediate_utility.to_numpy(dtype=np.float32), device=device
        ),
        "frame": combined,
    }


def loss_for_batch(model: CAURModel, packed: dict, protocol: dict) -> dict:
    output = model(
        packed["batch"],
        fallback_action_indices=packed["fallback_indices"],
        categorical=packed["categorical"],
        numeric=packed["numeric"],
    )
    objective = protocol["objective"]
    weights = objective["weights"]
    return caur_grouped_state_loss(
        output.advantage,
        output.beats_fallback_logit,
        output.immediate_utility,
        packed["advantage"],
        packed["beats"],
        packed["immediate"],
        packed["batch"].action_ptr,
        gap_scale=float(objective["pair_gap_scale"]),
        immediate_delta=float(objective["immediate_huber_delta"]),
        pairwise_weight=float(weights["pairwise_logistic_advantage"]),
        listnet_weight=float(weights["listnet_state_list"]),
        advantage_huber_weight=float(weights["huber_advantage"]),
        beats_bce_weight=float(weights["bce_beats_fallback"]),
        immediate_huber_weight=float(weights["huber_immediate_utility_auxiliary"]),
        gap_weight_clip=tuple(objective["gap_weight_clip"]),
    )


def predict(
    model: CAURModel,
    state_ids: list[str],
    samples: dict[str, object],
    frames: dict[str, pd.DataFrame],
    transform: FeatureTransform,
    protocol: dict,
    device: torch.device,
) -> pd.DataFrame:
    rows = []
    width = int(protocol["training"]["state_groups_per_batch"])
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(state_ids), width):
            ids = state_ids[start:start + width]
            packed = build_batch(ids, samples, frames, transform, device)
            output = model(
                packed["batch"],
                fallback_action_indices=packed["fallback_indices"],
                categorical=packed["categorical"],
                numeric=packed["numeric"],
            )
            frame = packed["frame"].copy()
            frame["predicted_continuation_advantage"] = (
                output.advantage.detach().float().cpu().numpy()
            )
            frame["predicted_beats_fallback_logit"] = (
                output.beats_fallback_logit.detach().float().cpu().numpy()
            )
            frame["predicted_beats_fallback_probability_raw"] = torch.sigmoid(
                output.beats_fallback_logit
            ).detach().float().cpu().numpy()
            frame["predicted_immediate_utility"] = (
                output.immediate_utility.detach().float().cpu().numpy()
            )
            frame["supported"] = packed["supported"]
            rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def ranking_state_metrics(
    frame: pd.DataFrame, prediction_column: str
) -> pd.DataFrame:
    rows = []
    for state_id, group in frame.groupby("state_id", sort=True):
        truth = group.continuation_advantage_mean.to_numpy(dtype=float)
        score = group[prediction_column].to_numpy(dtype=float)
        comparable = concordant = 0
        for first in range(len(group)):
            for second in range(first + 1, len(group)):
                truth_gap = truth[first] - truth[second]
                if abs(truth_gap) <= 1e-12:
                    continue
                comparable += 1
                score_gap = score[first] - score[second]
                concordant += 1 if truth_gap * score_gap > 0 else 0.5 if score_gap == 0 else 0
        order = np.lexsort((group.target_set_id.astype(str).to_numpy(), -score))
        winner = int(order[0])
        shifted = truth - truth.min()
        ndcg1 = 1.0 if shifted.max() <= 0 else float(shifted[winner] / shifted.max())
        correlation = spearmanr(truth, score).statistic
        rows.append({
            "state_id": str(state_id),
            "instance_id": str(group.instance_id.iloc[0]),
            "scale": str(group.scale.iloc[0]),
            "CF_level": str(group.CF_level.iloc[0]),
            "search_stage": str(group.search_stage.iloc[0]),
            "spearman": float(correlation) if np.isfinite(correlation) else 0.0,
            "pairwise_accuracy": concordant / comparable if comparable else 0.5,
            "ndcg_at_1": ndcg1,
            "top1_agreement": float(winner == int(np.argmax(truth))),
            "selected_lift": float(truth[winner]),
            "selection_regret": float(truth.max() - truth[winner]),
            "selected_target_set_id": str(group.target_set_id.iloc[winner]),
            "selected_origin_family": str(group.origin_family.iloc[winner]),
        })
    return pd.DataFrame(rows)


def validation_score(
    predictions: pd.DataFrame, protocol: dict
) -> tuple[float, dict[str, float]]:
    states = ranking_state_metrics(predictions, "predicted_continuation_advantage")
    lower, upper = grouped_bootstrap_interval(
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


def nested_fold_roles(held_fold: int) -> tuple[int, int]:
    """Return the single inner fit fold and inner epoch-selection fold."""
    if held_fold not in (0, 1, 2):
        raise ValueError(f"invalid held OOF fold: {held_fold}")
    validation_fold = (held_fold + 1) % 3
    training_fold = ({0, 1, 2} - {held_fold, validation_fold}).pop()
    return training_fold, validation_fold


def optimize_model(
    family: str,
    seed: int,
    train_ids: list[str],
    validation_ids: list[str] | None,
    samples: dict[str, object],
    frames: dict[str, pd.DataFrame],
    transform: FeatureTransform,
    protocol: dict,
    device: torch.device,
    *,
    maximum_epochs: int,
    shuffle_salt: int,
    early_stopping: bool,
) -> tuple[CAURModel, list[dict], int]:
    model = initialize_model(family, seed, transform, protocol, device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(protocol["training"]["learning_rate"][family]),
        weight_decay=float(protocol["training"]["weight_decay"]),
    )
    patience = int(protocol["training"]["patience"])
    best_score = -math.inf
    best_epoch = 0
    best_state = None
    stale = 0
    history = []
    batch_size = int(protocol["training"]["state_groups_per_batch"])
    for epoch in range(1, maximum_epochs + 1):
        model.train()
        rng = np.random.default_rng(seed + epoch * 1_000_003 + shuffle_salt)
        order = [train_ids[index] for index in rng.permutation(len(train_ids))]
        epoch_values: dict[str, list[float]] = {}
        for start in range(0, len(order), batch_size):
            packed = build_batch(
                order[start:start + batch_size], samples, frames, transform, device
            )
            optimizer.zero_grad(set_to_none=True)
            losses = loss_for_batch(model, packed, protocol)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                float(protocol["training"]["gradient_norm_clip"]),
            )
            optimizer.step()
            for name, value in losses.items():
                if name != "pair_count":
                    epoch_values.setdefault(name, []).append(float(value.detach()))
        record = {
            "epoch": epoch,
            **{name: float(np.mean(values)) for name, values in epoch_values.items()},
        }
        if validation_ids is not None:
            predictions = predict(
                model, validation_ids, samples, frames, transform, protocol, device
            )
            score, metrics = validation_score(predictions, protocol)
            record.update({f"validation_{name}": value for name, value in metrics.items()})
        else:
            score = float(epoch)
        history.append(record)
        improved = (
            score > best_score + float(protocol["training"]["minimum_lcb_improvement"])
            if early_stopping else True
        )
        if improved or best_state is None:
            best_score = score
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        print(json.dumps({
            "event": "phase6j_caur_epoch",
            "family": family,
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
        raise RuntimeError("CAUR training produced no checkpoint")
    model.load_state_dict(best_state)
    return model, history, best_epoch


def train_run(
    family: str,
    seed: int,
    held_fold: int,
    samples: dict[str, object],
    frames: dict[str, pd.DataFrame],
    full_frame: pd.DataFrame,
    protocol: dict,
    device: torch.device,
    *,
    epoch_cap: int | None = None,
) -> tuple[CAURModel, FeatureTransform, dict, int, pd.DataFrame]:
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
    inner_model, inner_history, best_epoch = optimize_model(
        family,
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
    del inner_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    outer_transform = fit_feature_transform(
        full_frame[full_frame.state_id.isin(outer_train_ids)]
    )
    model, final_history, final_epoch = optimize_model(
        family,
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
        raise RuntimeError("outer final fit did not reproduce the selected epoch count")
    final_predictions = predict(
        model, held_ids, samples, frames, outer_transform, protocol, device
    )
    history = {
        "inner_training_fold": inner_train_fold,
        "inner_validation_fold": inner_validation_fold,
        "inner_epoch_selection": inner_history,
        "outer_final_fit": final_history,
    }
    return model, outer_transform, history, best_epoch, final_predictions


def run_paths(family: str, seed: int, held_fold: int, root: Path = OUT):
    base = root / "oof" / family / f"seed_{seed}" / f"fold_{held_fold}"
    return base.with_suffix(".pt"), base.with_suffix(".parquet"), base.with_suffix(".json")


def save_checkpoint(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(payload, temporary)
    temporary.replace(path)


def valid_run(paths, protocol_sha256: str) -> bool:
    checkpoint, predictions, record_path = paths
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
        record.get("r13_accessed") is False,
        record.get("r14_accessed") is False,
    ))


def trainable_state(model: CAURModel) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if name.startswith("heads.")
        or (
            model.family == "J2_CONT_LASTBLOCK"
            and name.startswith(("base.state_encoder.layers.1", "base.action_encoder.projection"))
        )
    }


def cross_fitted_calibration(
    selected: pd.DataFrame, method: str
) -> tuple[dict, np.ndarray]:
    values = selected.ensemble_beats_fallback_logit.to_numpy(dtype=float)
    labels = selected.continuation_advantage_mean.gt(0).to_numpy(dtype=int)
    folds = selected.oof_fold.to_numpy(dtype=int)
    predictions = np.empty(len(selected), dtype=float)
    for fold in sorted(set(folds)):
        held = folds == fold
        calibrator = fit_probability_calibrator(values[~held], labels[~held], method)
        predictions[held] = calibrator.predict(values[held])
    full = fit_probability_calibrator(values, labels, method)
    return full.to_dict(), predictions


def ensemble_family(predictions: pd.DataFrame, family: str, protocol: dict) -> dict:
    seeds = tuple(int(value) for value in protocol["training"]["seeds"])
    family_rows = predictions[predictions.model_family.eq(family)].copy()
    key = ["state_id", "target_set_id"]
    if (
        tuple(sorted(family_rows.training_seed.unique())) != seeds
        or family_rows.duplicated([*key, "training_seed"]).any()
        or not family_rows.groupby(key).training_seed.nunique().eq(len(seeds)).all()
    ):
        raise RuntimeError(f"incomplete CAUR seed ensemble: {family}")
    first = family_rows[family_rows.training_seed.eq(seeds[0])].copy()
    first = first.sort_values(key, kind="stable").reset_index(drop=True)
    means = family_rows.groupby(key, sort=True).agg(
        ensemble_advantage_mean=("predicted_continuation_advantage", "mean"),
        ensemble_advantage_std=("predicted_continuation_advantage", lambda x: float(np.std(x, ddof=0))),
        ensemble_beats_fallback_logit=("predicted_beats_fallback_logit", "mean"),
        ensemble_beats_fallback_probability_raw=("predicted_beats_fallback_probability_raw", "mean"),
        ensemble_immediate_utility=("predicted_immediate_utility", "mean"),
        supported=("supported", "all"),
    ).reset_index()
    drop = [
        "training_seed", "predicted_continuation_advantage",
        "predicted_beats_fallback_logit",
        "predicted_beats_fallback_probability_raw",
        "predicted_immediate_utility", "supported",
    ]
    ensemble = first.drop(columns=drop).merge(means, on=key, validate="one_to_one")
    ensemble["model_family"] = family
    states = ranking_state_metrics(ensemble, "ensemble_advantage_mean")
    lower, upper = grouped_bootstrap_interval(
        states,
        "selected_lift",
        seed=int(protocol["bootstrap"]["seed"]),
        resamples=int(protocol["bootstrap"]["resamples"]),
    )
    scale = states.groupby("scale").agg(
        mean_spearman=("spearman", "mean"),
        mean_selected_lift=("selected_lift", "mean"),
        state_count=("state_id", "size"),
    ).to_dict("index")
    metrics = {
        "model_family": family,
        "state_count": len(states),
        "action_count": len(ensemble),
        "overall_spearman": float(states.spearman.mean()),
        "pairwise_accuracy": float(states.pairwise_accuracy.mean()),
        "ndcg_at_1": float(states.ndcg_at_1.mean()),
        "top1_agreement": float(states.top1_agreement.mean()),
        "selected_lift": float(states.selected_lift.mean()),
        "selected_lift_lcb": lower,
        "selected_lift_ucb": upper,
        "selection_regret": float(states.selection_regret.mean()),
        "mean_spearman_by_scale": {
            name: float(scale[name]["mean_spearman"]) for name in ("S", "M", "L")
        },
        "mean_selected_lift_by_scale": {
            name: float(scale[name]["mean_selected_lift"]) for name in ("S", "M", "L")
        },
        "selected_origin_counts": states.selected_origin_family.value_counts().to_dict(),
    }
    selected_ids = states[["state_id", "selected_target_set_id"]].rename(
        columns={"selected_target_set_id": "target_set_id"}
    )
    selected = ensemble.merge(selected_ids, on=key, validate="one_to_one")
    methods = ["PLATT"]
    minimum_oof_fit_rows = len(selected) - int(selected.groupby("oof_fold").size().max())
    isotonic_eligible = minimum_oof_fit_rows >= int(
        protocol["calibration"]["isotonic_minimum_selected_rows"]
    )
    if isotonic_eligible:
        methods.append("ISOTONIC")
    calibration_rows = []
    calibrated_by_method = {}
    full_calibrators = {}
    labels = selected.continuation_advantage_mean.gt(0).to_numpy(dtype=int)
    for method in methods:
        full, calibrated = cross_fitted_calibration(selected, method)
        calibrated_by_method[method] = calibrated
        full_calibrators[method] = full
        calibration_rows.append({
            "method": method,
            **calibration_metrics(calibrated, labels),
        })
    calibration_table = pd.DataFrame(calibration_rows).sort_values(
        ["expected_calibration_error", "brier_score", "method"], kind="stable"
    )
    calibration_method = str(calibration_table.iloc[0].method)
    selected["calibrated_probability"] = calibrated_by_method[calibration_method]
    calibration = {
        "selected_rows": len(selected),
        "minimum_cross_fit_train_rows": minimum_oof_fit_rows,
        "isotonic_eligible": isotonic_eligible,
        "selected_method": calibration_method,
        "deployment_calibrator": full_calibrators[calibration_method],
        "metrics": calibration_table.to_dict("records"),
    }
    gate_rows = []
    gate = protocol["gate"]
    for p_min in gate["p_min_grid"]:
        for lcb_lambda in gate["lcb_lambda_grid"]:
            for delta_min in gate["delta_min_grid"]:
                result = selected.copy()
                result["lcb"] = (
                    result.ensemble_advantage_mean
                    - float(lcb_lambda) * result.ensemble_advantage_std
                )
                result["intervened"] = (
                    ~result.is_fallback.astype(bool)
                    & result.calibrated_probability.ge(float(p_min))
                    & result.lcb.gt(float(delta_min))
                    & result.supported.astype(bool)
                    & result.ensemble_immediate_utility.ge(
                        float(gate["immediate_harm_floor"])
                    )
                )
                result["gate_selected_lift"] = result.continuation_advantage_mean.where(
                    result.intervened, 0.0
                )
                oracle = ensemble.groupby("state_id").continuation_advantage_mean.max()
                result["gate_regret"] = result.state_id.map(oracle) - result.gate_selected_lift
                lcb, ucb = grouped_bootstrap_interval(
                    result,
                    "gate_selected_lift",
                    seed=int(protocol["bootstrap"]["seed"]),
                    resamples=int(protocol["bootstrap"]["resamples"]),
                )
                scale_lift = result.groupby("scale").gate_selected_lift.mean().to_dict()
                scale_interventions = result.groupby("scale").intervened.sum().to_dict()
                coverage = {}
                forced = {}
                for scale_name in ("S", "M", "L"):
                    scale_rows = result[result.scale.eq(scale_name)]
                    abstained = scale_rows[
                        ~scale_rows.intervened & ~scale_rows.is_fallback.astype(bool)
                    ].copy()
                    if len(abstained):
                        _, abstention_ucb = grouped_bootstrap_interval(
                            abstained.assign(
                                forced_abstention_lift=abstained.continuation_advantage_mean
                            ),
                            "forced_abstention_lift",
                            seed=int(protocol["bootstrap"]["seed"]),
                            resamples=int(protocol["bootstrap"]["resamples"]),
                        )
                    else:
                        abstention_ucb = math.inf
                    exception = (
                        len(abstained) >= int(gate["forced_abstention_minimum"])
                        and abstention_ucb <= 0.0
                    )
                    coverage[scale_name] = (
                        int(scale_interventions.get(scale_name, 0))
                        >= int(gate["minimum_direct_interventions_per_scale"])
                        or exception
                    )
                    forced[scale_name] = {
                        "count": len(abstained),
                        "upper_confidence_bound": abstention_ucb,
                        "exception_pass": exception,
                    }
                retained = (
                    float(result.gate_selected_lift.mean()) > 0.0
                    and lcb > 0.0
                    and all(float(scale_lift[name]) >= 0.0 for name in ("S", "M", "L"))
                    and all(coverage.values())
                )
                gate_rows.append({
                    "model_family": family,
                    "p_min": float(p_min),
                    "lcb_lambda": float(lcb_lambda),
                    "delta_min": float(delta_min),
                    "interventions": int(result.intervened.sum()),
                    "selected_lift": float(result.gate_selected_lift.mean()),
                    "selected_lift_lcb": lcb,
                    "selected_lift_ucb": ucb,
                    "selection_regret": float(result.gate_regret.mean()),
                    "selected_winner_ece": float(
                        calibration_table.iloc[0].expected_calibration_error
                    ),
                    "scale_S_lift": float(scale_lift["S"]),
                    "scale_M_lift": float(scale_lift["M"]),
                    "scale_L_lift": float(scale_lift["L"]),
                    "scale_S_interventions": int(scale_interventions["S"]),
                    "scale_M_interventions": int(scale_interventions["M"]),
                    "scale_L_interventions": int(scale_interventions["L"]),
                    "coverage_pass": bool(all(coverage.values())),
                    "forced_abstention": json.dumps(forced, sort_keys=True),
                    "retained": bool(retained),
                })
    gate_table = pd.DataFrame(gate_rows)
    retained = gate_table[gate_table.retained].sort_values(
        [
            "selected_lift_lcb", "selection_regret", "selected_winner_ece",
            "interventions", "p_min", "lcb_lambda", "delta_min",
        ],
        ascending=[False, True, True, False, True, True, True],
        kind="stable",
    )
    selected_gate = retained.iloc[0].to_dict() if len(retained) else None
    return {
        "ensemble": ensemble,
        "states": states,
        "metrics": metrics,
        "selected_winners": selected,
        "calibration": calibration,
        "calibration_table": calibration_table,
        "gate_table": gate_table,
        "selected_gate": selected_gate,
    }


def summarize_oof(predictions: pd.DataFrame, protocol: dict) -> dict:
    summaries = {}
    family_rows = []
    seed_rows = []
    for family in FAMILIES:
        result = ensemble_family(predictions, family, protocol)
        summaries[family] = result
        atomic_parquet(result["ensemble"], OUT / f"{family}_ensemble_oof.parquet")
        atomic_parquet(result["states"], OUT / f"{family}_state_metrics.parquet")
        atomic_parquet(result["selected_winners"], OUT / f"{family}_selected_winners.parquet")
        atomic_csv(result["calibration_table"], OUT / f"{family}_calibration_metrics.csv")
        atomic_csv(result["gate_table"], OUT / f"{family}_gate_grid.csv")
        atomic_json({
            "schema": "phase6j-caur-family-oof-summary-v1",
            "model_family": family,
            "metrics": result["metrics"],
            "calibration": result["calibration"],
            "selected_gate": result["selected_gate"],
            "r13_accessed": False,
            "r14_accessed": False,
        }, OUT / f"{family}_oof_summary.json")
        family_rows.append({
            **{key: value for key, value in result["metrics"].items() if not isinstance(value, dict)},
            **{
                f"scale_{scale}_spearman": result["metrics"]["mean_spearman_by_scale"][scale]
                for scale in ("S", "M", "L")
            },
            "selected_winner_ece": result["calibration"]["metrics"][0]["expected_calibration_error"],
            "eligible_gate_count": int(result["gate_table"].retained.sum()),
        })
        for seed, seed_frame in predictions[predictions.model_family.eq(family)].groupby(
            "training_seed", sort=True
        ):
            state = ranking_state_metrics(
                seed_frame, "predicted_continuation_advantage"
            )
            seed_rows.append({
                "model_family": family,
                "training_seed": int(seed),
                "spearman": float(state.spearman.mean()),
                "pairwise_accuracy": float(state.pairwise_accuracy.mean()),
                "ndcg_at_1": float(state.ndcg_at_1.mean()),
                "selected_lift": float(state.selected_lift.mean()),
                "selection_regret": float(state.selection_regret.mean()),
            })
    family_table = pd.DataFrame(family_rows)
    seed_table = pd.DataFrame(seed_rows)
    atomic_csv(family_table, OUT / "family_oof_metrics.csv")
    atomic_csv(seed_table, OUT / "three_seed_stability.csv")
    best = family_table.sort_values(
        ["pairwise_accuracy", "overall_spearman", "model_family"],
        ascending=[False, False, True], kind="stable"
    ).iloc[0]
    activate_j3 = (
        float(best.pairwise_accuracy) < float(protocol["j3_activation"]["pairwise_accuracy_min"])
        or min(float(best[f"scale_{scale}_spearman"]) for scale in ("S", "M", "L")) < 0.0
    )
    activation = {
        "schema": "phase6j-caur-j3-activation-v1",
        "status": "J3_REQUIRED" if activate_j3 else "J3_NOT_REQUIRED",
        "j3_activated": bool(activate_j3),
        "best_regular_family": str(best.model_family),
        "best_pairwise_accuracy": float(best.pairwise_accuracy),
        "best_scale_spearman": {
            scale: float(best[f"scale_{scale}_spearman"]) for scale in ("S", "M", "L")
        },
        "rule": protocol["j3_activation"],
        "r13_accessed": False,
        "r14_accessed": False,
    }
    atomic_json(activation, OUT / "j3_activation_decision.json")
    return activation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--max-new-runs", type=int)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    protocol = validate_protocol()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CAUR CUDA training requested but CUDA is unavailable")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(4)
    frame = pd.read_parquet(SOURCE_PATH)
    frame["oof_fold"] = [
        grouped_oof_fold(str(scale), str(cf))
        for scale, cf in zip(frame.scale, frame.CF_level)
    ]
    samples = load_samples()
    frames = state_frames(frame)
    root = OUT / "smoke" if args.smoke else OUT
    protocol_sha256 = digest(PROTOCOL_PATH)
    predictions = []
    completed = 0
    new_runs = 0
    started = time.perf_counter()
    expected_runs = len(FAMILIES) * len(protocol["training"]["seeds"]) * 3
    stop = False
    for family in FAMILIES:
        for seed in protocol["training"]["seeds"]:
            for held_fold in range(3):
                paths = run_paths(family, int(seed), held_fold, root=root)
                if not args.smoke and valid_run(paths, protocol_sha256):
                    predictions.append(pd.read_parquet(paths[1]))
                    completed += 1
                    print(json.dumps({
                        "event": "phase6j_caur_run_skip",
                        "family": family,
                        "seed": seed,
                        "held_fold": held_fold,
                    }), flush=True)
                    continue
                if args.max_new_runs is not None and new_runs >= args.max_new_runs:
                    stop = True
                    break
                run_started = time.perf_counter()
                model, transform, history, best_epoch, held = train_run(
                    family,
                    int(seed),
                    held_fold,
                    samples,
                    frames,
                    frame,
                    protocol,
                    device,
                    epoch_cap=2 if args.smoke else None,
                )
                held["model_family"] = family
                held["training_seed"] = int(seed)
                held["held_fold"] = held_fold
                held["best_epoch"] = best_epoch
                checkpoint_path, prediction_path, record_path = paths
                save_checkpoint({
                    "schema": "phase6j-caur-oof-checkpoint-v1",
                    "model_family": family,
                    "training_seed": int(seed),
                    "held_fold": held_fold,
                    "training_protocol_sha256": protocol_sha256,
                    "base_checkpoint_sha256": protocol["base_checkpoint"]["sha256"],
                    "feature_transform": transform.to_dict(),
                    "trainable_model_state": trainable_state(model),
                }, checkpoint_path)
                atomic_parquet(held, prediction_path)
                record = {
                    "schema": "phase6j-caur-oof-run-v1",
                    "status": "COMPLETE",
                    "model_family": family,
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
                    "checkpoint_sha256": digest(checkpoint_path),
                    "predictions_sha256": digest(prediction_path),
                    "r13_accessed": False,
                    "r14_accessed": False,
                }
                atomic_json(record, record_path)
                predictions.append(held)
                completed += 1
                new_runs += 1
                print(json.dumps({
                    "event": "phase6j_caur_run_complete",
                    "family": family,
                    "seed": seed,
                    "held_fold": held_fold,
                    "best_epoch": best_epoch,
                    "inner_epochs_run": len(history["inner_epoch_selection"]),
                    "outer_final_epochs_run": len(history["outer_final_fit"]),
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
        if stop:
            break
    status = "RUNNING"
    activation = None
    if not args.smoke and completed == expected_runs:
        combined = pd.concat(predictions, ignore_index=True)
        atomic_parquet(combined, OUT / "oof_predictions.parquet")
        activation = summarize_oof(combined, protocol)
        status = "COMPLETE_J1_J2"
    progress = {
        "schema": "phase6j-caur-oof-training-progress-v1",
        "status": "SMOKE_COMPLETE" if args.smoke else status,
        "completed_runs": completed,
        "expected_runs": expected_runs,
        "new_runs": new_runs,
        "elapsed_seconds": time.perf_counter() - started,
        "training_protocol_sha256": protocol_sha256,
        "j3_activation": activation,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "r13_accessed": False,
        "r14_accessed": False,
    }
    atomic_json(progress, root / "progress.json")
    print(json.dumps(progress, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
