#!/usr/bin/env python3
"""Run the preregistered Phase 6M nested ranker and selective-risk OOF training."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
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

from rcias_clgri.ni.phase6m_selective_risk import (  # noqa: E402
    FAMILY,
    SelectiveRiskMLP,
    SelectorTransform,
    SupportTransform,
    build_selector_feature_frame,
    fit_selector_transform,
    fit_support_transform,
    initialize_selector,
    selective_risk_loss,
    transform_selector_features,
)


CONFIG_PATH = ROOT / "configs/phase6m_selective_confidence_v1.json"
PREREGISTRATION = ROOT / "outputs/phase6m_selective_confidence_v1/preregistration/preregistration.json"
AMENDMENT = ROOT / "outputs/phase6m_selective_confidence_v1/preregistration/amendment_m2_schema.json"
IMPLEMENTATION = ROOT / "outputs/phase6m_selective_confidence_v1/implementation/implementation_protocol_v2.json"
SOURCE = ROOT / "outputs/phase6l_legacy_score_decoupling_v1/data/r12_score_free_grouped_labels.parquet"
SEED_LABELS = ROOT / "outputs/phase6l_legacy_score_decoupling_v1/data/r12_score_free_seed_labels.parquet"
FROZEN_RANKER = ROOT / "outputs/phase6l_legacy_score_decoupling_v1/training/oof_predictions.parquet"
FROZEN_ENSEMBLE = ROOT / "outputs/phase6l_legacy_score_decoupling_v1/training/ensemble_oof.parquet"
FROZEN_SELECTED = ROOT / "outputs/phase6l_legacy_score_decoupling_v1/training/selected_winners.parquet"
OUT = ROOT / "outputs/phase6m_selective_confidence_v1/training"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def save_checkpoint(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(value, temporary)
    temporary.replace(path)


def validate_boundary() -> tuple[dict, dict, str]:
    config = json.loads(CONFIG_PATH.read_text())
    preregistration = json.loads(PREREGISTRATION.read_text())
    amendment = json.loads(AMENDMENT.read_text())
    implementation = json.loads(IMPLEMENTATION.read_text())
    require(config["primary_family"] == FAMILY, "Phase 6M family mismatch")
    require(preregistration["status"] == "FROZEN_BEFORE_PHASE6M_OUTER_OOF_OR_OPTIMIZER_STEP",
            "Phase 6M preregistration is not frozen")
    require(preregistration["input_sha256"][str(CONFIG_PATH.relative_to(ROOT))] == digest(CONFIG_PATH),
            "Phase 6M config changed after preregistration")
    require(amendment["status"] == "FROZEN_SCHEMA_CORRECTION_BEFORE_FIRST_OPTIMIZER_STEP",
            "Phase 6M schema amendment is not frozen")
    require(amendment["base_config_sha256"] == digest(CONFIG_PATH)
            and amendment["base_preregistration_sha256"] == digest(PREREGISTRATION),
            "Phase 6M schema amendment base changed")
    patch = amendment["active_feature_patch"]
    require(patch == {
        "remove_from_structural_numeric": ["bottleneck_proxy"],
        "add_to_categorical_one_hot": ["bottleneck_proxy"],
    }, "unexpected Phase 6M schema amendment")
    config["selector_features"]["structural_numeric"].remove("bottleneck_proxy")
    config["selector_features"]["categorical_one_hot"].append("bottleneck_proxy")
    require(implementation["status"] == "FROZEN_BEFORE_FIRST_PHASE6M_OPTIMIZER_STEP",
            "Phase 6M implementation is not frozen")
    for relative, expected in implementation["code_sha256"].items():
        require(digest(ROOT / relative) == expected, f"Phase 6M implementation changed: {relative}")
    for relative, expected in implementation["input_sha256"].items():
        require(digest(ROOT / relative) == expected, f"Phase 6M frozen input changed: {relative}")
    protected = json.loads((ROOT / implementation["protected_phase6l_manifest"]["path"]).read_text())
    require(digest(ROOT / implementation["protected_phase6l_manifest"]["path"])
            == implementation["protected_phase6l_manifest"]["sha256"],
            "Phase 6L protected manifest changed")
    for relative, expected in protected.items():
        path = ROOT / relative
        require(path.is_file() and digest(path) == expected["sha256"]
                and path.stat().st_size == expected["bytes"],
                f"protected Phase 6L evidence changed: {relative}")
    require(not (ROOT / "outputs/phase6j_caur/r13_selection/access_ledger.json").exists(),
            "R13 access detected")
    require(not (ROOT / "outputs/phase6j_caur/r14_holdout/access_ledger.json").exists(),
            "R14 access detected")
    return config, implementation, digest(IMPLEMENTATION)


def import_phase6l_training():
    """Import only in the standalone worker; the module intentionally rebinds Phase 6J globals."""
    from scripts import train_phase6l_score_free as phase6l
    return phase6l


def inner_ranker_paths(train_fold: int, prediction_fold: int, seed: int, *, root: Path = OUT):
    stem = root / "inner_ranker" / f"train_{train_fold}_predict_{prediction_fold}" / f"seed_{seed}"
    return stem.with_suffix(".pt"), stem.with_suffix(".parquet"), stem.with_suffix(".json")


def inner_cross_fit_pairs(outer_held_fold: int) -> tuple[tuple[int, int], tuple[int, int]]:
    if outer_held_fold not in (0, 1, 2):
        raise ValueError(f"invalid outer held fold: {outer_held_fold}")
    allowed = sorted({0, 1, 2} - {outer_held_fold})
    return ((allowed[0], allowed[1]), (allowed[1], allowed[0]))


def valid_ranker_run(paths: tuple[Path, Path, Path], implementation_sha256: str) -> bool:
    checkpoint, prediction, record_path = paths
    if not all(path.is_file() for path in paths):
        return False
    try:
        record = json.loads(record_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return all((
        record.get("status") == "COMPLETE",
        record.get("implementation_protocol_sha256") == implementation_sha256,
        record.get("checkpoint_sha256") == digest(checkpoint),
        record.get("prediction_sha256") == digest(prediction),
        record.get("r13_accessed") is False,
        record.get("r14_accessed") is False,
    ))


def train_inner_rankers(config: dict, implementation_sha256: str, device: torch.device,
                        *, max_new_runs: int | None = None) -> tuple[int, int]:
    phase6l = import_phase6l_training()
    phase6l_protocol = phase6l.validate_protocol()
    source = pd.read_parquet(SOURCE)
    frames = phase6l.base.state_frames(source)
    samples = phase6l.base.load_samples()
    seeds = [int(seed) for seed in config["ranker"]["seeds"]]
    epochs = int(config["ranker"]["inner_training_epochs"])
    completed = 0
    new_runs = 0
    for prediction_fold in range(3):
        for train_fold in sorted({0, 1, 2} - {prediction_fold}):
            for seed in seeds:
                paths = inner_ranker_paths(train_fold, prediction_fold, seed)
                if valid_ranker_run(paths, implementation_sha256):
                    completed += 1
                    print(json.dumps({"event": "phase6m_inner_ranker_skip", "train_fold": train_fold,
                                      "prediction_fold": prediction_fold, "seed": seed}), flush=True)
                    continue
                if max_new_runs is not None and new_runs >= max_new_runs:
                    return completed, new_runs
                train_ids = sorted(
                    state_id for state_id, frame in frames.items()
                    if int(frame.oof_fold.iloc[0]) == train_fold
                )
                prediction_ids = sorted(
                    state_id for state_id, frame in frames.items()
                    if int(frame.oof_fold.iloc[0]) == prediction_fold
                )
                transform = phase6l.base.fit_feature_transform(
                    source[source.state_id.isin(train_ids)]
                )
                started = time.perf_counter()
                model, history, final_epoch = phase6l.base.optimize_model(
                    phase6l.FAMILY, seed, train_ids, None, samples, frames, transform,
                    phase6l_protocol, device, maximum_epochs=epochs,
                    shuffle_salt=610_000 + train_fold * 10_000 + prediction_fold * 100,
                    early_stopping=False,
                )
                require(final_epoch == epochs, "inner ranker fixed epoch count changed")
                predictions = phase6l.base.predict(
                    model, prediction_ids, samples, frames, transform, phase6l_protocol, device
                )
                predictions["model_family"] = phase6l.FAMILY
                predictions["training_seed"] = seed
                predictions["ranker_train_fold"] = train_fold
                predictions["held_fold"] = prediction_fold
                checkpoint, prediction_path, record_path = paths
                save_checkpoint({
                    "schema": "phase6m-inner-ranker-checkpoint-v1",
                    "family": phase6l.FAMILY, "training_seed": seed,
                    "training_fold": train_fold, "prediction_fold": prediction_fold,
                    "epochs": epochs, "feature_transform": transform.to_dict(),
                    "trainable_model_state": phase6l.base.trainable_state(model),
                    "implementation_protocol_sha256": implementation_sha256,
                }, checkpoint)
                atomic_parquet(predictions, prediction_path)
                record = {
                    "schema": "phase6m-inner-ranker-run-v1", "status": "COMPLETE",
                    "training_seed": seed, "training_fold": train_fold,
                    "prediction_fold": prediction_fold, "epochs": epochs,
                    "states_trained": len(train_ids), "states_predicted": len(prediction_ids),
                    "prediction_rows": len(predictions), "history": history,
                    "runtime_seconds": time.perf_counter() - started,
                    "implementation_protocol_sha256": implementation_sha256,
                    "checkpoint_sha256": digest(checkpoint),
                    "prediction_sha256": digest(prediction_path),
                    "r13_accessed": False, "r14_accessed": False,
                }
                atomic_json(record_path, record)
                completed += 1
                new_runs += 1
                print(json.dumps({
                    "event": "phase6m_inner_ranker_complete", "train_fold": train_fold,
                    "prediction_fold": prediction_fold, "seed": seed,
                    "runtime_seconds": record["runtime_seconds"],
                }), flush=True)
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    return completed, new_runs


def load_inner_predictions(outer_held_fold: int, config: dict,
                           implementation_sha256: str) -> pd.DataFrame:
    frames = []
    for train_fold, prediction_fold in inner_cross_fit_pairs(outer_held_fold):
        for seed in config["ranker"]["seeds"]:
            paths = inner_ranker_paths(train_fold, prediction_fold, int(seed))
            require(valid_ranker_run(paths, implementation_sha256),
                    f"missing valid inner ranker run: {train_fold}->{prediction_fold}, seed {seed}")
            frames.append(pd.read_parquet(paths[1]))
    result = pd.concat(frames, ignore_index=True)
    require(result.state_id.nunique() == 192 and len(result) in (3 * 4539, 3 * 4540),
            f"invalid outer {outer_held_fold} selector-training ranker features")
    return result


def outcome_matrix(features: pd.DataFrame, seed_labels: pd.DataFrame) -> np.ndarray:
    keys = ["state_id", "target_set_id"]
    pivot = seed_labels.pivot(index=keys, columns="continuation_seed", values="continuation_advantage")
    require(pivot.shape[1] == 2 and not pivot.isna().any().any(),
            "Phase 6M requires two complete CRN outcomes per candidate")
    indexed = pd.MultiIndex.from_frame(features[keys])
    values = pivot.loc[indexed].to_numpy(dtype=np.float32)
    require(values.shape == (len(features), 2), "selector outcome alignment failed")
    return values


def state_batches(frame: pd.DataFrame, state_ids: list[str], width: int):
    for start in range(0, len(state_ids), width):
        ids = state_ids[start:start + width]
        indices = []
        ptr = [0]
        for state_id in ids:
            positions = np.flatnonzero(frame.state_id.eq(state_id).to_numpy())
            require(len(positions) > 0, f"missing selector state: {state_id}")
            indices.extend(positions.tolist())
            ptr.append(len(indices))
        yield np.asarray(indices, dtype=int), np.asarray(ptr, dtype=np.int64)


def evaluate_selector(model: SelectiveRiskMLP, matrix: np.ndarray, outcomes: np.ndarray,
                      frame: pd.DataFrame, config: dict, device: torch.device) -> dict[str, float]:
    model.eval()
    totals: dict[str, list[float]] = {}
    state_ids = sorted(frame.state_id.astype(str).unique())
    with torch.inference_mode():
        for indices, ptr in state_batches(frame, state_ids, int(config["selector"]["states_per_batch"])):
            output = model(torch.as_tensor(matrix[indices], device=device))
            values = selective_risk_loss(
                output, torch.as_tensor(outcomes[indices], device=device),
                torch.as_tensor(ptr, device=device),
                gaussian_weight=float(config["selector"]["loss"]["gaussian_negative_log_likelihood"]),
                bce_weight=float(config["selector"]["loss"]["seed_positive_binary_cross_entropy"]),
                huber_weight=float(config["selector"]["loss"]["candidate_mean_huber"]),
                huber_delta=float(config["selector"]["loss"]["huber_delta"]),
            )
            states_in_batch = len(ptr) - 1
            for name, value in values.items():
                totals.setdefault(name, []).extend([float(value)] * states_in_batch)
    return {name: float(np.mean(values)) for name, values in totals.items()}


def fit_selector_model(train: pd.DataFrame, validation: pd.DataFrame | None,
                       transform: SelectorTransform, seed_labels: pd.DataFrame,
                       config: dict, seed: int, device: torch.device, *,
                       maximum_epochs: int, early_stopping: bool, shuffle_salt: int):
    train_matrix = transform_selector_features(train, transform)
    train_outcomes = outcome_matrix(train, seed_labels)
    validation_matrix = (
        transform_selector_features(validation, transform) if validation is not None else None
    )
    validation_outcomes = (
        outcome_matrix(validation, seed_labels) if validation is not None else None
    )
    model = initialize_selector(
        train_matrix.shape[1], seed, dropout=float(config["selector"]["dropout"])
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["selector"]["learning_rate"]),
        weight_decay=float(config["selector"]["weight_decay"]),
    )
    best_loss = math.inf
    best_epoch = 0
    best_state = None
    stale = 0
    history = []
    state_ids = sorted(train.state_id.astype(str).unique())
    width = int(config["selector"]["states_per_batch"])
    for epoch in range(1, maximum_epochs + 1):
        model.train()
        rng = np.random.default_rng(seed + epoch * 1_000_003 + shuffle_salt)
        ordered_ids = [state_ids[index] for index in rng.permutation(len(state_ids))]
        training_values: dict[str, list[float]] = {}
        for indices, ptr in state_batches(train, ordered_ids, width):
            optimizer.zero_grad(set_to_none=True)
            output = model(torch.as_tensor(train_matrix[indices], device=device))
            losses = selective_risk_loss(
                output, torch.as_tensor(train_outcomes[indices], device=device),
                torch.as_tensor(ptr, device=device),
                gaussian_weight=float(config["selector"]["loss"]["gaussian_negative_log_likelihood"]),
                bce_weight=float(config["selector"]["loss"]["seed_positive_binary_cross_entropy"]),
                huber_weight=float(config["selector"]["loss"]["candidate_mean_huber"]),
                huber_delta=float(config["selector"]["loss"]["huber_delta"]),
            )
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["selector"]["gradient_norm_clip"]))
            optimizer.step()
            for name, value in losses.items():
                training_values.setdefault(name, []).append(float(value.detach()))
        record = {"epoch": epoch, **{
            f"training_{name}": float(np.mean(values)) for name, values in training_values.items()
        }}
        if validation is not None:
            validation_values = evaluate_selector(
                model, validation_matrix, validation_outcomes, validation, config, device
            )
            record.update({f"validation_{name}": value for name, value in validation_values.items()})
            score = validation_values["loss"]
        else:
            score = -float(epoch)
        improved = (
            score < best_loss - float(config["selector"]["minimum_validation_loss_improvement"])
            if early_stopping else True
        )
        if improved or best_state is None:
            best_loss = score
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        history.append(record)
        print(json.dumps({
            "event": "phase6m_selector_epoch", "training_seed": seed,
            "mode": "INNER_EPOCH_SELECTION" if early_stopping else "OUTER_FINAL_FIT",
            "best_epoch": best_epoch, "stale_epochs": stale, **record,
        }), flush=True)
        if early_stopping and stale >= int(config["selector"]["patience"]):
            break
    require(best_state is not None, "selector training produced no checkpoint")
    model.load_state_dict(best_state)
    return model, history, best_epoch


def selector_paths(seed: int, held_fold: int, *, root: Path = OUT):
    stem = root / "selector" / f"seed_{seed}" / f"fold_{held_fold}"
    return stem.with_suffix(".pt"), stem.with_suffix(".parquet"), stem.with_suffix(".json")


def valid_selector_run(paths: tuple[Path, Path, Path], implementation_sha256: str) -> bool:
    checkpoint, prediction, record_path = paths
    if not all(path.is_file() for path in paths):
        return False
    try:
        record = json.loads(record_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return all((
        record.get("status") == "COMPLETE",
        record.get("implementation_protocol_sha256") == implementation_sha256,
        record.get("checkpoint_sha256") == digest(checkpoint),
        record.get("prediction_sha256") == digest(prediction),
        record.get("r13_accessed") is False,
        record.get("r14_accessed") is False,
    ))


def predict_selector(model: SelectiveRiskMLP, features: pd.DataFrame,
                     transform: SelectorTransform, device: torch.device) -> pd.DataFrame:
    matrix = transform_selector_features(features, transform)
    model.eval()
    with torch.inference_mode():
        output = model(torch.as_tensor(matrix, device=device))
    result = features.copy()
    result["selector_continuation_mean"] = output.continuation_mean.cpu().numpy()
    result["selector_continuation_scale"] = output.continuation_scale.cpu().numpy()
    result["selector_seed_positive_logit"] = output.seed_positive_logit.cpu().numpy()
    result["selector_seed_positive_probability"] = torch.sigmoid(
        output.seed_positive_logit
    ).cpu().numpy()
    return result


def train_selectors(config: dict, implementation_sha256: str, device: torch.device) -> int:
    source = pd.read_parquet(SOURCE)
    seed_labels = pd.read_parquet(SEED_LABELS)
    frozen_ranker = pd.read_parquet(FROZEN_RANKER)
    completed = 0
    for held_fold in range(3):
        inner_ranker = load_inner_predictions(held_fold, config, implementation_sha256)
        fit_fold = (held_fold + 2) % 3
        validation_fold = (held_fold + 1) % 3
        inner_support = fit_support_transform(source[source.oof_fold.eq(fit_fold)])
        inner_features = build_selector_feature_frame(inner_ranker, inner_support)
        fit_features = inner_features[inner_features.oof_fold.eq(fit_fold)].reset_index(drop=True)
        validation_features = inner_features[
            inner_features.oof_fold.eq(validation_fold)
        ].reset_index(drop=True)
        inner_transform = fit_selector_transform(fit_features)

        outer_source = source[source.oof_fold.ne(held_fold)]
        outer_support = fit_support_transform(outer_source)
        outer_train_features = build_selector_feature_frame(
            inner_ranker, outer_support
        ).reset_index(drop=True)
        outer_transform = fit_selector_transform(outer_train_features)
        held_ranker = frozen_ranker[frozen_ranker.held_fold.eq(held_fold)].copy()
        held_features = build_selector_feature_frame(held_ranker, outer_support)
        require(held_features.state_id.nunique() == 96,
                f"outer held fold {held_fold} does not contain 96 states")
        for seed in config["selector"]["training_seeds"]:
            seed = int(seed)
            paths = selector_paths(seed, held_fold)
            if valid_selector_run(paths, implementation_sha256):
                completed += 1
                print(json.dumps({"event": "phase6m_selector_skip", "seed": seed,
                                  "held_fold": held_fold}), flush=True)
                continue
            started = time.perf_counter()
            inner_model, inner_history, best_epoch = fit_selector_model(
                fit_features, validation_features, inner_transform, seed_labels,
                config, seed, device,
                maximum_epochs=int(config["selector"]["maximum_epochs"]),
                early_stopping=True, shuffle_salt=held_fold * 10_007 + 301,
            )
            del inner_model
            model, final_history, final_epoch = fit_selector_model(
                outer_train_features, None, outer_transform, seed_labels,
                config, seed, device, maximum_epochs=best_epoch,
                early_stopping=False, shuffle_salt=held_fold * 10_007 + 302,
            )
            require(final_epoch == best_epoch, "selector final refit epoch changed")
            held = predict_selector(model, held_features, outer_transform, device)
            labels = source[[
                "state_id", "target_set_id", "continuation_advantage_mean",
                "continuation_advantage_std", "beats_fallback", "immediate_utility",
            ]]
            held = held.merge(labels, on=["state_id", "target_set_id"], validate="one_to_one")
            held["model_family"] = FAMILY
            held["selector_training_seed"] = seed
            held["held_fold"] = held_fold
            held["best_epoch"] = best_epoch
            checkpoint, prediction_path, record_path = paths
            save_checkpoint({
                "schema": "phase6m-selective-risk-checkpoint-v1", "model_family": FAMILY,
                "selector_training_seed": seed, "held_fold": held_fold,
                "best_epoch": best_epoch, "support_transform": outer_support.to_dict(),
                "selector_transform": outer_transform.to_dict(),
                "model_state": {name: value.detach().cpu() for name, value in model.state_dict().items()},
                "implementation_protocol_sha256": implementation_sha256,
            }, checkpoint)
            atomic_parquet(held, prediction_path)
            record = {
                "schema": "phase6m-selective-risk-run-v1", "status": "COMPLETE",
                "model_family": FAMILY, "selector_training_seed": seed,
                "held_fold": held_fold, "best_epoch": best_epoch,
                "inner_fit_fold": fit_fold, "inner_validation_fold": validation_fold,
                "inner_history": inner_history, "outer_final_history": final_history,
                "runtime_seconds": time.perf_counter() - started,
                "implementation_protocol_sha256": implementation_sha256,
                "checkpoint_sha256": digest(checkpoint),
                "prediction_sha256": digest(prediction_path),
                "r13_accessed": False, "r14_accessed": False,
            }
            atomic_json(record_path, record)
            completed += 1
            print(json.dumps({
                "event": "phase6m_selector_complete", "seed": seed,
                "held_fold": held_fold, "best_epoch": best_epoch,
                "runtime_seconds": record["runtime_seconds"],
            }), flush=True)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return completed


def summarize(config: dict, implementation_sha256: str) -> dict:
    frames = []
    runs = []
    for seed in config["selector"]["training_seeds"]:
        for fold in range(3):
            paths = selector_paths(int(seed), fold)
            require(valid_selector_run(paths, implementation_sha256),
                    f"invalid selector run: seed={seed}, fold={fold}")
            frames.append(pd.read_parquet(paths[1]))
            runs.append(json.loads(paths[2].read_text()))
    predictions = pd.concat(frames, ignore_index=True)
    keys = ["state_id", "target_set_id"]
    require(len(predictions) == 3 * 6809, "Phase 6M selector prediction count changed")
    require(predictions.groupby(keys).selector_training_seed.nunique().eq(3).all(),
            "Phase 6M selector ensemble incomplete")
    first_seed = int(config["selector"]["training_seeds"][0])
    first = predictions[predictions.selector_training_seed.eq(first_seed)].copy()
    first = first.sort_values(keys, kind="stable").reset_index(drop=True)
    aggregate = predictions.groupby(keys, sort=True).agg(
        selector_predictive_mean=("selector_continuation_mean", "mean"),
        selector_mean_epistemic_std=("selector_continuation_mean", lambda x: float(np.std(x, ddof=0))),
        selector_aleatoric_second_moment=(
            "selector_continuation_scale", lambda x: float(np.mean(np.asarray(x, dtype=float) ** 2)
        )),
        selector_mean_second_moment=(
            "selector_continuation_mean", lambda x: float(np.mean(np.asarray(x, dtype=float) ** 2)
        )),
        selector_probability=("selector_seed_positive_probability", "mean"),
        selector_probability_std=("selector_seed_positive_probability", lambda x: float(np.std(x, ddof=0))),
        hard_supported=("hard_supported", "all"),
    ).reset_index()
    drop = [
        "selector_training_seed", "selector_continuation_mean", "selector_continuation_scale",
        "selector_seed_positive_logit", "selector_seed_positive_probability", "hard_supported",
        "best_epoch",
    ]
    ensemble = first.drop(columns=[column for column in drop if column in first]).merge(
        aggregate, on=keys, validate="one_to_one"
    )
    variance = np.maximum(
        ensemble.selector_aleatoric_second_moment
        + ensemble.selector_mean_second_moment
        - ensemble.selector_predictive_mean ** 2
        + ensemble.ranker_advantage_std ** 2,
        0.0,
    )
    ensemble["selector_total_predictive_scale"] = np.sqrt(variance)
    source = pd.read_parquet(SOURCE)
    labels = source[[
        "state_id", "target_set_id", "continuation_advantage_mean",
        "continuation_advantage_std", "beats_fallback", "immediate_utility",
    ]]
    label_columns = set(labels) - set(keys)
    ensemble = ensemble.drop(columns=[column for column in label_columns if column in ensemble])
    ensemble = ensemble.merge(labels, on=keys, validate="one_to_one")
    winners = []
    for state_id, group in ensemble.groupby("state_id", sort=True):
        order = np.lexsort((
            group.target_set_id.astype(str).to_numpy(),
            -group.ranker_advantage_mean.to_numpy(dtype=float),
        ))
        winners.append(group.iloc[[int(order[0])]])
    selected = pd.concat(winners, ignore_index=True)
    frozen_ensemble = pd.read_parquet(FROZEN_ENSEMBLE).sort_values(keys, kind="stable")
    current = ensemble.sort_values(keys, kind="stable")
    require(current[keys].reset_index(drop=True).equals(frozen_ensemble[keys].reset_index(drop=True)),
            "Phase 6M candidate identity/order drift")
    require(np.allclose(
        current.ranker_advantage_mean, frozen_ensemble.ensemble_advantage_mean,
        atol=1e-7, rtol=1e-7,
    ), "Phase 6M held ranker output changed")
    frozen_selected = pd.read_parquet(FROZEN_SELECTED).sort_values("state_id", kind="stable")
    require(selected.sort_values("state_id", kind="stable").target_set_id.tolist()
            == frozen_selected.target_set_id.tolist(), "Phase 6M ranker winner drift")
    atomic_parquet(predictions, OUT / "oof_predictions.parquet")
    atomic_parquet(ensemble, OUT / "ensemble_oof.parquet")
    atomic_parquet(selected, OUT / "selected_winners.parquet")
    summary = {
        "schema": "phase6m-selective-risk-oof-training-summary-v1",
        "status": "M3_COMPLETE",
        "model_family": FAMILY,
        "inner_ranker_runs": 18,
        "selector_runs": 9,
        "prediction_rows": len(predictions),
        "ensemble_candidates": len(ensemble),
        "states": selected.state_id.nunique(),
        "ranker_candidate_identity_preserved": True,
        "ranker_winner_identity_preserved": True,
        "historical_score_online_forward_calls": 0,
        "implementation_protocol_sha256": implementation_sha256,
        "selector_runs_summary": [{
            "selector_training_seed": item["selector_training_seed"],
            "held_fold": item["held_fold"], "best_epoch": item["best_epoch"],
            "runtime_seconds": item["runtime_seconds"],
            "checkpoint_sha256": item["checkpoint_sha256"],
            "prediction_sha256": item["prediction_sha256"],
        } for item in runs],
        "r13_accessed": False, "r14_accessed": False,
    }
    atomic_json(OUT / "oof_summary.json", summary)
    return summary


def preflight(config: dict) -> dict:
    source = pd.read_parquet(SOURCE)
    frozen = pd.read_parquet(FROZEN_RANKER)
    transform = fit_support_transform(source[source.oof_fold.ne(0)])
    features = build_selector_feature_frame(frozen[frozen.held_fold.eq(0)], transform)
    selector_transform = fit_selector_transform(features)
    matrix = transform_selector_features(features, selector_transform)
    return {
        "status": "PREFLIGHT_PASS", "states": features.state_id.nunique(),
        "candidates": len(features), "input_dim": matrix.shape[1],
        "hard_support_rate": float(features.hard_supported.mean()),
        "historical_score_online_forward_calls": 0,
        "r13_accessed": False, "r14_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--max-new-ranker-runs", type=int)
    args = parser.parse_args()
    config, _, implementation_sha256 = validate_boundary()
    if args.preflight:
        print(json.dumps(preflight(config), indent=2, sort_keys=True))
        return
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Phase 6M CUDA training requested but CUDA is unavailable")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(4)
    started = time.perf_counter()
    completed_rankers, new_rankers = train_inner_rankers(
        config, implementation_sha256, device, max_new_runs=args.max_new_ranker_runs
    )
    status = "RUNNING"
    summary = None
    selectors = 0
    if completed_rankers == 18:
        selectors = train_selectors(config, implementation_sha256, device)
        require(selectors == 9, "Phase 6M selector runs incomplete")
        summary = summarize(config, implementation_sha256)
        status = "COMPLETE"
    progress = {
        "schema": "phase6m-selective-risk-training-progress-v1", "status": status,
        "completed_inner_ranker_runs": completed_rankers, "expected_inner_ranker_runs": 18,
        "new_inner_ranker_runs": new_rankers, "completed_selector_runs": selectors,
        "expected_selector_runs": 9, "elapsed_seconds": time.perf_counter() - started,
        "implementation_protocol_sha256": implementation_sha256, "summary": summary,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "r13_accessed": False, "r14_accessed": False,
    }
    atomic_json(OUT / "progress.json", progress)
    print(json.dumps(progress, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
