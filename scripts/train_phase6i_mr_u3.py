#!/usr/bin/env python3
"""Train conditional Phase 6I U3 by adapting only the frozen final graph block."""

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
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_clgri.ni.batching import NIBatch, batch_state_samples  # noqa: E402
from rcias_clgri.ni.cache import load_shard_cache  # noqa: E402
from rcias_clgri.ni.dataset import NIStateSample  # noqa: E402
from rcias_clgri.ni.encoder import NIModelConfig  # noqa: E402
from rcias_clgri.ni.phase6i_heads import Phase6IObjective, phase6i_state_loss  # noqa: E402
from rcias_clgri.ni.scorer import CSGTargetSetScorer  # noqa: E402
from rcias_clgri.ni.tensorize import CSGTensorizer  # noqa: E402
from rcias_clgri.ni.utility_losses import within_state_standardize  # noqa: E402
from scripts.prepare_phase6i_mr_training_data import (  # noqa: E402
    atomic_json,
    atomic_parquet,
    digest,
    load_json,
)
from scripts.train_phase6i_mr_heads import state_metrics  # noqa: E402


U3_PROTOCOL_PATH = ROOT / "outputs/phase6i_mr/frozen/u3_training_protocol.json"
PHASE6F_FREEZE = ROOT / "outputs/phase6f/audit/experiment_freeze.json"
R09_CACHE = ROOT / "outputs/phase6i_mr/u3_tensor_cache"
OLD_SELECTION = ROOT / "outputs/phase6i_mr/training_data/old_train_state_selection.csv"
HARD_MANIFEST = ROOT / "outputs/phase6i_mr/hard_state_round1/hard_state_manifest.csv"
EMBEDDING_CACHE = ROOT / "outputs/phase6i_mr/embedding_cache"
OUT = ROOT / "outputs/phase6i_mr/model_training/u3"


def load_r09_samples() -> list[NIStateSample]:
    samples = []
    manifest = pd.read_csv(R09_CACHE / "u3_tensor_manifest.csv")
    for row in manifest.sort_values("instance_id").itertuples(index=False):
        shard, _ = load_shard_cache(
            Path(row.cache_path), expected_tensor_schema_hash=str(row.tensor_schema_hash)
        )
        samples.extend(shard)
    return samples


def load_old_samples() -> list[NIStateSample]:
    selection = pd.read_csv(OLD_SELECTION)
    samples = []
    for cache_path, rows in selection.groupby("cache_path", sort=True):
        selected = set(rows.state_id)
        shard, _ = load_shard_cache(Path(cache_path))
        chosen = [sample for sample in shard if sample.graph.state_id in selected]
        if {sample.graph.state_id for sample in chosen} != selected:
            raise RuntimeError(f"old U3 selection mismatch: {cache_path}")
        samples.extend(chosen)
    return samples


def load_reference_scores() -> dict[tuple[str, str], float]:
    result = {}
    for source in ["r09", "old_train"]:
        for path in sorted((EMBEDDING_CACHE / source).glob("*.parquet")):
            frame = pd.read_parquet(
                path, columns=["state_id", "target_set_id", "frozen_reference_score"]
            )
            for row in frame.itertuples(index=False):
                key = (str(row.state_id), str(row.target_set_id))
                if key in result:
                    raise RuntimeError(f"duplicate frozen score key: {key}")
                result[key] = float(row.frozen_reference_score)
    return result


def load_model(seed: int, protocol: dict, device: torch.device) -> CSGTargetSetScorer:
    freeze = load_json(PHASE6F_FREEZE)
    checkpoint_path = Path(freeze["selected_checkpoint_path"])
    if digest(checkpoint_path) != freeze["selected_checkpoint_sha256"]:
        raise RuntimeError("Phase 6F checkpoint hash mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = CSGTargetSetScorer(
        CSGTensorizer(), NIModelConfig(**checkpoint["model_config"])
    )
    model.load_state_dict(checkpoint["model_state"])
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    prefixes = tuple(protocol["trainable_prefixes"])
    for name, parameter in model.named_parameters():
        if name.startswith(prefixes):
            parameter.requires_grad_(True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    assert model.utility_head is not None
    for module in model.utility_head.modules():
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            torch.nn.init.zeros_(module.bias)
    model.to(device)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if trainable != protocol["trainable_parameter_count"]:
        raise RuntimeError(f"U3 trainable parameter mismatch: {trainable}")
    return model


def objective_from(protocol: dict) -> Phase6IObjective:
    target = protocol["target"]
    loss = protocol["objective"]
    return Phase6IObjective(
        pairwise_weight=loss["pairwise_weight"],
        listwise_weight=loss["listwise_weight"],
        huber_weight=loss["huber_weight"],
        positive_weight=loss["positive_consistency_weight"],
        pair_gap_scale=target["pair_gap_scale"],
        pair_weight_min=loss["pair_gap_weight_clip"][0],
        pair_weight_max=loss["pair_gap_weight_clip"][1],
        pair_margin_min=loss["pair_margin_clip"][0],
        pair_margin_max=loss["pair_margin_clip"][1],
        listwise_temperature=target["listwise_temperature"],
        huber_delta=loss["huber_delta"],
        positive_class_weight=target["positive_class_weight"],
    )


def padded(values: torch.Tensor, action_ptr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pieces = [
        values[start:stop]
        for start, stop in zip(action_ptr[:-1].tolist(), action_ptr[1:].tolist())
    ]
    widths = [len(piece) for piece in pieces]
    result = torch.nn.utils.rnn.pad_sequence(pieces, batch_first=True)
    width_index = torch.arange(result.shape[1], device=result.device).unsqueeze(0)
    mask = width_index < torch.tensor(widths, device=result.device).unsqueeze(1)
    return result, mask


def reference_tensor(
    batch: NIBatch,
    references: dict[tuple[str, str], float],
    device: torch.device,
) -> torch.Tensor:
    values = []
    states = batch.action_to_state.tolist()
    for action_index, target_set_id in enumerate(batch.target_set_ids):
        key = (batch.state_ids[states[action_index]], target_set_id)
        if key not in references:
            raise RuntimeError(f"missing U3 frozen score: {key}")
        values.append(references[key])
    return torch.tensor(values, dtype=torch.float32, device=device)


def state_balanced_score_stability(
    scores: torch.Tensor, references: torch.Tensor, action_ptr: torch.Tensor
) -> torch.Tensor:
    student = within_state_standardize(scores, action_ptr)
    teacher = within_state_standardize(references, action_ptr)
    losses = []
    for start, stop in zip(action_ptr[:-1].tolist(), action_ptr[1:].tolist()):
        losses.append(F.mse_loss(student[start:stop], teacher[start:stop]))
    return torch.stack(losses).mean()


def source_loss(
    model: CSGTargetSetScorer,
    samples: list[NIStateSample],
    objective: Phase6IObjective,
    references: dict[tuple[str, str], float],
    score_weight: float,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    batch = batch_state_samples(samples).to(device)
    output = model(batch)
    if output.utility_predictions is None:
        raise RuntimeError("U3 utility head is missing")
    prediction, mask = padded(output.utility_predictions, batch.action_ptr)
    raw, _ = padded(batch.utility, batch.action_ptr)
    normalized = (raw / objective_from_scale(objective)).clamp(-1, 1)
    positive, _ = padded(batch.positive, batch.action_ptr)
    losses = phase6i_state_loss(
        prediction, raw, normalized, positive, mask, objective
    )
    frozen_scores = reference_tensor(batch, references, device)
    stability = state_balanced_score_stability(
        output.scores, frozen_scores, batch.action_ptr
    )
    total = losses["loss"] + score_weight * stability
    values = {
        name: float(value.detach())
        for name, value in losses.items() if name != "pair_count"
    }
    values["score_stability_loss"] = float(stability.detach())
    values["raw_score_mae"] = float((output.scores - frozen_scores).abs().mean().detach())
    return total, values


def objective_from_scale(objective: Phase6IObjective) -> float:
    # The p99 target scale is stored alongside the objective at module scope by train().
    return float(_TARGET_SCALE)


_TARGET_SCALE = 1.0


def cyclic_indices(
    size: int, count: int, rng: np.random.Generator
) -> np.ndarray:
    chunks = []
    remaining = count
    base = np.arange(size, dtype=np.int64)
    while remaining:
        shuffled = rng.permutation(base)
        amount = min(remaining, size)
        chunks.append(shuffled[:amount])
        remaining -= amount
    return np.concatenate(chunks)


def train(
    seed: int,
    held_fold: str | None,
    r09: list[NIStateSample],
    old: list[NIStateSample],
    hard: list[NIStateSample],
    references: dict[tuple[str, str], float],
    protocol: dict,
    device: torch.device,
) -> tuple[CSGTargetSetScorer, list[dict[str, float]]]:
    global _TARGET_SCALE
    _TARGET_SCALE = float(protocol["target"]["absolute_p99_scale"])
    model = load_model(seed, protocol, device)
    groups = []
    learning_rates = protocol["learning_rates"]
    for prefix, learning_rate in [
        ("state_encoder.layers.1", learning_rates["last_relation_block"]),
        ("action_encoder.projection", learning_rates["action_projection"]),
        ("utility_head", learning_rates["utility_head"]),
    ]:
        groups.append({
            "params": [parameter for name, parameter in model.named_parameters() if name.startswith(prefix)],
            "lr": learning_rate,
        })
    optimizer = torch.optim.AdamW(
        groups, weight_decay=protocol["weight_decay"]
    )
    objective = objective_from(protocol)
    r09_train = [
        sample for sample in r09
        if held_fold is None or sample.structural_metadata["oof_fold"] != held_fold
    ]
    hard_train = [
        sample for sample in hard
        if held_fold is None or sample.structural_metadata["oof_fold"] != held_fold
    ]
    counts = protocol["state_batch_counts"]
    steps = math.ceil(len(r09_train) / counts["r09_live"])
    history = []
    model.train()
    for epoch in range(protocol["epochs"]):
        rng = np.random.default_rng(seed + epoch * 1_000_003)
        draws = {
            "r09_live": cyclic_indices(len(r09_train), steps * counts["r09_live"], rng),
            "earlier_train": cyclic_indices(len(old), steps * counts["earlier_train"], rng),
            "r09_hard_state": cyclic_indices(len(hard_train), steps * counts["r09_hard_state"], rng),
        }
        epoch_values: dict[str, list[float]] = {}
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            totals = []
            source_values = []
            for source, samples in [
                ("r09_live", r09_train),
                ("earlier_train", old),
                ("r09_hard_state", hard_train),
            ]:
                count = counts[source]
                indices = draws[source][step * count:(step + 1) * count]
                total, values = source_loss(
                    model,
                    [samples[index] for index in indices],
                    objective,
                    references,
                    protocol["score_stability_weight"],
                    device,
                )
                weight = protocol["source_loss_weights"][source]
                totals.append(weight * total)
                source_values.append((weight, values))
            loss = sum(totals)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                protocol["gradient_norm_clip"],
            )
            optimizer.step()
            for name in source_values[0][1]:
                value = sum(weight * values[name] for weight, values in source_values)
                epoch_values.setdefault(name, []).append(value)
        record = {"epoch": epoch + 1}
        record.update({name: float(np.mean(values)) for name, values in epoch_values.items()})
        history.append(record)
        print(json.dumps({
            "event": "u3_epoch",
            "seed": seed,
            "held_fold": held_fold,
            **record,
        }), flush=True)
    return model, history


def evaluate(
    model: CSGTargetSetScorer,
    samples: list[NIStateSample],
    source_frame: pd.DataFrame,
    device: torch.device,
    *,
    seed: int,
    held_fold: str,
) -> pd.DataFrame:
    rows = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(samples), 16):
            batch = batch_state_samples(samples[start:start + 16]).to(device)
            output = model(batch)
            utilities = output.utility_predictions.detach().float().cpu().numpy()
            scores = output.scores.detach().float().cpu().numpy()
            state_indices = batch.action_to_state.cpu().numpy()
            for index, target_set_id in enumerate(batch.target_set_ids):
                rows.append({
                    "state_id": batch.state_ids[int(state_indices[index])],
                    "target_set_id": target_set_id,
                    "predicted_normalized_value": float(utilities[index]),
                    "adapted_score": float(scores[index]),
                })
    predictions = pd.DataFrame(rows)
    result = source_frame.merge(
        predictions, on=["state_id", "target_set_id"], how="inner", validate="one_to_one"
    )
    if len(result) != len(predictions):
        raise RuntimeError("U3 evaluation/source join mismatch")
    result["model_family"] = "U3"
    result["data_variant"] = "HARD_AGG_20_60_20"
    result["training_seed"] = seed
    result["held_fold"] = held_fold
    result["target"] = "immediate"
    return result


def trainable_state(model: CSGTargetSetScorer) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if name.startswith(("state_encoder.layers.1", "action_encoder.projection", "utility_head"))
    }


def paths(seed: int, held_fold: str | None) -> tuple[Path, Path, Path | None]:
    directory = OUT / ("oof" if held_fold else "full") / f"seed_{seed}"
    if held_fold:
        directory = directory / held_fold
        return directory.with_suffix(".pt"), directory.with_suffix(".json"), directory.with_suffix(".parquet")
    return directory.with_suffix(".pt"), directory.with_suffix(".json"), None


def valid_run(run_paths: tuple[Path, Path, Path | None], protocol_hash: str) -> bool:
    checkpoint, record_path, prediction_path = run_paths
    required = [checkpoint, record_path] + ([] if prediction_path is None else [prediction_path])
    if not all(path.is_file() for path in required):
        return False
    record = load_json(record_path)
    return all([
        record.get("status") == "COMPLETE",
        record.get("u3_training_protocol_sha256") == protocol_hash,
        record.get("checkpoint_sha256") == digest(checkpoint),
        prediction_path is None or record.get("predictions_sha256") == digest(prediction_path),
        record.get("r10_accessed") is False,
        record.get("r11_accessed") is False,
    ])


def save_checkpoint(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def summarize(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seed, group in predictions.groupby("training_seed", sort=True):
        state = state_metrics(group, "decoded_immediate_utility")
        instance = state.groupby(["instance_id", "scale"], as_index=False).mean(numeric_only=True)
        scales = instance.groupby("scale").spearman.mean().to_dict()
        rows.append({
            "model_family": "U3",
            "data_variant": "HARD_AGG_20_60_20",
            "training_seed": int(seed),
            "overall_per_instance_spearman": float(instance.groupby("instance_id").spearman.mean().mean()),
            "scale_S_per_instance_spearman": float(scales["S"]),
            "scale_M_per_instance_spearman": float(scales["M"]),
            "scale_L_per_instance_spearman": float(scales["L"]),
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
    parser.add_argument("--stage", choices=["oof", "full"], required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("U3 CUDA training was requested but CUDA is unavailable")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    protocol = load_json(U3_PROTOCOL_PATH)
    if not all([
        protocol.get("status") == "FROZEN_BEFORE_FIRST_U3_OPTIMIZER_STEP_AND_R10_R11_ACCESS",
        protocol.get("r10_accessed") is False,
        protocol.get("r11_accessed") is False,
    ]):
        raise RuntimeError("U3 training protocol is not frozen")
    for path, expected in protocol["code_hashes"].items():
        if digest(ROOT / path) != expected:
            raise RuntimeError(f"U3 code changed after freeze: {path}")
    protocol_hash = digest(U3_PROTOCOL_PATH)
    r09 = load_r09_samples()
    old = load_old_samples()
    hard_ids = set(pd.read_csv(HARD_MANIFEST).state_id)
    hard = [sample for sample in r09 if sample.graph.state_id in hard_ids]
    if (len(r09), len(old), len(hard)) != (1620, 540, 360):
        raise RuntimeError("U3 source state cardinality mismatch")
    references = load_reference_scores()
    source_frame = load_source_frame()
    folds: list[str | None] = list(protocol["grouped_oof_folds"]) if args.stage == "oof" else [None]
    predictions = []
    completed = 0
    started = time.perf_counter()
    for seed in protocol["training_seeds"]:
        for held_fold in folds:
            run_paths = paths(seed, held_fold)
            if valid_run(run_paths, protocol_hash):
                completed += 1
                if run_paths[2] is not None:
                    predictions.append(pd.read_parquet(run_paths[2]))
                print(json.dumps({"event": "u3_skip", "seed": seed, "held_fold": held_fold}), flush=True)
                continue
            run_started = time.perf_counter()
            model, history = train(
                seed, held_fold, r09, old, hard, references, protocol, device
            )
            checkpoint_path, record_path, prediction_path = run_paths
            save_checkpoint({
                "schema": "phase6i-mr-u3-checkpoint-v1.2",
                "base_checkpoint_sha256": protocol["base_checkpoint_sha256"],
                "u3_training_protocol_sha256": protocol_hash,
                "training_seed": seed,
                "held_fold": held_fold,
                "trainable_model_state": trainable_state(model),
            }, checkpoint_path)
            record = {
                "schema": "phase6i-mr-u3-training-run-v1.2",
                "status": "COMPLETE",
                "training_seed": seed,
                "held_fold": held_fold,
                "trainable_parameter_count": protocol["trainable_parameter_count"],
                "history": history,
                "runtime_seconds": time.perf_counter() - run_started,
                "u3_training_protocol_sha256": protocol_hash,
                "checkpoint_sha256": digest(checkpoint_path),
                "r10_accessed": False,
                "r11_accessed": False,
            }
            if held_fold is not None and prediction_path is not None:
                held = [sample for sample in r09 if sample.structural_metadata["oof_fold"] == held_fold]
                frame = evaluate(
                    model, held, source_frame, device, seed=seed, held_fold=held_fold
                )
                atomic_parquet(frame, prediction_path)
                record["predictions_sha256"] = digest(prediction_path)
                predictions.append(frame)
            atomic_json(record, record_path)
            completed += 1
            print(json.dumps({"event": "u3_complete", "stage": args.stage, "seed": seed, "held_fold": held_fold, "runtime_seconds": record["runtime_seconds"]}), flush=True)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    if args.stage == "oof":
        combined = pd.concat(predictions, ignore_index=True)
        atomic_parquet(combined, OUT / "u3_oof_predictions.parquet")
        atomic_parquet(summarize(combined), OUT / "u3_oof_metrics.parquet")
    progress = {
        "schema": "phase6i-mr-u3-training-progress-v1.2",
        "status": "COMPLETE",
        "stage": args.stage,
        "completed_runs": completed,
        "expected_runs": len(protocol["training_seeds"]) * len(folds),
        "elapsed_seconds": time.perf_counter() - started,
        "u3_training_protocol_sha256": protocol_hash,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "r10_accessed": False,
        "r11_accessed": False,
    }
    atomic_json(progress, OUT / f"{args.stage}_progress.json")
    print(json.dumps(progress, indent=2), flush=True)


def load_source_frame() -> pd.DataFrame:
    paths = sorted((EMBEDDING_CACHE / "r09").glob("*.parquet"))
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


if __name__ == "__main__":
    main()
