#!/usr/bin/env python3
"""Train the frozen Phase 6O symmetric-inner-validation OOF ensemble."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcias_clgri.ni.phase6o_top_utility import (  # noqa: E402
    FAMILY,
    TopUtilityCSGModel,
    top_utility_loss,
)
from scripts import train_phase6j_caur as metrics  # noqa: E402
from scripts import train_phase6n_candidate_conditioned as phase6n  # noqa: E402


CONFIG = ROOT / "configs/phase6o_neural_shortlist_v1.json"
PROTOCOL = ROOT / "outputs/phase6o_neural_shortlist_v1/training/training_protocol.json"
SOURCE = ROOT / "outputs/phase6o_neural_shortlist_v1/relabeling/training_grouped_labels.parquet"
RAW = ROOT / "outputs/phase6o_neural_shortlist_v1/relabeling/combined_seed_labels.parquet"
CACHE = ROOT / "outputs/phase6n_candidate_conditioned_csg_v1/tensor_cache"
OUT = ROOT / "outputs/phase6o_neural_shortlist_v1/training"
REPORT = ROOT / "docs/reports/phase6o_top_utility_training_report.md"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def digest(path: Path) -> str:
    return metrics.digest(path)


def validate_protocol() -> dict:
    protocol = load_json(PROTOCOL)
    checks = (
        protocol.get("schema") == "phase6o-top-utility-training-protocol-v1",
        protocol.get("status") == "FROZEN_BEFORE_FIRST_OPTIMIZER_STEP",
        protocol.get("optimizer_steps_started") is False,
        protocol.get("r13_accessed") is False,
        protocol.get("r14_accessed") is False,
        digest(CONFIG) == protocol["input_hashes"].get("config"),
        digest(SOURCE) == protocol["input_hashes"].get("training_grouped_labels"),
        digest(RAW) == protocol["input_hashes"].get("combined_seed_labels"),
        digest(CACHE / "tensor_cache_integrity.json")
        == protocol["input_hashes"].get("tensor_cache_integrity"),
        not (ROOT / "outputs/phase6j_caur/r13_selection/access_ledger.json").exists(),
        not (ROOT / "outputs/phase6j_caur/r14_holdout/access_ledger.json").exists(),
        not (ROOT / "outputs/phase6o_neural_shortlist_v1/r13_selection/access_ledger.json").exists(),
        not (ROOT / "outputs/phase6o_neural_shortlist_v1/r14_holdout/access_ledger.json").exists(),
    )
    if not all(checks):
        raise RuntimeError("Phase 6O training protocol or access boundary failed")
    for relative, expected in protocol["code_hashes"].items():
        if digest(ROOT / relative) != expected:
            raise RuntimeError(f"Phase 6O training code changed after freeze: {relative}")
    return protocol


def attach_candidate_noise(frame: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    keys = ["state_id", "target_set_id"]
    noise = raw.groupby(keys, sort=True).continuation_advantage.agg(
        lambda values: float(
            1.4826
            * np.median(np.abs(np.asarray(values, dtype=float) - np.median(values)))
        )
    ).rename("candidate_noise").reset_index()
    result = frame.merge(noise, on=keys, validate="one_to_one")
    if len(result) != len(frame) or not np.isfinite(result.candidate_noise).all():
        raise RuntimeError("Phase 6O candidate-noise attachment failed")
    return result


@dataclass(frozen=True)
class FoldContract:
    noise_margin: float
    policy_temperature: float
    utility_scale: float
    regret_scale: float
    huber_delta: float
    opportunity_scale: float
    state_weights: dict[str, float]
    source_weight_totals: dict[str, float]
    scale_weight_totals: dict[str, float]
    near_candidate_count: int
    retained_pair_count: int

    def to_dict(self) -> dict:
        return {
            "noise_margin": self.noise_margin,
            "policy_temperature": self.policy_temperature,
            "utility_scale": self.utility_scale,
            "regret_scale": self.regret_scale,
            "huber_delta": self.huber_delta,
            "opportunity_scale": self.opportunity_scale,
            "state_weight_count": len(self.state_weights),
            "state_weight_sha256": hashlib_for_mapping(self.state_weights),
            "source_weight_totals": self.source_weight_totals,
            "scale_weight_totals": self.scale_weight_totals,
            "near_candidate_count": self.near_candidate_count,
            "retained_pair_count": self.retained_pair_count,
        }


def hashlib_for_mapping(values: dict[str, float]) -> str:
    import hashlib

    encoded = json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def fit_fold_contract(frame: pd.DataFrame) -> FoldContract:
    if frame.empty or frame.state_id.nunique() == 0:
        raise ValueError("cannot fit empty Phase 6O fold contract")
    margin = float(np.clip(np.quantile(frame.candidate_noise, 0.75), 0.0025, 0.03))
    state_rows = []
    regret_gaps = []
    near_count = retained_pairs = 0
    for state_id, group in frame.groupby("state_id", sort=True):
        values = group.continuation_advantage_mean.to_numpy(dtype=float)
        fallback = group.loc[group.is_fallback.astype(bool), "continuation_advantage_mean"]
        if len(fallback) != 1:
            raise RuntimeError("Phase 6O fold state must have one fallback")
        maximum = float(values.max())
        near = values >= maximum - margin
        rest = ~near
        gaps = values[near, None] - values[None, rest]
        kept = gaps[gaps > margin]
        regret_gaps.extend(kept.tolist())
        near_count += int(near.sum())
        retained_pairs += len(kept)
        q10, q90 = np.quantile(values, [0.1, 0.9])
        state_rows.append({
            "state_id": str(state_id),
            "phase6n_data_origin": str(group.phase6n_data_origin.iloc[0]),
            "instance_id": str(group.instance_id.iloc[0]),
            "scale": str(group.scale.iloc[0]),
            "iqr": float(np.quantile(values, 0.75) - np.quantile(values, 0.25)),
            "spread": maximum - float(values.min()),
            "opportunity": max(float(q90 - q10), maximum - float(fallback.iloc[0]), 0.0),
        })
    states = pd.DataFrame(state_rows)
    positive_spreads = states.loc[states.spread > 0, "spread"].to_numpy(dtype=float)
    if not len(positive_spreads) or not regret_gaps:
        raise RuntimeError("Phase 6O fold lacks positive utility structure")
    positive_opportunity = states.loc[states.opportunity > 0, "opportunity"].to_numpy(dtype=float)
    opportunity_scale = max(float(np.median(positive_opportunity)), 0.005)
    states["opportunity_clipped"] = np.clip(
        states.opportunity / opportunity_scale, 0.5, 2.0
    )
    states["opportunity_normalized"] = states.opportunity_clipped / states.groupby(
        ["phase6n_data_origin", "instance_id"], sort=False
    ).opportunity_clipped.transform("mean")
    source_instances = states.groupby("phase6n_data_origin").instance_id.nunique()
    state_counts = states.groupby(
        ["phase6n_data_origin", "instance_id"]
    ).state_id.transform("count")
    states["base_weight"] = [
        0.5 / int(source_instances[source]) / int(count)
        for source, count in zip(states.phase6n_data_origin, state_counts)
    ]
    states["weight"] = states.base_weight * states.opportunity_normalized
    source_totals = states.groupby("phase6n_data_origin").weight.sum().to_dict()
    if set(source_totals) != {"ORIGINAL_PHASE6J_CAUR", "NEW_ALNS_EXPANSION"}:
        raise RuntimeError("Phase 6O source balance lacks a source")
    if any(not np.isclose(value, 0.5, atol=1e-12) for value in source_totals.values()):
        raise RuntimeError(f"Phase 6O source balance drifted: {source_totals}")
    values = frame.continuation_advantage_mean.to_numpy(dtype=float)
    median = float(np.median(values))
    return FoldContract(
        noise_margin=margin,
        policy_temperature=float(np.clip(np.median(states.iqr), 0.005, 0.05)),
        utility_scale=float(np.clip(np.median(positive_spreads), 0.005, 0.05)),
        regret_scale=float(np.clip(np.median(regret_gaps), 0.0025, 0.05)),
        huber_delta=float(np.clip(np.median(np.abs(values - median)), 0.005, 0.05)),
        opportunity_scale=opportunity_scale,
        state_weights=dict(zip(states.state_id, states.weight)),
        source_weight_totals={key: float(value) for key, value in source_totals.items()},
        scale_weight_totals={key: float(value) for key, value in states.groupby("scale").weight.sum().items()},
        near_candidate_count=near_count,
        retained_pair_count=retained_pairs,
    )


def initialize_model(seed: int, transform, protocol: dict, device: torch.device):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    sizes = tuple(
        len(transform.vocabularies[column]) + 1
        for column in phase6n.CATEGORICAL_COLUMNS
    )
    model = TopUtilityCSGModel(phase6n.load_base_model(protocol), sizes)
    for parent in (
        model.pooler,
        model.fallback_fusion,
        model.continuation_advantage_head,
        model.beats_fallback_head,
    ):
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
    if (total, trainable) != (
        expected["total_parameters"], expected["trainable_parameters"]
    ):
        raise RuntimeError(f"Phase 6O parameter boundary changed: {(total, trainable)}")
    if any(parameter.requires_grad for parameter in model.state_encoder.parameters()):
        raise RuntimeError("Phase 6O encoder is not fully frozen")
    return model.to(device)


def loss_kwargs(protocol: dict, contract: FoldContract) -> dict:
    weights = protocol["training"]["objective_weights"]
    return {
        "noise_margin": contract.noise_margin,
        "policy_temperature": contract.policy_temperature,
        "utility_scale": contract.utility_scale,
        "regret_scale": contract.regret_scale,
        "huber_delta": contract.huber_delta,
        "top_set_weight": float(weights["top_set_cross_entropy"]),
        "regret_weight": float(weights["near_best_vs_rest_regret_logistic"]),
        "expected_utility_weight": float(weights["soft_expected_utility"]),
        "advantage_huber_weight": float(weights["advantage_huber"]),
        "beats_bce_weight": float(weights["beats_fallback_bce"]),
    }


def batch_loss(model, packed: dict, state_ids: list[str], contract, protocol):
    output = phase6n.forward_model(model, packed)
    weights = torch.tensor(
        [contract.state_weights[state_id] for state_id in state_ids],
        dtype=torch.float32,
        device=packed["advantage"].device,
    )
    return top_utility_loss(
        output.advantage,
        output.beats_fallback_logit,
        packed["advantage"],
        packed["beats"],
        packed["batch"].action_ptr,
        weights,
        **loss_kwargs(protocol, contract),
    )


def prediction_loss(frame: pd.DataFrame, contract: FoldContract, protocol: dict) -> float:
    groups = [group.sort_values("target_set_id", kind="stable") for _, group in frame.groupby("state_id", sort=True)]
    joined = pd.concat(groups, ignore_index=True)
    ptr = np.cumsum([0, *[len(group) for group in groups]])
    value = top_utility_loss(
        torch.tensor(joined.predicted_continuation_advantage.to_numpy(dtype=np.float32)),
        torch.tensor(joined.predicted_beats_fallback_logit.to_numpy(dtype=np.float32)),
        torch.tensor(joined.continuation_advantage_mean.to_numpy(dtype=np.float32)),
        torch.tensor(joined.beats_fallback.to_numpy(dtype=np.float32)),
        torch.tensor(ptr, dtype=torch.long),
        torch.ones(len(groups)),
        **loss_kwargs(protocol, contract),
    )["loss"]
    return float(value)


def selection_metrics(frame: pd.DataFrame, margin: float) -> dict[str, float]:
    rows = []
    for _, group in frame.groupby("state_id", sort=True):
        ordered = group.sort_values("target_set_id", kind="stable")
        selected = ordered.loc[ordered.predicted_continuation_advantage.idxmax()]
        truth = ordered.continuation_advantage_mean.to_numpy(dtype=float)
        best = float(truth.max())
        fallback = float(ordered.loc[ordered.is_fallback.astype(bool), "continuation_advantage_mean"].iloc[0])
        rows.append((float(selected.continuation_advantage_mean) - fallback,
                     float(selected.continuation_advantage_mean) >= best - margin,
                     best - float(selected.continuation_advantage_mean)))
    values = np.asarray(rows, dtype=float)
    return {
        "selected_lift": float(values[:, 0].mean()),
        "near_best_hit_rate": float(values[:, 1].mean()),
        "top1_regret": float(values[:, 2].mean()),
    }


def optimize(
    seed: int,
    train_ids: list[str],
    validation_ids: list[str] | None,
    samples,
    frames,
    full_frame,
    protocol,
    device,
    *,
    epochs: int,
    shuffle_salt: int,
    label: str,
):
    train_frame = full_frame[full_frame.state_id.isin(train_ids)]
    transform = phase6n.fit_feature_transform(train_frame)
    contract = fit_fold_contract(train_frame)
    model = initialize_model(seed, transform, protocol, device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(protocol["training"]["learning_rate"]),
        weight_decay=float(protocol["training"]["weight_decay"]),
    )
    history = []
    batch_size = int(protocol["training"]["state_groups_per_batch"])
    first_batch = False
    for epoch in range(1, epochs + 1):
        model.train()
        rng = np.random.default_rng(seed + epoch * 1_000_003 + shuffle_salt)
        order = [train_ids[index] for index in rng.permutation(len(train_ids))]
        collected: dict[str, list[float]] = {}
        for start in range(0, len(order), batch_size):
            state_ids = order[start:start + batch_size]
            packed = phase6n.build_batch(
                state_ids, samples, frames, transform, device
            )
            optimizer.zero_grad(set_to_none=True)
            losses = batch_loss(model, packed, state_ids, contract, protocol)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                float(protocol["training"]["gradient_norm_clip"]),
            )
            optimizer.step()
            if not first_batch:
                print(json.dumps({
                    "event": "phase6o_first_batch",
                    "label": label,
                    "seed": seed,
                    "loss": float(losses["loss"].detach()),
                    "finite": bool(torch.isfinite(losses["loss"]).item()),
                    "states": packed["batch"].state_count,
                    "actions": packed["batch"].action_count,
                }), flush=True)
                first_batch = True
            for name, value in losses.items():
                if name not in ("pair_count", "near_count"):
                    collected.setdefault(name, []).append(float(value.detach()))
        record = {"epoch": epoch, **{
            f"training_{name}": float(np.mean(values))
            for name, values in collected.items()
        }}
        if validation_ids is not None:
            predicted = phase6n.predict(
                model, validation_ids, samples, frames, transform, protocol, device
            )
            record.update({
                f"validation_{key}": value
                for key, value in selection_metrics(predicted, contract.noise_margin).items()
            })
            record["validation_joint_loss"] = prediction_loss(
                predicted, contract, protocol
            )
        history.append(record)
        print(json.dumps({
            "event": "phase6o_epoch", "label": label, "seed": seed, **record
        }), flush=True)
    return model, transform, contract, history


def train_run(seed, held_fold, samples, frames, full_frame, protocol, device, *, epoch_cap=None):
    outer_ids = sorted(
        state_id for state_id, frame in frames.items()
        if int(frame.oof_fold.iloc[0]) != held_fold
    )
    held_ids = sorted(set(frames) - set(outer_ids))
    folds = sorted({int(frames[state_id].oof_fold.iloc[0]) for state_id in outer_ids})
    maximum = min(
        int(protocol["training"]["maximum_epochs"]),
        epoch_cap if epoch_cap is not None else int(protocol["training"]["maximum_epochs"]),
    )
    directions = []
    fold_directions = ((folds[0], folds[1]), (folds[1], folds[0]))
    for index, (train_fold, validation_fold) in enumerate(fold_directions):
        train_ids = [x for x in outer_ids if int(frames[x].oof_fold.iloc[0]) == train_fold]
        validation_ids = [x for x in outer_ids if int(frames[x].oof_fold.iloc[0]) == validation_fold]
        model, transform, contract, history = optimize(
            seed, train_ids, validation_ids, samples, frames, full_frame,
            protocol, device, epochs=maximum,
            shuffle_salt=held_fold * 10007 + index * 1009 + 101,
            label=f"held_{held_fold}_inner_{train_fold}_to_{validation_fold}",
        )
        directions.append({
            "train_fold": train_fold,
            "validation_fold": validation_fold,
            "contract": contract.to_dict(),
            "history": history,
        })
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    selection = []
    for epoch in range(1, maximum + 1):
        records = [direction["history"][epoch - 1] for direction in directions]
        row = {
            "epoch": epoch,
            "mean_selected_lift": float(np.mean([x["validation_selected_lift"] for x in records])),
            "mean_near_best_hit_rate": float(np.mean([x["validation_near_best_hit_rate"] for x in records])),
            "mean_top1_regret": float(np.mean([x["validation_top1_regret"] for x in records])),
            "mean_joint_loss": float(np.mean([x["validation_joint_loss"] for x in records])),
        }
        selection.append(row)
    chosen = max(selection, key=lambda row: (
        row["mean_selected_lift"], row["mean_near_best_hit_rate"],
        -row["mean_top1_regret"], -row["mean_joint_loss"], -row["epoch"]
    ))
    selected_epoch = int(chosen["epoch"])
    model, transform, outer_contract, outer_history = optimize(
        seed, outer_ids, None, samples, frames, full_frame, protocol, device,
        epochs=selected_epoch, shuffle_salt=held_fold * 10007 + 303,
        label=f"held_{held_fold}_outer_refit",
    )
    held = phase6n.predict(
        model, held_ids, samples, frames, transform, protocol, device
    )
    return model, transform, outer_contract, held, {
        "directions": directions,
        "selection": selection,
        "selected_epoch": selected_epoch,
        "selected_metrics": chosen,
        "outer_contract": outer_contract.to_dict(),
        "outer_history": outer_history,
    }


def run_paths(seed: int, held_fold: int, root: Path = OUT):
    stem = root / "oof" / f"seed_{seed}" / f"fold_{held_fold}"
    return Path(f"{stem}.pt"), Path(f"{stem}.parquet"), Path(f"{stem}.json")


def trainable_state(model) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if not name.startswith("state_encoder.")
    }


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


def scope_summary(frame: pd.DataFrame, bootstrap: dict) -> dict:
    states = metrics.ranking_state_metrics(frame, "ensemble_advantage_mean")
    lower, upper = metrics.grouped_bootstrap_interval(
        states, "selected_lift", seed=int(bootstrap["seed"]),
        resamples=int(bootstrap["resamples"]),
    )
    return {
        "states": int(states.state_id.nunique()),
        "candidates": len(frame),
        "selected_lift": float(states.selected_lift.mean()),
        "selected_lift_lcb": lower,
        "selected_lift_ucb": upper,
        "selection_regret": float(states.selection_regret.mean()),
        "top1_agreement": float(states.top1_agreement.mean()),
        "spearman": float(states.spearman.mean()),
        "pairwise_accuracy": float(states.pairwise_accuracy.mean()),
        "ndcg_at_1": float(states.ndcg_at_1.mean()),
    }


def summarize(predictions: pd.DataFrame, protocol: dict) -> dict:
    key = ["state_id", "target_set_id"]
    seeds = sorted(predictions.training_seed.unique())
    if len(seeds) != 3 or not predictions.groupby(key).training_seed.nunique().eq(3).all():
        raise RuntimeError("Phase 6O OOF seed ensemble is incomplete")
    first = predictions[predictions.training_seed.eq(seeds[0])].sort_values(key).reset_index(drop=True)
    means = predictions.groupby(key, sort=True).agg(
        ensemble_advantage_mean=("predicted_continuation_advantage", "mean"),
        ensemble_advantage_std=("predicted_continuation_advantage", lambda x: float(np.std(x, ddof=0))),
        ensemble_beats_fallback_logit=("predicted_beats_fallback_logit", "mean"),
        ensemble_beats_fallback_probability_raw=("predicted_beats_fallback_probability_raw", "mean"),
        supported=("supported", "all"),
    ).reset_index()
    drop = [
        "training_seed", "held_fold", "selected_epoch",
        "predicted_continuation_advantage", "predicted_beats_fallback_logit",
        "predicted_beats_fallback_probability_raw", "supported",
    ]
    ensemble = first.drop(columns=[x for x in drop if x in first]).merge(
        means, on=key, validate="one_to_one"
    )
    metrics.atomic_parquet(ensemble, OUT / "ensemble_oof.parquet")
    states = metrics.ranking_state_metrics(ensemble, "ensemble_advantage_mean")
    metrics.atomic_parquet(states, OUT / "state_metrics.parquet")
    summary = {
        "schema": "phase6o-top-utility-training-summary-v1",
        "status": "COMPLETE",
        "family": FAMILY,
        "expanded": scope_summary(ensemble, protocol["bootstrap"]),
        "common_original_288": scope_summary(
            ensemble[ensemble.phase6n_data_origin.eq("ORIGINAL_PHASE6J_CAUR")],
            protocol["bootstrap"],
        ),
        "historical_score_calls": 0,
        "r13_accessed": False,
        "r14_accessed": False,
    }
    metrics.atomic_json(summary, OUT / "training_summary.json")
    REPORT.write_text(f"""# Phase 6O Top-Utility 训练报告

状态：**COMPLETE — awaiting preregistered OOF quality gate**。

Route B 完成 3 seeds × 3 whole-instance outer folds。每个 run 使用双向 inner validation 选择 epoch，再在两个 outer-training folds 上重训。整个 Phase 6F encoder 固定为 eval 且没有可训练参数；只有 candidate pooler、fallback fusion、continuation head 和 beats-fallback head 更新。

Expanded selected lift 为 {summary['expanded']['selected_lift']:.8f}，95% grouped LCB 为 {summary['expanded']['selected_lift_lcb']:.8f}；common original selected lift 为 {summary['common_original_288']['selected_lift']:.8f}，selection regret 为 {summary['common_original_288']['selection_regret']:.8f}。正式 OOF gate 由后续独立审计判定。

historical scorer calls 为 0；未访问 R13/R14。
""")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--max-new-runs", type=int)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    protocol = validate_protocol()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Phase 6O CUDA training requested but CUDA is unavailable")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(4)
    frame = attach_candidate_noise(pd.read_parquet(SOURCE), pd.read_parquet(RAW))
    samples = phase6n.load_samples()
    frames = phase6n.state_frames(frame)
    root = OUT / "smoke" if args.smoke else OUT
    protocol_sha256 = digest(PROTOCOL)
    predictions = []
    completed = new_runs = 0
    started = time.perf_counter()
    run_limit = 1 if args.smoke and args.max_new_runs is None else args.max_new_runs
    print(json.dumps({
        "event": "phase6o_training_start", "device": str(device),
        "states": len(frames), "candidates": len(frame), "smoke": args.smoke,
        "historical_score_calls": 0, "r13_accessed": False, "r14_accessed": False,
    }), flush=True)
    stop = False
    for seed in protocol["training"]["seeds"]:
        for held_fold in range(3):
            paths = run_paths(int(seed), held_fold, root)
            if not args.smoke and valid_run(paths, protocol_sha256):
                predictions.append(pd.read_parquet(paths[1]))
                completed += 1
                continue
            if run_limit is not None and new_runs >= run_limit:
                stop = True
                break
            run_started = time.perf_counter()
            model, transform, contract, held, history = train_run(
                int(seed), held_fold, samples, frames, frame, protocol, device,
                epoch_cap=1 if args.smoke else None,
            )
            held["model_family"] = FAMILY
            held["training_seed"] = int(seed)
            held["held_fold"] = held_fold
            held["selected_epoch"] = history["selected_epoch"]
            checkpoint_path, prediction_path, record_path = paths
            metrics.save_checkpoint({
                "schema": "phase6o-top-utility-oof-checkpoint-v1",
                "model_family": FAMILY,
                "training_seed": int(seed),
                "held_fold": held_fold,
                "training_protocol_sha256": protocol_sha256,
                "base_checkpoint_sha256": protocol["base_checkpoint"]["sha256"],
                "feature_transform": transform.to_dict(),
                "fold_contract": contract.to_dict(),
                "trainable_model_state": trainable_state(model),
            }, checkpoint_path)
            metrics.atomic_parquet(held, prediction_path)
            record = {
                "schema": "phase6o-top-utility-oof-run-v1",
                "status": "COMPLETE",
                "model_family": FAMILY,
                "training_seed": int(seed),
                "held_fold": held_fold,
                "selected_epoch": history["selected_epoch"],
                "history": history,
                "runtime_seconds": time.perf_counter() - run_started,
                "training_protocol_sha256": protocol_sha256,
                "checkpoint_sha256": digest(checkpoint_path),
                "predictions_sha256": digest(prediction_path),
                "historical_score_calls": 0,
                "r13_accessed": False,
                "r14_accessed": False,
            }
            metrics.atomic_json(record, record_path)
            predictions.append(held)
            completed += 1
            new_runs += 1
            progress = {
                "schema": "phase6o-top-utility-training-progress-v1",
                "status": "SMOKE_RUNNING" if args.smoke else "RUNNING",
                "runs_complete": completed,
                "runs_expected": 9,
                "new_runs_this_process": new_runs,
                "last_seed": int(seed),
                "last_held_fold": held_fold,
                "last_selected_epoch": history["selected_epoch"],
                "last_run_seconds": record["runtime_seconds"],
                "elapsed_seconds": time.perf_counter() - started,
                "historical_score_calls": 0,
                "r13_accessed": False,
                "r14_accessed": False,
            }
            metrics.atomic_json(progress, root / "progress.json")
        if stop:
            break
    if args.smoke:
        progress = load_json(root / "progress.json")
        progress["status"] = "SMOKE_COMPLETE"
        metrics.atomic_json(progress, root / "progress.json")
        print(json.dumps(progress, indent=2, sort_keys=True))
        return
    if completed != 9:
        raise SystemExit(2)
    summary = summarize(pd.concat(predictions, ignore_index=True), protocol)
    progress = {
        "schema": "phase6o-top-utility-training-progress-v1",
        "status": "COMPLETE",
        "runs_complete": 9,
        "runs_expected": 9,
        "new_runs_this_process": new_runs,
        "elapsed_seconds": time.perf_counter() - started,
        "summary_sha256": digest(OUT / "training_summary.json"),
        "historical_score_calls": 0,
        "r13_accessed": False,
        "r14_accessed": False,
    }
    metrics.atomic_json(progress, OUT / "progress.json")
    print(json.dumps({"progress": progress, "summary": summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
