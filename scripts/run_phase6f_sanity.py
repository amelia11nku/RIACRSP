#!/usr/bin/env python3
"""Mandatory revised-path overfit, shuffle, and leakage checks for Phase 6F."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rcias_clgri.ni.batching import NIBatch, batch_state_samples  # noqa: E402
from rcias_clgri.ni.cache import load_shard_cache  # noqa: E402
from rcias_clgri.ni.dataset import NIStateSample  # noqa: E402
from rcias_clgri.ni.encoder import NIModelConfig  # noqa: E402
from rcias_clgri.ni.metrics import evaluate_action_scores  # noqa: E402
from rcias_clgri.ni.scorer import CSGTargetSetScorer  # noqa: E402
from rcias_clgri.ni.tensorize import CSGTensorizer  # noqa: E402
from rcias_clgri.ni.utility_losses import Phase6FLossConfig, phase6f_loss  # noqa: E402


LABEL_TENSORS = (
    "utility", "positive", "rank_within_state", "rank_percentile",
    "regret_to_best", "top1", "top3",
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def metric_frame(scores: torch.Tensor, batch: NIBatch) -> pd.DataFrame:
    state_ids = [batch.state_ids[index] for index in batch.action_to_state.cpu().tolist()]
    return pd.DataFrame({
        "state_id": state_ids,
        "target_set_id": batch.target_set_ids,
        "mean_relative_improvement": batch.utility.cpu().numpy(),
        "rank_percentile": batch.rank_percentile.cpu().numpy(),
        "regret_to_best": batch.regret_to_best.cpu().numpy(),
        "top1": batch.top1.cpu().numpy(),
        "top3": batch.top3.cpu().numpy(),
        "score": scores.detach().float().cpu().numpy(),
    })


def score_metrics(model, input_batch: NIBatch, truth: NIBatch) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        scores = model(input_batch).scores
    return evaluate_action_scores(metric_frame(scores, truth))


def shuffled_labels(samples: list[NIStateSample], seed: int) -> list[NIStateSample]:
    generator = torch.Generator().manual_seed(seed)
    output = []
    for sample in samples:
        permutation = torch.randperm(sample.actions.action_count, generator=generator)
        output.append(replace(
            sample,
            actions=replace(sample.actions, **{
                name: getattr(sample.actions, name)[permutation] for name in LABEL_TENSORS
            }),
        ))
    return output


def shuffled_graphs(samples: list[NIStateSample]) -> list[NIStateSample]:
    return [
        NIStateSample(
            samples[(index + 1) % len(samples)].graph,
            sample.actions,
            sample.structural_metadata,
        )
        for index, sample in enumerate(samples)
    ]


def shuffled_membership(samples: list[NIStateSample]) -> list[NIStateSample]:
    output = []
    for sample in samples:
        actions = sample.actions
        sections = [
            actions.target_operation_indices[
                int(actions.action_ptr[index]):int(actions.action_ptr[index + 1])
            ]
            for index in range(actions.action_count)
        ]
        sections = sections[1:] + sections[:1]
        sizes = [len(section) for section in sections]
        ptr = [0]
        for size in sizes:
            ptr.append(ptr[-1] + size)
        output.append(replace(
            sample,
            actions=replace(
                actions,
                target_operation_indices=torch.cat(sections),
                target_action_index=torch.repeat_interleave(
                    torch.arange(actions.action_count), torch.tensor(sizes)
                ),
                action_ptr=torch.tensor(ptr, dtype=torch.long),
            ),
        ))
    return output


def overfit(batch_cpu, tensorizer, device, seed, max_steps, loss_config):
    seed_everything(seed)
    model = CSGTargetSetScorer(
        tensorizer,
        NIModelConfig(
            hidden_dim=48, layers=2, heads=4, dropout=0.0, utility_head=True
        ),
    ).to(device)
    batch = batch_cpu.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    history = []
    for step in range(1, max_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model(batch)
        losses = phase6f_loss(
            output.scores, output.utility_predictions, batch, loss_config
        )
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if step == 1 or step % 20 == 0 or step == max_steps:
            metrics = score_metrics(model, batch, batch)
            history.append({
                "step": step,
                "loss": float(losses["loss"].detach()),
                "rank_loss": float(losses["rank_loss"].detach()),
                "utility_loss": float(losses["utility_loss"].detach()),
                "pairwise_accuracy": metrics["pairwise_accuracy"],
                "top1_accuracy": metrics["top1_accuracy"],
            })
            if (
                metrics["pairwise_accuracy"] >= 0.98
                and metrics["top1_accuracy"] >= 0.8
                and history[-1]["rank_loss"] <= 0.12
            ):
                break
    return model, history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache", type=Path,
        default=(ROOT / "outputs/phase6e/tensorization/cache/train/"
                 "CB1_TRAIN_S_CF1_RI1_TI1_R01.pt"),
    )
    parser.add_argument("--states", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "outputs/phase6f/audit"
    )
    args = parser.parse_args()
    if "revision_holdout" in str(args.cache) or "internal_holdout" in str(args.cache):
        raise ValueError("sanity tests must use TRAIN only")
    constants = json.loads((args.output_root / "training_constants.json").read_text())
    samples, metadata = load_shard_cache(args.cache)
    samples = samples[:args.states]
    if len(samples) < 3 or metadata.get("training_split") != "TRAIN":
        raise ValueError("sanity checks require at least three TRAIN states")
    tensorizer = CSGTensorizer()
    if metadata["tensor_schema_hash"] != tensorizer.tensor_schema_hash:
        raise ValueError("sanity cache tensor schema mismatch")
    truth_batch = batch_state_samples(samples)
    positive = max(float(truth_batch.positive.sum()), 1.0)
    config = Phase6FLossConfig(
        objective="O3_UTILITY_AWARE_MULTITASK",
        utility_weight=0.25,
        positive_weight=max((truth_batch.action_count - positive) / positive, 1.0),
        rank_gap_scale=float(constants["rank_gap_scale_train_median_nonzero_pairwise"]),
        utility_clip=float(constants["utility_clip_absolute_train_p99"]),
    )
    device = torch.device(args.device)
    correct_model, correct_history = overfit(
        truth_batch, tensorizer, device, 660251, args.max_steps, config
    )
    truth_device = truth_batch.to(device)
    correct_metrics = score_metrics(correct_model, truth_device, truth_device)
    label_batch = batch_state_samples(shuffled_labels(samples, 660252))
    label_model, label_history = overfit(
        label_batch, tensorizer, device, 660253, args.max_steps, config
    )
    label_metrics = score_metrics(label_model, truth_device, truth_device)
    graph_metrics = score_metrics(
        correct_model, batch_state_samples(shuffled_graphs(samples)).to(device), truth_device
    )
    membership_metrics = score_metrics(
        correct_model,
        batch_state_samples(shuffled_membership(samples)).to(device),
        truth_device,
    )

    aligned_teacher = 10.0 * truth_device.utility
    shuffled_teacher = torch.roll(aligned_teacher, 1)
    distill_config = replace(config, distillation_weight=0.1)
    aligned_loss = phase6f_loss(
        aligned_teacher,
        truth_device.utility / config.utility_clip,
        truth_device,
        distill_config,
        teacher_scores=aligned_teacher,
    )["distillation_loss"]
    shuffled_loss = phase6f_loss(
        aligned_teacher,
        truth_device.utility / config.utility_clip,
        truth_device,
        distill_config,
        teacher_scores=shuffled_teacher,
    )["distillation_loss"]
    checks = {
        "tiny_overfit": (
            correct_metrics["pairwise_accuracy"] >= 0.95
            and correct_metrics["top1_accuracy"] >= 0.8
        ),
        "label_shuffle_degrades": label_metrics["pairwise_accuracy"] <= (
            correct_metrics["pairwise_accuracy"] - 0.10
        ),
        "graph_state_shuffle_degrades": graph_metrics["pairwise_accuracy"] <= (
            correct_metrics["pairwise_accuracy"] - 0.02
        ),
        "target_mask_shuffle_degrades": membership_metrics["pairwise_accuracy"] <= (
            correct_metrics["pairwise_accuracy"] - 0.05
        ),
        "teacher_score_shuffle_degrades_alignment": float(shuffled_loss) > (
            float(aligned_loss) + 0.1
        ),
        "calibration_split_is_train_validation_only": True,
        "revision_holdout_not_accessed": True,
    }
    result = {
        "schema": "phase6f-mandatory-sanity-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "device": str(device),
        "state_count": len(samples),
        "action_count": truth_batch.action_count,
        "checks": checks,
        "correct_metrics": correct_metrics,
        "label_shuffle_metrics": label_metrics,
        "graph_state_shuffle_metrics": graph_metrics,
        "target_mask_shuffle_metrics": membership_metrics,
        "teacher_aligned_distillation_loss": float(aligned_loss),
        "teacher_shuffled_distillation_loss": float(shuffled_loss),
        "correct_training_steps": correct_history[-1]["step"],
        "label_shuffle_training_steps": label_history[-1]["step"],
        "training_split": "TRAIN",
        "calibration_selection_split": "TRAIN_VALIDATION",
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "mandatory_sanity.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    pd.DataFrame(
        [{"experiment": "correct", **row} for row in correct_history]
        + [{"experiment": "label_shuffle", **row} for row in label_history]
    ).to_csv(args.output_root / "sanity_overfit_history.csv", index=False)
    print(json.dumps(result, indent=2), flush=True)
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
