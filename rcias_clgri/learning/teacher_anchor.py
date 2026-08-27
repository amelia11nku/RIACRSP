"""Frozen-BC distribution anchor for stable policy fine-tuning."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from rcias_clgri.env.insertion_decoder import Action
from rcias_clgri.nn import GraphTensor, RCIASNeuralModel

STAGES = ("operation", "island", "w", "f")


def freeze_teacher(model: RCIASNeuralModel) -> RCIASNeuralModel:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def teacher_stage_kl(
    current: RCIASNeuralModel,
    teacher: RCIASNeuralModel,
    graph: GraphTensor,
    current_hidden: Mapping[str, torch.Tensor],
    action: Action,
) -> dict[str, torch.Tensor]:
    """Compute KL(current || frozen teacher) on identical hard-masked paths."""
    with torch.no_grad():
        teacher_hidden = teacher.encode(graph)
        teacher_distributions = teacher.policy.action_distributions(
            graph, teacher_hidden, action
        )
    current_distributions = current.policy.action_distributions(
        graph, current_hidden, action
    )
    result = {}
    for stage in STAGES:
        current_distribution = current_distributions[stage]
        teacher_distribution = teacher_distributions[stage]
        if current_distribution.candidate_ids != teacher_distribution.candidate_ids:
            raise RuntimeError(f"teacher and current hard masks differ at {stage}")
        if len(current_distribution.candidate_ids) <= 1:
            result[stage] = current_distribution.logits.sum() * 0.0
            continue
        current_log = torch.log_softmax(current_distribution.logits, dim=0)
        teacher_log = torch.log_softmax(teacher_distribution.logits, dim=0)
        result[stage] = (current_log.exp() * (current_log - teacher_log)).sum()
    return result


def active_stage_mean(
    values: Mapping[str, torch.Tensor], candidate_counts: Mapping[str, int],
    coefficients: Mapping[str, float] | None = None,
) -> torch.Tensor:
    coefficients = coefficients or {stage: 1.0 for stage in STAGES}
    active = [
        (float(coefficients[stage]), values[stage])
        for stage in STAGES if candidate_counts[stage] > 1
    ]
    if not active:
        return next(iter(values.values())).new_zeros(())
    denominator = sum(weight for weight, _ in active)
    if denominator <= 0:
        raise ValueError("active stage coefficients must sum to a positive value")
    return sum(weight * value for weight, value in active) / denominator


def anchor_beta(config: Mapping[str, object], update: int) -> float:
    if not bool(config.get("enabled", False)):
        return 0.0
    initial = float(config["beta_initial"])
    minimum = float(config.get("beta_minimum", 0.0))
    horizon = max(1, int(config.get("anchor_updates", 1)))
    fraction = max(0.0, 1.0 - update / horizon)
    schedule = str(config.get("schedule", "constant"))
    if schedule == "constant":
        return initial
    if schedule == "linear_decay":
        return initial * fraction
    if schedule == "floor_decay":
        return minimum + (initial - minimum) * fraction
    raise ValueError(f"unknown teacher anchor schedule: {schedule}")
