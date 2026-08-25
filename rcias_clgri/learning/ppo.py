"""Numerically stable joint-log-probability PPO objective."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class PPOLossOutput:
    total_loss: torch.Tensor
    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    entropy: torch.Tensor
    approx_kl: torch.Tensor
    clip_fraction: torch.Tensor
    ratio: torch.Tensor
    explained_variance: torch.Tensor


def explained_variance(values: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
    variance = torch.var(returns, unbiased=False)
    if float(variance.detach()) < 1e-12:
        return returns.new_zeros(())
    return 1.0 - torch.var(returns - values, unbiased=False) / variance


def clipped_ppo_loss(
    *,
    new_joint_log_prob: torch.Tensor,
    old_joint_log_prob: torch.Tensor,
    advantages: torch.Tensor,
    values: torch.Tensor,
    returns: torch.Tensor,
    entropy: torch.Tensor,
    clip_epsilon: float = 0.2,
    value_coefficient: float = 0.5,
    entropy_coefficient: float = 0.01,
) -> PPOLossOutput:
    """Use one ratio built from the sum of O/M/W/F stage log probabilities."""

    log_ratio = new_joint_log_prob - old_joint_log_prob
    ratio = torch.exp(log_ratio)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    policy_loss = -torch.minimum(unclipped, clipped).mean()
    value_loss = F.smooth_l1_loss(values, returns)
    mean_entropy = entropy.mean()
    total = policy_loss + value_coefficient * value_loss - entropy_coefficient * mean_entropy
    approx_kl = ((ratio - 1.0) - log_ratio).mean()
    clip_fraction = ((ratio - 1.0).abs() > clip_epsilon).to(ratio.dtype).mean()
    output = PPOLossOutput(
        total_loss=total,
        policy_loss=policy_loss,
        value_loss=value_loss,
        entropy=mean_entropy,
        approx_kl=approx_kl,
        clip_fraction=clip_fraction,
        ratio=ratio,
        explained_variance=explained_variance(values, returns),
    )
    for value in (
        output.total_loss, output.policy_loss, output.value_loss,
        output.entropy, output.approx_kl, output.clip_fraction,
    ):
        if not torch.isfinite(value):
            raise FloatingPointError("non-finite PPO loss component")
    return output
