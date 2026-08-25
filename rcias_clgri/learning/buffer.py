"""CPU-owned rollout transitions and generalized advantage estimation."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterator, Mapping

import torch

from rcias_clgri.env.insertion_decoder import Action
from rcias_clgri.nn.tensorizer import GraphTensor


@dataclass(frozen=True)
class RolloutTransition:
    graph: GraphTensor
    action: Action
    old_joint_log_prob: float
    old_stage_log_probs: Mapping[str, float]
    old_value: float
    reward: float
    done: bool
    instance_id: str
    step_index: int

    def __post_init__(self) -> None:
        tensors = list(self.graph.node_features.values())
        tensors.extend(edge.features for edge in self.graph.edges.values())
        if any(tensor.device.type != "cpu" or tensor.requires_grad for tensor in tensors):
            raise ValueError("rollout graphs must be detached CPU observations")


@dataclass(frozen=True)
class AdvantageBatch:
    advantages: torch.Tensor
    returns: torch.Tensor


def generalized_advantage_estimate(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    *,
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
) -> AdvantageBatch:
    """Compute finite-horizon GAE with terminal value fixed to zero."""

    if not (rewards.ndim == values.ndim == dones.ndim == 1):
        raise ValueError("rewards, values, and dones must be one-dimensional")
    if not (len(rewards) == len(values) == len(dones)):
        raise ValueError("rewards, values, and dones must have equal length")
    advantages = torch.zeros_like(rewards)
    gae = rewards.new_zeros(())
    for index in range(len(rewards) - 1, -1, -1):
        nonterminal = 1.0 - dones[index].to(rewards.dtype)
        next_value = values[index + 1] if index + 1 < len(values) else values.new_zeros(())
        delta = rewards[index] + gamma * next_value * nonterminal - values[index]
        gae = delta + gamma * gae_lambda * nonterminal * gae
        advantages[index] = gae
    return AdvantageBatch(advantages=advantages, returns=advantages + values)


def normalize_advantages(
    advantages: torch.Tensor, epsilon: float = 1e-8,
) -> torch.Tensor:
    if advantages.numel() <= 1:
        return torch.zeros_like(advantages)
    standard_deviation = advantages.std(unbiased=False)
    if float(standard_deviation) < epsilon:
        return advantages - advantages.mean()
    return (advantages - advantages.mean()) / standard_deviation


class RolloutBuffer:
    def __init__(self) -> None:
        self.transitions: list[RolloutTransition] = []
        self.advantages: torch.Tensor | None = None
        self.returns: torch.Tensor | None = None

    def __len__(self) -> int:
        return len(self.transitions)

    def add(self, transition: RolloutTransition) -> None:
        self.transitions.append(transition)
        self.advantages = None
        self.returns = None

    def clear(self) -> None:
        self.transitions.clear()
        self.advantages = None
        self.returns = None

    def compute_advantages(
        self, *, gamma: float = 1.0, gae_lambda: float = 0.95,
    ) -> AdvantageBatch:
        rewards = torch.tensor(
            [transition.reward for transition in self.transitions], dtype=torch.float32
        )
        values = torch.tensor(
            [transition.old_value for transition in self.transitions], dtype=torch.float32
        )
        dones = torch.tensor(
            [transition.done for transition in self.transitions], dtype=torch.bool
        )
        batch = generalized_advantage_estimate(
            rewards, values, dones, gamma=gamma, gae_lambda=gae_lambda
        )
        self.advantages = normalize_advantages(batch.advantages)
        self.returns = batch.returns
        return AdvantageBatch(self.advantages, self.returns)

    def minibatches(
        self, batch_size: int, *, seed: int,
    ) -> Iterator[list[int]]:
        if self.advantages is None or self.returns is None:
            raise RuntimeError("compute advantages before requesting minibatches")
        indices = list(range(len(self.transitions)))
        random.Random(seed).shuffle(indices)
        for start in range(0, len(indices), batch_size):
            yield indices[start:start + batch_size]
