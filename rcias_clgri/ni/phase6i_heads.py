"""Frozen-embedding utility heads and losses for Phase 6I-MR."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Phase6IObjective:
    pairwise_weight: float = 1.0
    listwise_weight: float = 0.5
    huber_weight: float = 0.5
    positive_weight: float = 0.1
    pair_gap_scale: float = 1.0
    pair_weight_min: float = 0.25
    pair_weight_max: float = 4.0
    pair_margin_min: float = 0.05
    pair_margin_max: float = 1.0
    listwise_temperature: float = 0.01
    huber_delta: float = 0.1
    positive_class_weight: float = 1.0

    def __post_init__(self) -> None:
        if min(
            self.pairwise_weight,
            self.listwise_weight,
            self.huber_weight,
            self.positive_weight,
        ) < 0:
            raise ValueError("loss weights must be non-negative")
        if self.pair_gap_scale <= 0 or self.listwise_temperature <= 0:
            raise ValueError("loss scaling constants must be positive")
        if self.huber_delta <= 0 or self.positive_class_weight <= 0:
            raise ValueError("Huber delta and positive class weight must be positive")
        if not 0 < self.pair_weight_min <= self.pair_weight_max:
            raise ValueError("invalid pair weight bounds")
        if not 0 < self.pair_margin_min <= self.pair_margin_max:
            raise ValueError("invalid pair margin bounds")


class ImmediateUtilityHead(nn.Module):
    """U1: the Phase 6F utility-head shape, trained on frozen embeddings."""

    def __init__(self, embedding_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, 1),
        )

    def forward(
        self, embeddings: torch.Tensor, context: torch.Tensor | None = None
    ) -> torch.Tensor:
        del context
        return self.network(embeddings).squeeze(-1)


class RegimeConditionedUtilityHead(nn.Module):
    """U2: compact context-gated residual utility head."""

    def __init__(
        self,
        embedding_dim: int = 128,
        context_dim: int = 19,
        context_hidden_dim: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, context_hidden_dim),
            nn.GELU(),
        )
        self.context_gate = nn.Linear(context_hidden_dim, embedding_dim)
        self.context_residual = nn.Linear(context_hidden_dim, embedding_dim)
        self.utility_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, 1),
        )

    def forward(
        self, embeddings: torch.Tensor, context: torch.Tensor | None = None
    ) -> torch.Tensor:
        if context is None:
            raise ValueError("U2 requires normalized regime context")
        encoded = self.context_encoder(context)
        gate = torch.sigmoid(self.context_gate(encoded))
        residual = torch.tanh(self.context_residual(encoded))
        fused = embeddings + gate * residual
        return self.utility_head(fused).squeeze(-1)


def build_phase6i_head(family: str, *, dropout: float = 0.1) -> nn.Module:
    if family == "U1":
        return ImmediateUtilityHead(dropout=dropout)
    if family in {"U2", "U2_H_CONTINUATION"}:
        return RegimeConditionedUtilityHead(dropout=dropout)
    raise ValueError(f"unknown Phase 6I head family: {family}")


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def phase6i_state_loss(
    predictions: torch.Tensor,
    raw_targets: torch.Tensor,
    normalized_targets: torch.Tensor,
    positive_labels: torch.Tensor,
    mask: torch.Tensor,
    objective: Phase6IObjective,
) -> dict[str, torch.Tensor]:
    """Return state-balanced ranking, regression, and sign losses.

    Tensors use ``[state, padded_action]`` layout. Every component is first
    averaged within state and then across states, so wider legacy banks cannot
    dominate the four-role live-state bank.
    """
    if not (
        predictions.shape
        == raw_targets.shape
        == normalized_targets.shape
        == positive_labels.shape
        == mask.shape
    ):
        raise ValueError("all Phase 6I loss tensors must share one shape")
    if predictions.ndim != 2 or not bool(mask.any(dim=1).all()):
        raise ValueError("each padded state must contain at least one action")

    pair_mask = mask.unsqueeze(2) & mask.unsqueeze(1)
    raw_gap = raw_targets.unsqueeze(2) - raw_targets.unsqueeze(1)
    better = pair_mask & raw_gap.gt(1e-12)
    predicted_gap = predictions.unsqueeze(2) - predictions.unsqueeze(1)
    scaled_gap = raw_gap.abs() / objective.pair_gap_scale
    pair_weights = scaled_gap.clamp(
        objective.pair_weight_min, objective.pair_weight_max
    )
    margins = scaled_gap.clamp(
        objective.pair_margin_min, objective.pair_margin_max
    )
    pair_terms = pair_weights * F.softplus(margins - predicted_gap)
    pair_count = better.sum(dim=(1, 2))
    pair_per_state = (pair_terms * better).sum(dim=(1, 2)) / pair_count.clamp_min(1)
    comparable_states = pair_count.gt(0)
    ranking = (
        pair_per_state[comparable_states].mean()
        if bool(comparable_states.any())
        else predictions.sum() * 0.0
    )

    masked_prediction = predictions.masked_fill(~mask, -torch.inf)
    masked_target = (raw_targets / objective.listwise_temperature).masked_fill(
        ~mask, -torch.inf
    )
    target_distribution = torch.softmax(masked_target, dim=1)
    log_distribution = torch.log_softmax(masked_prediction, dim=1)
    listwise_per_state = -(
        target_distribution.masked_fill(~mask, 0.0)
        * log_distribution.masked_fill(~mask, 0.0)
    ).sum(dim=1)
    listwise = listwise_per_state.mean()

    huber_elements = F.huber_loss(
        predictions,
        normalized_targets,
        delta=objective.huber_delta,
        reduction="none",
    )
    state_width = mask.sum(dim=1).clamp_min(1)
    huber = ((huber_elements * mask).sum(dim=1) / state_width).mean()

    positive_elements = F.binary_cross_entropy_with_logits(
        predictions,
        positive_labels.to(predictions.dtype),
        pos_weight=torch.as_tensor(
            objective.positive_class_weight,
            dtype=predictions.dtype,
            device=predictions.device,
        ),
        reduction="none",
    )
    positive = ((positive_elements * mask).sum(dim=1) / state_width).mean()
    total = (
        objective.pairwise_weight * ranking
        + objective.listwise_weight * listwise
        + objective.huber_weight * huber
        + objective.positive_weight * positive
    )
    return {
        "loss": total,
        "pairwise_loss": ranking,
        "listwise_loss": listwise,
        "huber_loss": huber,
        "positive_consistency_loss": positive,
        "pair_count": pair_count.sum(),
    }
